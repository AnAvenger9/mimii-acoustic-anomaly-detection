"""Fetch and verify the original MIMII release using its saved Zenodo metadata.

Run from the repository root with Python 3.9+ (standard library only). New archives are temporary;
existing fan archives and recordings are preserved. Interrupted work is resumable.
"""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from urllib.request import Request, urlopen
from zipfile import ZipFile
import zlib


GIB = 2**30
SNR_NAMES = {"-6": "minus6db", "0": "0db", "6": "plus6db"}


def report(message):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


def checksum(path, algorithm="md5"):
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def crc32(path):
    crc = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            crc = zlib.crc32(chunk, crc)
    return crc


def write_json(path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def download_range(url, path, start, end, total):
    """Download one bounded range, rejecting servers that ignore Range."""
    for attempt in range(8):
        try:
            request = Request(url, headers={"Range": f"bytes={start}-{end}",
                                            "User-Agent": "mimii-research/0.1"})
            with urlopen(request, timeout=60) as response:
                expected = f"bytes {start}-{end}/{total}"
                if response.status != 206 or response.headers.get("Content-Range") != expected:
                    raise ValueError("Server did not return the requested byte range")
                with path.open("wb") as output:
                    shutil.copyfileobj(response, output, 2**20)
            if path.stat().st_size != end - start + 1:
                raise ValueError("Incomplete byte range")
            return
        except Exception as error:
            if attempt == 7:
                raise
            report(f"Retrying range {start}-{end}: {error}")
            time.sleep(min(60, 10 * (attempt + 1)))


def download(entry, archive):
    """Use four range requests at a time, retaining a resumable contiguous prefix."""
    total = entry["size"]
    start_time = time.monotonic()
    initial = archive.stat().st_size if archive.exists() else 0
    position = initial
    with ThreadPoolExecutor(max_workers=4) as pool:
        while position < total:
            batch = []
            for start in range(position, min(position + 4 * 64 * 2**20, total), 64 * 2**20):
                end = min(start + 64 * 2**20, total) - 1
                chunk = archive.with_name(archive.name + f".range-{start}")
                future = pool.submit(download_range, entry["links"]["self"],
                                     chunk, start, end, total)
                batch.append((future, chunk))
            pending = [f for f, _ in batch]
            while pending:
                _, pending = wait(pending, timeout=30)
                received = position + sum(p.stat().st_size for _, p in batch if p.exists())
                elapsed = time.monotonic() - start_time
                report(f"{entry['key']}: {received / total:.1%}, "
                       f"{received / GIB:.2f}/{total / GIB:.2f} GiB, "
                       f"{(received - initial) / elapsed / 2**20:.2f} MiB/s")
            for future, chunk in batch:
                future.result()
                with archive.open("ab") as output, chunk.open("rb") as source:
                    shutil.copyfileobj(source, output, 2**20)
                chunk.unlink()
            position = archive.stat().st_size


def process(entry, data):
    name = entry["key"]
    snr, _, machine = name.removesuffix(".zip").split("_")
    target = data / f"{machine}_{SNR_NAMES[snr]}"
    receipt = data / "metadata" / f"{name}.verified.json"
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved["checksum"] != entry["checksum"]:
            raise ValueError(f"Metadata changed for {name}")
        for item in saved["files"]:
            path = target / item["path"]
            if not path.is_file() or path.stat().st_size != item["bytes"]:
                raise ValueError(f"Previously verified file missing or changed: {path}")
        report(f"Already verified: {name}")
        return

    # Only files created in this dedicated temporary directory are removed.
    temporary_archive = data / ".mimii-downloads" / (name + ".part")
    existing = [p for p in (data / name, data / "archives" / name) if p.exists()]
    archive = existing[0] if existing else temporary_archive
    if not existing:
        archive.parent.mkdir(exist_ok=True)
        present = archive.stat().st_size if archive.exists() else 0
        if shutil.disk_usage(data).free < entry["size"] - present + 4 * GIB:
            raise OSError(f"Insufficient space to download {name}")
        if present != entry["size"]:
            report(f"Downloading {name}: {entry['size'] / GIB:.2f} GiB")
            download(entry, archive)
    if archive.stat().st_size != entry["size"]:
        raise ValueError(f"Wrong archive size: {archive}")
    report(f"Checking official MD5: {name}")
    algorithm, expected = entry["checksum"].split(":", 1)
    if checksum(archive, algorithm) != expected:
        raise ValueError(f"Checksum mismatch: {archive}; preserved for inspection")

    staging = data / (".extracting_" + target.name)
    root = target if target.exists() else staging
    root.mkdir(exist_ok=True)
    with ZipFile(archive) as package:
        members = [m for m in package.infolist() if not m.is_dir()]
        paths = set()
        for member in members:
            path = Path(member.filename)
            if (path.is_absolute() or ".." in path.parts or
                    not path.parts or path.parts[0] != machine or
                    member.filename in paths):
                raise ValueError(f"Unexpected archive path: {member.filename}")
            paths.add(member.filename)
            if not (root / path).resolve().is_relative_to(root.resolve()):
                raise ValueError(f"Unsafe destination: {path}")
        missing_bytes = sum(m.file_size for m in members
                            if not (root / m.filename).exists())
        if shutil.disk_usage(data).free < missing_bytes + 3 * GIB:
            raise OSError(f"Insufficient space to extract {name}: "
                          f"need {missing_bytes / GIB:.2f} GiB plus reserve")
        report(f"Validating/extracting {name}: {len(members)} files, "
               f"{sum(m.file_size for m in members) / GIB:.2f} GiB")
        counts = Counter()
        manifest = []
        for index, member in enumerate(members, 1):
            path = root / member.filename
            if path.exists():
                if path.stat().st_size != member.file_size or crc32(path) != member.CRC:
                    raise ValueError(f"Existing file differs from official archive: {path}")
            else:
                if root == target:
                    raise ValueError(f"Existing dataset incomplete; refusing to modify: {path}")
                path.parent.mkdir(parents=True, exist_ok=True)
                partial = path.with_suffix(path.suffix + ".part")
                # Reading through EOF checks the ZIP member CRC before publication.
                with package.open(member) as source, partial.open("wb") as output:
                    shutil.copyfileobj(source, output, 2**20)
                partial.replace(path)
            if path.suffix.lower() == ".wav":
                counts["/".join(Path(member.filename).parts[1:3])] += 1
            manifest.append({"path": member.filename, "bytes": member.file_size,
                             "crc32": f"{member.CRC:08x}"})
            if index % 1000 == 0:
                report(f"{name}: {index}/{len(members)} files verified")
        actual = {str(p.relative_to(root)) for p in root.rglob("*.wav")}
        expected_wavs = {m.filename for m in members if m.filename.endswith(".wav")}
        if actual != expected_wavs:
            raise ValueError(f"Unexpected WAV file inventory in {root}")
    if root == staging:
        staging.rename(target)
    write_json(receipt, {
        "archive": name, "checksum": entry["checksum"], "url": entry["links"]["self"],
        "destination": str(target), "verified_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "counts": dict(sorted(counts.items())), "files": manifest,
    })
    if archive == temporary_archive:
        archive.unlink()
    report(f"READY {target}: {sum(counts.values())} WAVs; "
           f"{shutil.disk_usage(data).free / GIB:.1f} GiB free")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    args = parser.parse_args()
    metadata = args.data / "metadata"
    lock = (metadata / "download.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (metadata / "download.pid").write_text(str(os.getpid()) + "\n")
    record = json.loads((metadata / "zenodo-3384388.json").read_text())
    if str(record["id"]) != "3384388":
        raise ValueError("Expected the original MIMII Zenodo record 3384388")
    entries = {item["key"]: item for item in record["files"]}
    # Finish new machine types first, then audit the existing fan data.
    status_path = metadata / "download-status.json"
    try:
        for machine in ("pump", "slider", "valve", "fan"):
            for snr in ("0", "6", "-6"):
                name = f"{snr}_dB_{machine}.zip"
                write_json(status_path, {"state": "running", "archive": name, "pid": os.getpid()})
                process(entries[name], args.data)
    except BaseException as error:
        write_json(status_path, {"state": "interrupted" if isinstance(error, KeyboardInterrupt)
                                else "failed", "archive": name, "error": str(error)})
        raise
    receipts = [json.loads(p.read_text()) for p in sorted(metadata.glob("*.verified.json"))]
    summary = {r["destination"]: r["counts"] for r in receipts}
    write_json(metadata / "dataset-inventory.json", summary)
    write_json(status_path, {"state": "complete", "subsets": len(receipts),
                            "wav_files": sum(sum(r["counts"].values()) for r in receipts)})
    report("COMPLETE: all 12 original MIMII machine/SNR subsets verified")


if __name__ == "__main__":
    main()
