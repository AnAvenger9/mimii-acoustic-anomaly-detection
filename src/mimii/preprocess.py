"""CLI for audited, split-isolated feature extraction on a MIMII subset."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import time

import numpy as np

from .audio import load_audio
from .config import PreprocessConfig
from .dataset import assign_reference_splits, assign_splits, discover_recordings
from .features import fit_normalizer, log_mel_spectrogram


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def extract_recording(task):
    row, config = task
    path = Path(config.data_root) / row["relative_path"]
    before = path.stat()
    waveform, metadata = load_audio(path, config)
    if metadata["silent"]:
        raise ValueError(f"Silent recording requires investigation: {path}")
    features = log_mel_spectrogram(waveform, config)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"Source recording changed during extraction: {path}")
    destination = Path("logmel") / Path(row["relative_path"]).with_suffix(".npy")
    target = Path(config.output_dir) / destination
    target.parent.mkdir(parents=True, exist_ok=True)
    np.save(target, features, allow_pickle=False)
    return {**row, **metadata, "feature_path": destination.as_posix(),
            "snr_db": config.snr_db, "selected_channel": config.channel,
            "feature_shape": list(features.shape),
            "source_size_bytes": after.st_size, "source_mtime_ns": after.st_mtime_ns}


def run(config, recordings=None, split_assigner=assign_splits):
    start = time.monotonic()
    rows = discover_recordings(config.data_root) if recordings is None else recordings
    reference_bytes = (Path(config.split_reference_manifest).read_bytes()
                       if config.split_reference_manifest else None)
    reference_rows = ([json.loads(line) for line in reference_bytes.decode().splitlines()]
                      if reference_bytes is not None else None)
    output = Path(config.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}. Choose a new output_dir.")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config.to_dict())
    write_json(output / "status.json", {"state": "running"})
    print(f"Found {len(rows)} recordings; extracting with {config.workers} workers", flush=True)
    try:
        tasks = ((row, config) for row in rows)
        records = []
        if config.workers == 1:
            results = map(extract_recording, tasks)
            for index, row in enumerate(results, 1):
                records.append(row)
                if index % 250 == 0 or index == len(rows):
                    print(f"Extracted {index}/{len(rows)} ({time.monotonic() - start:.1f}s)", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=config.workers) as executor:
                for index, row in enumerate(executor.map(extract_recording, tasks, chunksize=8), 1):
                    records.append(row)
                    if index % 250 == 0 or index == len(rows):
                        print(f"Extracted {index}/{len(rows)} ({time.monotonic() - start:.1f}s)", flush=True)

        if reference_rows is not None:
            assign_reference_splits(records, reference_rows)
        else:
            split_assigner(records, config.seed, config.validation_fraction, config.test_normal_fraction)
        machine_ids = sorted({r["machine_id"] for r in records})
        id_classes = {name: index for index, name in enumerate(machine_ids)}
        for row in records:
            row["id_class"] = id_classes[row["machine_id"]]
        manifest = "".join(json.dumps(row, sort_keys=True) + "\n" for row in records)
        (output / "manifest.jsonl").write_text(manifest)
        for split in ("train", "validation", "test"):
            (output / f"{split}.jsonl").write_text("".join(
                json.dumps(row, sort_keys=True) + "\n" for row in records if row["split"] == split))

        stats = fit_normalizer(records, output)
        np.savez(output / "normalizer.npz", **stats)
        fingerprint = hashlib.sha256(manifest.encode()).hexdigest()
        write_json(output / "normalizer_provenance.json", {
            "fit_split": "train", "fit_label": 0, "scope": "global_per_mel_band",
            "n_clips": int(stats["n_clips"]), "n_frames": int(stats["n_frames"]),
            "manifest_sha256": fingerprint, "std_floor": 1e-6,
        })
        counts = Counter((r["machine_id"], r["condition"], r["split"]) for r in records)
        summary = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "scope": {"machine_type": Path(config.data_root).name, "snr_db": config.snr_db},
            "dataset_protocol": ("official_dcase2020_development_test_partition"
                                 if all("official_partition" in r for r in records)
                                 else "custom_mimii_recording_split"),
            "recordings": len(records), "machine_ids": id_classes,
            "audio_hours": sum(r["duration_seconds"] for r in records) / 3600,
            "feature_shape": [config.n_mels, config.expected_frames],
            "ae_vectors_per_clip": config.expected_frames - config.context_frames + 1,
            "ae_vector_dimensions": config.n_mels * config.context_frames,
            "split_counts": dict(Counter(r["split"] for r in records)),
            "counts": [{"machine_id": mid, "condition": condition, "split": split, "clips": n}
                       for (mid, condition, split), n in sorted(counts.items())],
            "qa": {
                "sample_rates": sorted({r["sample_rate"] for r in records}),
                "channel_counts": sorted({r["channels"] for r in records}),
                "subtypes": sorted({r["subtype"] for r in records}),
                "silent_recordings": sum(r["silent"] for r in records),
                "recordings_with_clipped_samples": sum(r["clipped_fraction"] > 0 for r in records),
                "duplicate_selected_channel_recordings": len(records) - len({r["waveform_sha256"] for r in records}),
            },
            "manifest_sha256": fingerprint,
            "split_reference_manifest_sha256": (hashlib.sha256(reference_bytes).hexdigest()
                                                if reference_bytes is not None else None),
            "environment": {"python": platform.python_version(), **{
                name: version(name) for name in ("numpy", "scipy", "soundfile", "librosa", "matplotlib")}},
            "elapsed_seconds": round(time.monotonic() - start, 2),
        }
        write_json(output / "summary.json", summary)
        from .visualization import create_figures
        create_figures(records, config)
        write_json(output / "status.json", {"state": "complete", "manifest_sha256": fingerprint})
        print(json.dumps(summary, indent=2), flush=True)
        return summary
    except Exception as error:
        write_json(output / "status.json", {"state": "failed", "error": str(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/fan_0db.json")
    parser.add_argument("--output-dir", help="Override output_dir; must be empty or new")
    parser.add_argument("--workers", type=int, help="Override process count")
    args = parser.parse_args()
    values = PreprocessConfig.from_json(args.config).to_dict()
    if args.output_dir is not None:
        values["output_dir"] = args.output_dir
    if args.workers is not None:
        values["workers"] = args.workers
    # Keep numerical libraries from creating a thread pool in every worker.
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(key, "1")
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/mimii-numba")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mimii-matplotlib")
    run(PreprocessConfig(**values))


if __name__ == "__main__":
    main()
