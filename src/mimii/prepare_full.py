"""Extract and audit all original MIMII subsets, with aligned SNR splits.

Completed subset caches are reusable after verification. A partial subset is
never silently reused or overwritten. Combined stores reference the arrays;
they do not duplicate features or expand overlapping AE context vectors.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from .audio import load_audio
from .config import PreprocessConfig
from .dataset import discover_recordings, validate_splits
from .features import fit_normalizer, log_mel_spectrogram
from .prepare_multisnr import FRONTEND_KEYS, combine_records
from .preprocess import run as extract_subset, write_json
from .training_data import FeatureStore


SNR_NAMES = {0: "0db", -6: "minus6db", 6: "plus6db"}
PROTOCOL_KEYS = FRONTEND_KEYS + ("validation_fraction", "test_normal_fraction")


def audit_cache(root, expected_config):
    """Check every array and source stat; recompute one clip per ID/condition.

    Source waveform hashes were calculated at extraction. Stat checks detect
    source changes since then; this is not a fresh full-source cryptographic audit.
    """
    store = FeatureStore(root)
    expected = expected_config.to_dict()
    if any(store.config[key] != expected[key] for key in PROTOCOL_KEYS):
        raise ValueError(f"Frontend or split settings differ: {root}")
    if store.config["snr_db"] != expected["snr_db"]:
        raise ValueError(f"SNR differs: {root}")
    if Path(store.config["data_root"]).resolve() != Path(expected["data_root"]).resolve():
        raise ValueError(f"Source root differs: {root}")
    discovered = {r["relative_path"]: r for r in discover_recordings(expected_config.data_root)}
    if set(discovered) != {r["relative_path"] for r in store.records}:
        raise ValueError(f"Source inventory differs: {root}")
    digest = hashlib.sha256()
    size, minimum, maximum = 0, float("inf"), float("-inf")
    checked_examples = set()
    max_recompute_error = 0.0
    for row in store.records:
        source = Path(expected_config.data_root) / row["relative_path"]
        stat = source.stat()
        if (stat.st_size, stat.st_mtime_ns) != (row["source_size_bytes"], row["source_mtime_ns"]):
            raise ValueError(f"Source changed since extraction: {source}")
        if any(row[key] != discovered[row["relative_path"]][key]
               for key in ("machine_type", "machine_id", "condition", "label", "filename")):
            raise ValueError(f"Source labels differ: {source}")
        if row["snr_db"] != expected_config.snr_db or row["selected_channel"] != expected_config.channel:
            raise ValueError(f"Recording frontend metadata differs: {source}")
        target = store.root / row["feature_path"]
        x = store.load(row)
        if x.dtype != np.float32 or x.shape != (expected_config.n_mels, expected_config.expected_frames):
            raise ValueError(f"Invalid feature shape/dtype: {target}")
        digest.update(row["relative_path"].encode() + b"\0")
        digest.update(hashlib.sha256(target.read_bytes()).digest())
        size += target.stat().st_size
        minimum, maximum = min(minimum, float(x.min())), max(maximum, float(x.max()))
        example_key = (row["machine_id"], row["condition"])
        if example_key not in checked_examples:
            waveform, metadata = load_audio(source, expected_config)
            if metadata["waveform_sha256"] != row["waveform_sha256"]:
                raise ValueError(f"Example source checksum differs: {source}")
            fresh = log_mel_spectrogram(waveform, expected_config)
            error = float(np.max(np.abs(fresh - x)))
            max_recompute_error = max(max_recompute_error, error)
            np.testing.assert_allclose(x, fresh, rtol=0, atol=1e-4)
            checked_examples.add(example_key)
    for split in ("train", "validation", "test"):
        actual = [json.loads(line) for line in (store.root / f"{split}.jsonl").read_text().splitlines()]
        if actual != store.split(split):
            raise ValueError(f"Split manifest differs: {root}/{split}")
    stats = fit_normalizer(store.records, store.root)
    provenance = json.loads((store.root / "normalizer_provenance.json").read_text())
    if (provenance["manifest_sha256"] != store.manifest_sha256
            or provenance["fit_split"] != "train" or provenance["fit_label"] != 0):
        raise ValueError(f"Invalid normalizer provenance: {root}")
    with np.load(store.root / "normalizer.npz", allow_pickle=False) as saved:
        for key, value in stats.items():
            np.testing.assert_allclose(saved[key], value, rtol=0, atol=1e-6)
    return store, {
        "recordings_verified": len(store.records), "feature_bytes": size,
        "feature_files_sha256": digest.hexdigest(), "min_db": minimum, "max_db": maximum,
        "recomputed_examples": len(checked_examples), "max_recompute_absolute_error": max_recompute_error,
        "all_arrays_finite_float32": True, "source_inventory_and_stats_verified": True,
        "training_normalizer_recomputed": True, "manifest_sha256": store.manifest_sha256,
    }


def merge_machine_stores(stores, output):
    """Create globally unique paths/source keys and 16 type/ID target classes."""
    rows = []
    reference = stores[0].config
    for store in stores:
        if any(store.config[key] != reference[key] for key in PROTOCOL_KEYS):
            raise ValueError("Frontend or split settings differ across machines")
        for original in store.records:
            row = dict(original)
            machine = row["machine_type"]
            row["relative_path"] = f"{machine}/{original['relative_path']}"
            row["source_recording_key"] = f"{machine}/{original['source_recording_key']}"
            row["feature_path"] = os.path.relpath(store.root / original["feature_path"], output)
            row["machine_id_class"] = row["id_class"]
            rows.append(row)
    classes = {key: index for index, key in enumerate(sorted(
        {f"{r['machine_type']}/{r['machine_id']}" for r in rows}))}
    for row in rows:
        row["id_class"] = classes[f"{row['machine_type']}/{row['machine_id']}"]
    validate_splits(rows)
    return rows, classes


def write_store(output, rows, config, classes, sources, scope):
    """Write a FeatureStore-compatible view and fit only its training normals."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    validate_splits(rows)
    write_json(output / "status.json", {"state": "running"})
    manifest = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    digest = hashlib.sha256(manifest.encode()).hexdigest()
    (output / "manifest.jsonl").write_text(manifest)
    for split in ("train", "validation", "test"):
        (output / f"{split}.jsonl").write_text("".join(
            json.dumps(row, sort_keys=True) + "\n" for row in rows if row["split"] == split))
    write_json(output / "config.json", config)
    stats = fit_normalizer(rows, output)
    if not np.isfinite(stats["mean"]).all() or not np.isfinite(stats["std"]).all():
        raise ValueError("Nonfinite normalizer")
    np.savez(output / "normalizer.npz", **stats)
    write_json(output / "normalizer_provenance.json", {
        "fit_split": "train", "fit_label": 0, "scope": "pooled_training_per_mel_band",
        "n_clips": int(stats["n_clips"]), "n_frames": int(stats["n_frames"]),
        "manifest_sha256": digest, "std_floor": 1e-6,
    })
    counts = Counter((r["machine_type"], r["snr_db"], r["machine_id"], r["condition"], r["split"]) for r in rows)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "scope": scope,
        "dataset_protocol": "custom_mimii_recording_split",
        "recordings": len(rows), "unique_source_recordings": len({r["source_recording_key"] for r in rows}),
        "audio_hours_including_snr_versions": sum(r["duration_seconds"] for r in rows) / 3600,
        "machine_ids": classes, "feature_shape": rows[0]["feature_shape"],
        "split_counts": dict(Counter(r["split"] for r in rows)),
        "label_counts": dict(Counter(r["condition"] for r in rows)),
        "counts": [{"machine_type": m, "snr_db": snr, "machine_id": mid,
                    "condition": label, "split": split, "clips": n}
                   for (m, snr, mid, label, split), n in sorted(counts.items())],
        "qa": {
            "sample_rates": sorted({r["sample_rate"] for r in rows}),
            "channel_counts": sorted({r["channels"] for r in rows}),
            "subtypes": sorted({r["subtype"] for r in rows}),
            "silent_recordings": sum(r["silent"] for r in rows),
            "recordings_with_clipped_samples": sum(r["clipped_fraction"] > 0 for r in rows),
            "duplicate_selected_channel_recordings": len(rows) - len({r["waveform_sha256"] for r in rows}),
        },
        "manifest_sha256": digest, "source_group_split_isolation": True,
        "source_identity_assumption": "Machine type/ID/condition/filename identifies corresponding recordings across SNR releases",
        "sources": [{"path": str(s.root), "manifest_sha256": s.manifest_sha256} for s in sources],
        "storage": "Manifest references arrays; no feature copies",
    }
    write_json(output / "summary.json", summary)
    write_json(output / "status.json", {"state": "complete", "manifest_sha256": digest})
    return summary


def run(config):
    start = time.monotonic()
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    saved_config = output / "pipeline_config.json"
    if saved_config.exists():
        if json.loads(saved_config.read_text()) != config:
            raise ValueError("Pipeline configuration changed; use a new output directory")
    elif any(output.iterdir()):
        raise FileExistsError(f"Unrecognized nonempty output: {output}")
    write_json(saved_config, config)
    write_json(output / "pipeline_status.json", {"state": "running"})
    try:
        machines, snrs = config["machine_types"], config["snr_levels_db"]
        if len(set(machines)) != len(machines) or set(snrs) != {-6, 0, 6} or len(snrs) != 3:
            raise ValueError("Expected distinct machines and exactly three SNRs: -6, 0, 6")
        expected_paths, planned = set(), []
        for machine in machines:
            for snr in [0] + sorted(s for s in snrs if s != 0):
                name = f"{machine}_{SNR_NAMES[snr]}"
                data_root = Path(config["data_dir"]) / name / machine
                cache = Path(config.get("reuse_caches", {}).get(name, output / "subsets" / name))
                frontend = PreprocessConfig(**config["frontend"], data_root=str(data_root),
                                            output_dir=str(cache), snr_db=snr)
                inventory = discover_recordings(data_root)
                expected_paths.update((data_root / r["relative_path"]).resolve() for r in inventory)
                planned.append((machine, snr, name, cache, frontend, len(inventory)))
        actual_paths = {p.resolve() for p in Path(config["data_dir"]).rglob("*")
                        if p.is_file() and p.suffix.lower() == ".wav"}
        if actual_paths != expected_paths:
            raise ValueError("Configured subsets do not cover exactly all WAV files in data_dir")
        new_bytes = sum(n * (f.n_mels * f.expected_frames * 4 + 4096)
                        for _, _, _, cache, f, n in planned if not (cache / "status.json").exists())
        if shutil.disk_usage(output).free < new_bytes + 1024 ** 3:
            raise OSError("Insufficient disk space for planned features plus 1 GiB reserve")
        print(f"Full inventory: {len(expected_paths)} WAVs, estimated new features {new_bytes / 1024**3:.2f} GiB", flush=True)
        stores_by_machine, audits = {}, {}
        for machine, snr, name, cache, frontend, _ in planned:
            if snr != 0:
                frontend = PreprocessConfig(**{**frontend.to_dict(), "split_reference_manifest":
                    str(stores_by_machine[machine][0].root / "manifest.jsonl")})
            if not (cache / "status.json").exists():
                print(f"Extracting {name}", flush=True)
                extract_subset(frontend)
            print(f"Auditing {name}: all arrays, source inventory, splits, normalizer", flush=True)
            store, audit = audit_cache(cache, frontend)
            stores_by_machine.setdefault(machine, []).append(store)
            audits[name] = {"path": str(cache), **audit}
            write_json(output / "audit.json", audits)
            write_json(output / "pipeline_status.json", {"state": "running", "verified_subsets": len(audits),
                "verified_recordings": sum(a["recordings_verified"] for a in audits.values())})

        machine_stores = []
        for machine, stores in stores_by_machine.items():
            pooled = output / "by_machine" / machine
            rows = combine_records(stores, pooled)
            for row in rows:
                row["source_path"] = str(Path(next(s.config["data_root"] for s in stores
                    if s.config["snr_db"] == row["snr_db"])) / row["source_recording_key"])
            cfg = {**stores[0].config, "data_root": None, "output_dir": str(pooled),
                   "snr_db": None, "snr_levels_db": sorted(snrs), "split_reference_manifest": None}
            write_store(pooled, rows, cfg, stores[0].summary["machine_ids"], stores,
                        {"machine_type": machine, "snr_levels_db": sorted(snrs)})
            machine_stores.append(FeatureStore(pooled))
        rows, classes = merge_machine_stores(machine_stores, output)
        cfg = {**machine_stores[0].config, "output_dir": str(output), "machine_types": machines,
               "id_class_semantics": "machine_type/machine_id"}
        summary = write_store(output, rows, cfg, classes, machine_stores,
                              {"machine_types": machines, "snr_levels_db": sorted(snrs)})
        summary["feature_bytes"] = sum(a["feature_bytes"] for a in audits.values())
        summary["elapsed_seconds"] = round(time.monotonic() - start, 2)
        write_json(output / "summary.json", summary)
        write_json(output / "pipeline_status.json", {"state": "complete", "verified_subsets": len(audits),
                   "verified_recordings": len(rows), "manifest_sha256": summary["manifest_sha256"]})
        print(json.dumps({k: summary[k] for k in ("recordings", "unique_source_recordings", "split_counts",
            "label_counts", "feature_shape", "feature_bytes", "elapsed_seconds")}, indent=2), flush=True)
        return summary
    except BaseException as error:
        write_json(output / "pipeline_status.json", {"state": "failed", "error": repr(error)})
        write_json(output / "status.json", {"state": "failed", "error": repr(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/full_dataset_logmel.json")
    args = parser.parse_args()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(key, "1")
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/mimii-numba")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mimii-matplotlib")
    run(json.loads(Path(args.config).read_text()))


if __name__ == "__main__":
    main()
