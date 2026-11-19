"""Independently verify final group coverage and pooled training statistics."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from mimii.training_data import FeatureStore


def check_statistics(store, total, squared_total, frames, clips):
    # Independent direct-moment calculation, rather than the extraction
    # pipeline's streaming batch-variance combination.
    mean = total / frames
    std = np.maximum(np.sqrt(np.maximum(squared_total / frames - mean ** 2, 0)), 1e-6)
    with np.load(store.root / "normalizer.npz", allow_pickle=False) as saved:
        if int(saved["n_frames"]) != frames or int(saved["n_clips"]) != clips:
            raise ValueError("Normalizer counts differ")
        mean_error = float(np.max(np.abs((mean - saved["mean"]) / saved["std"])))
        std_error = float(np.max(np.abs(std / saved["std"] - 1)))
    if mean_error > 1e-5 or std_error > 1e-5:
        raise ValueError(f"Normalizer moments differ: {store.root}")
    return {"training_clips": clips, "training_frames": frames,
            "maximum_absolute_normalized_mean": mean_error,
            "maximum_absolute_normalized_std_error": std_error}


def verify(root, output):
    store = FeatureStore(root)
    groups = defaultdict(list)
    for row in store.records:
        groups[row["source_recording_key"]].append(row)
        source = Path(row["source_path"]).stat()
        if (source.st_size, source.st_mtime_ns) != (row["source_size_bytes"], row["source_mtime_ns"]):
            raise ValueError(f"Source changed: {row['source_path']}")
    for rows in groups.values():
        if len(rows) != 3 or {r["snr_db"] for r in rows} != {-6, 0, 6}:
            raise ValueError("Expected exactly three SNR versions per source")
        if len({(r["split"], r["label"], r["id_class"]) for r in rows}) != 1:
            raise ValueError("Source versions have inconsistent assignments")
    total = np.zeros(store.shape[0], dtype=np.float64)
    squared_total = total.copy()
    frames, clips, checks = 0, 0, {}
    for machine in store.config["machine_types"]:
        subset = FeatureStore(store.root / "by_machine" / machine)
        machine_total, machine_squared = np.zeros_like(total), np.zeros_like(total)
        machine_frames = 0
        train = subset.split("train")
        for row in train:
            if row["label"] != 0:
                raise ValueError("Fault in training normalization")
            x = subset.load(row).astype(np.float64)
            machine_total += x.sum(axis=1)
            machine_squared += (x * x).sum(axis=1)
            machine_frames += x.shape[1]
        checks[machine] = check_statistics(subset, machine_total, machine_squared, machine_frames, len(train))
        total += machine_total
        squared_total += machine_squared
        frames += machine_frames
        clips += len(train)
    checks["all_machines"] = check_statistics(store, total, squared_total, frames, clips)
    result = {"state": "passed", "recordings": len(store.records), "source_groups": len(groups),
              "all_groups_have_three_aligned_snr_versions": True,
              "all_source_sizes_and_mtimes_unchanged": True,
              "manifest_sha256": store.manifest_sha256, "normalization_checks": checks}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default="features/full_dataset_logmel128_seed42")
    parser.add_argument("--output", default="runs/full_dataset_preprocessing/final_validation.json")
    args = parser.parse_args()
    verify(args.features, args.output)
