"""Reference existing SNR caches and fit pooled training-normal statistics."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .dataset import validate_splits
from .features import fit_normalizer
from .preprocess import write_json
from .training_data import FeatureStore


FRONTEND_KEYS = ("sample_rate", "duration_seconds", "channel", "n_fft", "hop_length",
                 "n_mels", "fmin", "fmax", "center", "pad_mode", "context_frames", "seed")
DEFAULT_INPUTS = ["features/fan_minus6db_128mel_seed42", "features/fan_0db_128mel_seed42",
                  "features/fan_plus6db_128mel_seed42"]


def combine_records(stores, output):
    reference = stores[0]
    inventory = {r["relative_path"]: (r["machine_id"], r["label"], r["split"], r["id_class"])
                 for r in reference.records}
    rows, seen_snr = [], set()
    for store in stores:
        snr = store.config["snr_db"]
        if snr in seen_snr:
            raise ValueError("Duplicate SNR cache")
        seen_snr.add(snr)
        if any(store.config[k] != reference.config[k] for k in FRONTEND_KEYS):
            raise ValueError("Frontend settings differ across SNR caches")
        current = {r["relative_path"]: (r["machine_id"], r["label"], r["split"], r["id_class"])
                   for r in store.records}
        if current != inventory:
            raise ValueError("Source recordings, labels, or splits differ across SNRs")
        for original in store.records:
            if original["snr_db"] != snr:
                raise ValueError("SNR metadata mismatch")
            row = dict(original)
            row["source_recording_key"] = original["relative_path"]
            row["relative_path"] = f"snr_{snr:+d}db/{original['relative_path']}"
            row["feature_path"] = os.path.relpath(store.root / original["feature_path"], output)
            rows.append(row)
    validate_splits(rows)
    return rows


def prepare(inputs, output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output exists: {output}")
    stores = [FeatureStore(path, "none") for path in inputs]
    rows = combine_records(stores, output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "status.json", {"state": "running"})
    try:
        manifest = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
        digest = hashlib.sha256(manifest.encode()).hexdigest()
        (output / "manifest.jsonl").write_text(manifest)
        for split in ("train", "validation", "test"):
            (output / f"{split}.jsonl").write_text("".join(
                json.dumps(row, sort_keys=True) + "\n" for row in rows if row["split"] == split))
        config = {**stores[0].config, "data_root": None, "output_dir": str(output),
                  "snr_db": None, "snr_levels_db": sorted(s.config["snr_db"] for s in stores),
                  "source_feature_dirs": list(map(str, inputs))}
        config.pop("split_reference_manifest", None)
        write_json(output / "config.json", config)
        stats = fit_normalizer(rows, output)
        if not np.isfinite(stats["mean"]).all() or not np.isfinite(stats["std"]).all():
            raise ValueError("Non-finite normalization statistics")
        np.savez(output / "normalizer.npz", **stats)
        write_json(output / "normalizer_provenance.json", {
            "fit_split": "train", "fit_label": 0, "scope": "pooled_snr_per_mel_band",
            "n_clips": int(stats["n_clips"]), "n_frames": int(stats["n_frames"]),
            "manifest_sha256": digest, "std_floor": 1e-6})
        summary = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "scope": {"machine_type": "fan", "snr_levels_db": config["snr_levels_db"]},
            "recordings": len(rows), "unique_source_recordings": len(stores[0].records),
            "machine_ids": stores[0].summary["machine_ids"], "feature_shape": list(stores[0].shape),
            "split_counts": dict(Counter(row["split"] for row in rows)),
            "unique_source_split_counts": stores[0].summary["split_counts"],
            "manifest_sha256": digest,
            "sources": [{"path": str(s.root), "manifest_sha256": s.manifest_sha256} for s in stores],
            "source_group_split_isolation": True,
            "source_identity_assumption": "Original ID/condition/filename identifies corresponding recordings across noise releases",
            "storage": "Manifest references existing arrays; no WAV or feature copies"}
        write_json(output / "summary.json", summary)
        write_json(output / "status.json", {"state": "complete", "manifest_sha256": digest})
        print(json.dumps(summary, indent=2), flush=True)
        return summary
    except BaseException as error:
        write_json(output / "status.json", {"state": "failed", "error": repr(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", default=DEFAULT_INPUTS)
    parser.add_argument("--output", default="features/fan_all_snr_128mel_seed42")
    args = parser.parse_args()
    prepare(args.inputs, args.output)


if __name__ == "__main__":
    main()
