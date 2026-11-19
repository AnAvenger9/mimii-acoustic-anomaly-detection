"""Recompute detection metrics from saved scores and verify run provenance."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from mimii.evaluation import detection_metrics
from mimii.training_data import FeatureStore


def read_csv(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def verify(root, allow_partial=False):
    root = Path(root)
    protocol = json.loads((root / "protocol.json").read_text())
    config = protocol["config"]
    completed, checked_test, checked_train = [], 0, 0
    for name, expected in protocol["source_sha256"].items():
        snapshot = root / "source_snapshot" / name
        if hashlib.sha256(snapshot.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Source snapshot differs: {name}")
    for machine in config["machine_types"]:
        store = FeatureStore(Path(config["feature_root"]) / "by_machine" / machine)
        test = {r["relative_path"]: r for r in store.split("test")}
        train = {r["relative_path"]: r for r in store.split("train")}
        for seed in config["seeds"]:
            directory = root / machine / f"seed_{seed}"
            status_path = directory / "status.json"
            status = json.loads(status_path.read_text()) if status_path.exists() else {}
            if status.get("state") != "complete":
                if allow_partial:
                    continue
                raise ValueError(f"Incomplete run: {machine}/{seed}")
            if hashlib.sha256((directory / "model.pt").read_bytes()).hexdigest() != status["checkpoint_sha256"]:
                raise ValueError("Checkpoint checksum differs")
            metrics = json.loads((directory / "metrics.json").read_text())
            if metrics["feature_manifest_sha256"] != store.manifest_sha256:
                raise ValueError("Feature fingerprint differs")
            scores, train_scores = read_csv(directory / "scores.csv"), read_csv(directory / "train_scores.csv")
            for table, expected in ((scores, test), (train_scores, train)):
                if len(table) != len(expected) or {r["relative_path"] for r in table} != set(expected):
                    raise ValueError("Scores do not cover the exact split manifest")
                if not np.isfinite([float(r["score"]) for r in table]).all():
                    raise ValueError("Nonfinite scores")
                for row in table:
                    if row["machine_id"] != expected[row["relative_path"]]["machine_id"]:
                        raise ValueError("Machine ID differs")
            thresholds = {mid: float(np.quantile([float(r["score"]) for r in train_scores
                if r["machine_id"] == mid], config["threshold_quantile"])) for mid in store.summary["machine_ids"]}
            for row in scores:
                if int(row["label"]) != test[row["relative_path"]]["label"]:
                    raise ValueError("Saved test label differs from the manifest")
                if float(row["threshold"]) != thresholds[row["machine_id"]]:
                    raise ValueError("Threshold is not the training-normal quantile")
                if int(row["prediction"]) != int(float(row["score"]) > float(row["threshold"])):
                    raise ValueError("Decision differs from score/threshold")
            recalculated = []
            for cell in metrics["cells"]:
                selected = [r for r in scores if r["machine_id"] == cell["machine_id"]
                            and test[r["relative_path"]]["snr_db"] == cell["snr_db"]]
                actual = detection_metrics([int(r["label"]) for r in selected],
                    [float(r["score"]) for r in selected], thresholds[cell["machine_id"]])
                for key, value in actual.items():
                    if isinstance(value, (float, int)):
                        np.testing.assert_allclose(value, cell[key], rtol=0, atol=1e-12)
                    elif value != cell[key]:
                        raise ValueError(f"Metric convention differs: {key}")
                recalculated.append(actual)
            for metric in ("auc", "pauc"):
                np.testing.assert_allclose(np.mean([r[metric] for r in recalculated]),
                                           metrics[f"macro_{metric}"], rtol=0, atol=1e-12)
            history = [json.loads(line) for line in (directory / "history.jsonl").read_text().splitlines()]
            selected_epoch = min(history, key=lambda r: r["validation_loss"])["epoch"]
            if selected_epoch != metrics["selected_epoch"]:
                raise ValueError("Checkpoint was not selected by minimum healthy-validation loss")
            completed.append({"machine_type": machine, "seed": seed, "macro_auc": metrics["macro_auc"],
                              "macro_pauc": metrics["macro_pauc"]})
            checked_test += len(scores)
            checked_train += len(train_scores)
    expected_runs = len(config["machine_types"]) * len(config["seeds"])
    if len(completed) == expected_runs:
        summary = json.loads((root / "summary.json").read_text())
        for metric in ("macro_auc", "macro_pauc"):
            values = [np.mean([r[metric] for r in completed if r["seed"] == seed]) for seed in config["seeds"]]
            for key, value in (("mean", np.mean(values)), ("std", np.std(values)), ("variance", np.var(values))):
                np.testing.assert_allclose(value, summary[metric][key], rtol=0, atol=1e-12)
    result = {"state": "passed" if len(completed) == expected_runs else "partial",
              "verified_runs": len(completed), "expected_runs": expected_runs,
              "verified_test_scores": checked_test, "verified_training_scores": checked_train,
              "source_snapshots_and_checkpoint_hashes_verified": True,
              "thresholds_recomputed_from_training_normals": True,
              "metrics_recomputed_from_saved_scores": True,
              "checkpoint_selection_verified": True, "runs": completed}
    (root / ("validation_partial.json" if allow_partial else "validation.json")).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="runs/mobilenet_v3_small_full_dataset")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    verify(args.run_dir, args.allow_partial)
