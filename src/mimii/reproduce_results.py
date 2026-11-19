"""Recalculate the fault-detection table from published per-clip anomaly scores."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def score_fit(bundle, key):
    """Fit alarm thresholds on healthy scores and evaluate one seed."""
    train, test = bundle[key + "_train"], bundle[key + "_test"]
    y = bundle["test_label"]
    if not (np.isfinite(train).all() and np.isfinite(test).all()):
        raise ValueError("Expected finite scores")
    if np.any(bundle["train_label"]) or (len(train), len(test), y.sum()) != (30948, 18714, 9900):
        raise ValueError("Unexpected split or label inventory")
    alarm = np.zeros(len(test), dtype=bool)
    cells = []
    for machine in np.unique(bundle["test_machine"]):
        for mid in np.unique(bundle["test_id"]):
            healthy = (bundle["train_machine"] == machine) & (bundle["train_id"] == mid)
            held_out = (bundle["test_machine"] == machine) & (bundle["test_id"] == mid)
            cutoff = np.quantile(train[healthy], 0.95)
            alarm[held_out] = test[held_out] > cutoff
            for snr in (-6, 0, 6):
                group = held_out & (bundle["test_snr"] == snr)
                cells.append([roc_auc_score(y[group], test[group]),
                              roc_auc_score(y[group], test[group], max_fpr=0.1)])
    if len(cells) != 48:
        raise ValueError("Expected 48 machine, ID and noise groups")
    return [*np.mean(cells, axis=0), alarm[y == 1].mean(), alarm[y == 0].mean()]


def results_table(root=Path("results")):
    """Return mean percentages for each model and preserve seed-level variation."""
    by_model = defaultdict(list)
    with np.load(root / "clip_scores.npz", allow_pickle=False) as archive:
        bundle = {name: archive[name] for name in archive.files if not name.endswith("_path")}
        for fit in json.loads((root / "fits.json").read_text()):
            by_model[fit["model"]].append(score_fit(bundle, fit["key"]))
    return {model: {"seeds": len(rows), "mean": 100 * np.mean(rows, axis=0),
                    "sd": 100 * np.std(rows, axis=0)} for model, rows in by_model.items()}


def print_table(table):
    print(f"{'Model':<22} {'Seeds':>5} {'AUC':>8} {'pAUC':>8} {'Recall':>8} {'FPR':>8}")
    for model, stats in table.items():
        print(f"{model:<22} {stats['seeds']:5d}" + "".join(f"{n:9.2f}" for n in stats["mean"]))


if __name__ == "__main__":
    print_table(results_table())
