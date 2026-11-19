"""Full-corpus seed-42 score fusion and boosted-tree experiments.

Run from the repository root using python -m mimii.ensemble_xgb.
"""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
import xgboost as xgb

from .evaluation import detection_metrics, evaluate_scores
from .training_data import FeatureStore


FEATURES = Path("features/full_dataset_logmel128_seed42/by_machine")
OUTPUT = Path("runs/ensemble_xgb_seed42")
MACHINES = ("fan", "pump", "slider", "valve")
MEMBERS = {
    "mobilenet": "mobilenet_v3_small_full_dataset/{machine}/seed_42",
    "resnet18": "cnn_optimization_12epochs/resnet18/lr_1/{machine}",
    "efficientnet_b0": "cnn_optimization_12epochs/efficientnet_b0/lr_1/{machine}",
    "custom": "acoustic_residual_stats_cnn/stats/{machine}/seed_42",
    "large_custom": "large_acoustic_residual_stats_cnn/stats/{machine}/seed_42",
}
PARAMS = dict(objective="multi:softprob", num_class=4, tree_method="hist",
              max_depth=4, eta=0.1, nthread=2, seed=42, max_bin=128)
ROUNDS = 100


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def aligned_scores(path, records):
    with path.open() as handle:
        entries = list(csv.DictReader(handle))
    index = {r["relative_path"]: r for r in entries}
    if len(index) != len(entries) or len(entries) != len(records):
        raise ValueError(f"Duplicate or missing entries in {path}")
    values = []
    for r in records:
        entry = index[r["relative_path"]]
        if entry["machine_id"] != r["machine_id"] or ("label" in entry and int(entry["label"]) != r["label"]):
            raise ValueError(f"Label or ID mismatch in {path}")
        values.append(float(entry["score"]))
    values = np.array(values)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError(f"Invalid CNN scores in {path}")
    return np.log(np.maximum(values, np.finfo(float).tiny))


def fuse(machine, train, test, directory):
    fused_train = np.zeros(len(train))
    fused_test = np.zeros(len(test))
    train_ids = np.array([r["machine_id"] for r in train])
    test_ids = np.array([r["machine_id"] for r in test])
    calibration = {}
    for name, template in MEMBERS.items():
        root = Path("runs") / template.format(machine=machine)
        train_path, test_path = root/"train_scores.csv", root/"scores.csv"
        a, b = aligned_scores(train_path, train), aligned_scores(test_path, test)
        calibration[name] = {"source": str(root), "sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (train_path, test_path)}, "by_id": {}}
        for mid in np.unique(train_ids):
            mask, test_mask = train_ids == mid, test_ids == mid
            center, scale = a[mask].mean(), a[mask].std()
            if scale <= 0:
                raise ValueError(f"Constant training score for {name}/{mid}")
            calibration[name]["by_id"][mid] = {"log_mean": float(center), "log_std": float(scale)}
            fused_train[mask] += (a[mask] - center) / scale
            fused_test[test_mask] += (b[test_mask] - center) / scale
    save(directory/"calibration.json", calibration)
    return fused_train / len(MEMBERS), fused_test / len(MEMBERS)


def tree_inputs(store, rows):
    values = np.empty((len(rows), 2 * store.shape[0]), dtype=np.float32)
    for i, row in enumerate(rows):
        array = store.load(row)
        values[i] = np.concatenate([array.mean(axis=1), array.std(axis=1)])
    return xgb.DMatrix(values, nthread=2)


def tree_scores(model, matrix, rows):
    margins = model.predict(matrix, output_margin=True).astype(np.float64)
    labels = np.array([r["id_class"] for r in rows])
    pos = np.arange(len(rows))
    differences = margins - margins[pos, labels, None]
    differences[pos, labels] = -np.inf
    return np.logaddexp(0, logsumexp(differences, axis=1))


def fit_trees(store, train, test, directory):
    training = tree_inputs(store, train)
    training.set_label([r["id_class"] for r in train])
    model = xgb.train(PARAMS, training, num_boost_round=ROUNDS)
    model.save_model(directory/"model.ubj")
    return tree_scores(model, training, train), tree_scores(model, tree_inputs(store, test), test)


def summarize(cells):
    counts = {key: sum(c[key] for c in cells) for key in ("tp", "fn", "fp", "tn")}
    return {**counts, "auc": float(np.mean([c["auc"] for c in cells])),
            "pauc": float(np.mean([c["pauc"] for c in cells])),
            "recall": counts["tp"] / (counts["tp"] + counts["fn"]),
            "fpr": counts["fp"] / (counts["fp"] + counts["tn"])}


def evaluate(train, test, train_scores, test_scores, directory, machine):
    evaluation = evaluate_scores(test, test_scores, train, train_scores, directory)
    ids = np.array([r["machine_id"] for r in test])
    snrs = np.array([r["snr_db"] for r in test])
    labels = np.array([r["label"] for r in test])
    cells = []
    for mid, overall in evaluation["per_id"].items():
        for snr in (-6, 0, 6):
            mask = (ids == mid) & (snrs == snr)
            metrics = detection_metrics(labels[mask], test_scores[mask], overall["threshold"])
            cells.append({"machine": machine, "id": mid, "snr_db": snr, **metrics})
    save(directory/"cells.json", cells)
    return cells


def main():
    OUTPUT.mkdir(exist_ok=True)
    protocol = {"seed": 42, "members": MEMBERS, "fusion": "equal mean of log anomaly scores standardized per member and ID on training normals",
                "tree_features": "128 temporal means and 128 population SDs of train-normalized log-mel bands",
                "tree_params": PARAMS, "boosting_rounds": ROUNDS, "xgboost": xgb.__version__,
                "threshold_quantile": .95, "manifest_sha256": {},
                "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    results = {"cnn_ensemble": [], "xgboost": []}
    for machine in MACHINES:
        store = FeatureStore(FEATURES/machine, "train_per_mel")
        train, test = store.split("train"), store.split("test")
        if any(r["label"] != 0 for r in train):
            raise ValueError("Training requires healthy recordings")
        protocol["manifest_sha256"][machine] = store.manifest_sha256
        for name in results:
            directory = OUTPUT/name/machine
            directory.mkdir(exist_ok=True, parents=True)
            print(f"{name} {machine}", flush=True)
            if name == "cnn_ensemble":
                scores = fuse(machine, train, test, directory)
            else:
                scores = fit_trees(store, train, test, directory)
            cells = evaluate(train, test, *scores, directory, machine)
            results[name].extend(cells)
    summary = {}
    for name, cells in results.items():
        overall = summarize(cells)
        if len(cells) != 48 or overall["tp"]+overall["fn"] != 9900 or overall["fp"]+overall["tn"] != 8814:
            raise ValueError("Incomplete test coverage")
        summary[name] = {"overall": overall, "by_machine": {
            m: summarize([c for c in cells if c["machine"] == m]) for m in MACHINES}}
    save(OUTPUT/"protocol.json", protocol)
    save(OUTPUT/"summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
