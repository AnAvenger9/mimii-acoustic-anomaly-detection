"""Clip-level detection metrics, training-normal thresholds, and audit outputs."""

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve


def detection_metrics(labels, scores, threshold, max_fpr=0.1):
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    if labels.ndim != 1 or scores.shape != labels.shape or not len(labels):
        raise ValueError("Expected equally sized, nonempty clip labels and scores")
    if not np.isfinite(scores).all() or set(np.unique(labels)) != {0, 1}:
        raise ValueError("Metrics require finite scores and both normal/anomaly labels")
    tn, fp, fn, tp = confusion_matrix(labels, scores > threshold, labels=[0, 1]).ravel()
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "pauc": float(roc_auc_score(labels, scores, max_fpr=max_fpr)),
        "max_fpr": max_fpr, "threshold": float(threshold),
        "threshold_source": "training_normal_scores",
        "normal_clips": int(np.sum(labels == 0)), "abnormal_clips": int(np.sum(labels == 1)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "false_positive_rate": float(fp / (fp + tn)),
        "true_positive_rate": float(tp / (tp + fn)),
    }


def evaluate_scores(records, test_scores, train_records, train_scores, output_dir,
                    threshold_quantile=0.95):
    if not 0 < threshold_quantile < 1:
        raise ValueError("Invalid normal-score threshold quantile")
    if len(records) != len(test_scores) or len(train_records) != len(train_scores):
        raise ValueError("Scores do not align with recording manifests")
    if any(r["split"] != "test" for r in records):
        raise ValueError("Evaluation requires the test split")
    if any(r["label"] != 0 or r["split"] != "train" for r in train_records):
        raise ValueError("Threshold fitting requires training normals only")
    test_scores, train_scores = np.asarray(test_scores), np.asarray(train_scores)
    if not np.isfinite(train_scores).all() or not np.isfinite(test_scores).all():
        raise ValueError("Non-finite anomaly scores")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    by_id, thresholds = {}, {}
    for mid in sorted({r["machine_id"] for r in records}):
        mask = np.array([r["machine_id"] == mid for r in records])
        train_mask = np.array([r["machine_id"] == mid for r in train_records])
        if not train_mask.any():
            raise ValueError(f"No training normals for {mid}")
        threshold = float(np.quantile(train_scores[train_mask], threshold_quantile))
        thresholds[mid] = threshold
        labels = [r["label"] for r in records if r["machine_id"] == mid]
        by_id[mid] = detection_metrics(labels, test_scores[mask], threshold)
    metrics = {
        "per_id": by_id,
        "macro_auc": float(np.mean([m["auc"] for m in by_id.values()])),
        "macro_pauc": float(np.mean([m["pauc"] for m in by_id.values()])),
        "pauc_convention": "sklearn standardized partial AUC, max_fpr=0.1",
        "threshold_quantile": threshold_quantile,
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    with (output / "scores.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path", "machine_id", "label", "score", "threshold", "prediction"])
        writer.writeheader()
        for row, score in zip(records, test_scores):
            threshold = thresholds[row["machine_id"]]
            writer.writerow({"relative_path": row["relative_path"], "machine_id": row["machine_id"],
                             "label": row["label"], "score": float(score), "threshold": threshold,
                             "prediction": int(score > threshold)})
    with (output / "train_scores.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["relative_path", "machine_id", "score"])
        writer.writerows((r["relative_path"], r["machine_id"], float(s)) for r, s in zip(train_records, train_scores))
    create_evaluation_figures(records, test_scores, metrics, output)
    return metrics


def create_evaluation_figures(records, scores, metrics, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mids = sorted(metrics["per_id"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for mid in mids:
        mask = np.array([r["machine_id"] == mid for r in records])
        labels = [r["label"] for r in records if r["machine_id"] == mid]
        fpr, tpr, _ = roc_curve(labels, scores[mask])
        axes[0].plot(fpr, tpr, label=f'{mid}: {metrics["per_id"][mid]["auc"]:.3f}')
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC by machine ID")
    axes[0].legend(fontsize=9)
    x = np.arange(len(mids))
    for offset, key, label in ((-0.18, "auc", "AUC"), (0.18, "pauc", "pAUC (FPR ≤ 0.1)")):
        bars = axes[1].bar(x + offset, [metrics["per_id"][mid][key] for mid in mids], 0.36, label=label)
        axes[1].bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
    axes[1].set(xticks=x, xticklabels=mids, ylim=(0, 1.12), title="Clip-level detection")
    axes[1].legend(fontsize=9)
    fig.savefig(output / "detection_results.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(mids), figsize=(3 * len(mids), 3.2), layout="constrained", squeeze=False)
    for ax, mid in zip(axes.ravel(), mids):
        m = metrics["per_id"][mid]
        values = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        ax.imshow(values, cmap="Blues")
        for (i, j), count in np.ndenumerate(values):
            ax.text(j, i, str(count), ha="center", va="center")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Normal", "Anomaly"],
               yticklabels=["Normal", "Anomaly"], xlabel="Predicted", ylabel="Actual", title=mid)
    fig.savefig(output / "confusion_matrices.png", dpi=170)
    plt.close(fig)
