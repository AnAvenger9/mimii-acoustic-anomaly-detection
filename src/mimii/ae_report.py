"""Independent saved-score checks and report artifacts for the AE study."""

import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score
import torch

from .features import context_vectors
from .models import DenseAutoencoder


def audit_run(output, store, config):
    result = json.loads((output / "result.json").read_text())
    history = json.loads((output / "history.json").read_text())
    if [r["epoch"] for r in history] != list(range(1, config.epochs + 1)):
        raise ValueError("Incomplete epoch history")
    best_epoch = min(history, key=lambda row: row["validation_loss"])["epoch"]
    if result["best_validation_epoch"] != best_epoch:
        raise ValueError("Selected checkpoint did not minimize healthy-validation MSE")
    test = {r["relative_path"]: r for r in store.split("test")}
    train = {r["relative_path"]: r for r in store.split("train")}
    for view in ("final", "best_validation"):
        with (output / view / "scores.csv").open() as handle:
            scores = list(csv.DictReader(handle))
        with (output / view / "train_scores.csv").open() as handle:
            train_scores = list(csv.DictReader(handle))
        for rows, expected in ((scores, test), (train_scores, train)):
            if len(rows) != len(expected) or {r["relative_path"] for r in rows} != set(expected):
                raise ValueError("Score CSV recording coverage differs from frozen split")
            if not np.isfinite([float(r["score"]) for r in rows]).all():
                raise ValueError("Nonfinite saved scores")
            if any(row["machine_id"] != expected[row["relative_path"]]["machine_id"] for row in rows):
                raise ValueError("Score CSV machine ID mismatch")
        thresholds = {mid: np.quantile([float(r["score"]) for r in train_scores if r["machine_id"] == mid], config.threshold_quantile)
                      for mid in {r["machine_id"] for r in train_scores}}
        for row in scores:
            if int(row["label"]) != test[row["relative_path"]]["label"]:
                raise ValueError("Score CSV label mismatch")
            np.testing.assert_allclose(float(row["threshold"]), thresholds[row["machine_id"]], rtol=0, atol=1e-10)
            if int(row["prediction"]) != int(float(row["score"]) > float(row["threshold"])):
                raise ValueError("Saved prediction mismatch")
        expected_cells = {(r["snr_db"], r["machine_id"]) for r in test.values()}
        cells = result[view]["cells"]
        if len(cells) != len(expected_cells) or {(r["snr_db"], r["machine_id"]) for r in cells} != expected_cells:
            raise ValueError("Missing/duplicated ID-SNR evaluation cells")
        for cell in cells:
            selected = [r for r in scores if r["machine_id"] == cell["machine_id"] and test[r["relative_path"]]["snr_db"] == cell["snr_db"]]
            labels, values = np.array([int(r["label"]) for r in selected]), np.array([float(r["score"]) for r in selected])
            for key, max_fpr in (("auc", None), ("pauc", 0.1)):
                np.testing.assert_allclose(cell[key], roc_auc_score(labels, values, max_fpr=max_fpr), rtol=0, atol=1e-12)
            counts = confusion_matrix(labels, values > thresholds[cell["machine_id"]], labels=[0, 1]).ravel()
            np.testing.assert_array_equal([cell[k] for k in ("tn", "fp", "fn", "tp")], counts)
        for metric in ("auc", "pauc"):
            np.testing.assert_allclose(result[view][f"macro_{metric}"], np.mean([c[metric] for c in cells]), rtol=0, atol=1e-12)
        checkpoint = torch.load(output / f"{view}.pt", map_location="cpu", weights_only=False)
        expected_epoch = config.epochs if view == "final" else best_epoch
        if checkpoint["epoch"] != expected_epoch or checkpoint["manifest_sha256"] != store.manifest_sha256:
            raise ValueError("Checkpoint provenance/selection mismatch")
        # Independently reconstruct one normal and one anomalous clip on CPU.
        model = DenseAutoencoder(checkpoint["input_dimensions"], config.latent_dimensions)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        with torch.inference_mode():
            for label in (0, 1):
                row = next(r for r in scores if int(r["label"]) == label)
                x = torch.from_numpy(context_vectors(store.load(test[row["relative_path"]]), store.context_frames))
                score = float((model(x) - x).square().mean())
                np.testing.assert_allclose(score, float(row["score"]), rtol=2e-4, atol=1e-4)
    audit = {"state": "passed", "checks": ["all epochs", "healthy-validation checkpoint selection", "full train/test score coverage",
             "training-only thresholds", "all ID/SNR AUC and standardized pAUC", "confusion counts", "checkpoint provenance",
             "independent CPU rescore of normal/anomalous clips for both checkpoints"],
             "test_clips_per_view": len(test), "train_clips_per_view": len(train),
             "artifact_sha256": {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for pattern in ("*.pt", "history.json", "result.json", "*/*.csv", "*/metrics*.json") for p in sorted(output.glob(pattern))}}
    (output / "validation.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def publish_report(root, summary, config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not summary["complete"]:
        raise ValueError("Cannot publish incomplete experiment as final")
    mobile_path = Path("docs/results/mobilenet_v3_small/summary.json")
    mobile = json.loads(mobile_path.read_text()) if mobile_path.exists() else None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    x = np.arange(len(config.machine_types))
    for ax, metric, title in zip(axes, ("auc", "pauc"), ("ROC-AUC", "Standardized pAUC (FPR ≤ 0.1)")):
        for offset, view, label in ((-0.25, "final", "AE: epoch 100"), (0, "best_validation", "AE: healthy-val best")):
            values = [summary["views"][view]["by_machine"][m][metric] for m in config.machine_types]
            ax.bar(x + offset, [v["mean"] * 100 for v in values], .25,
                   yerr=[(v["std"] or 0) * 100 for v in values], label=label, capsize=3)
        if mobile:
            values = [mobile["per_machine"][m][f"macro_{metric}"] for m in config.machine_types]
            ax.bar(x + .25, [v["mean"] * 100 for v in values], .25, yerr=[v["std"] * 100 for v in values], label="MobileNetV3-Small", capsize=3)
        ax.set(xticks=x, xticklabels=config.machine_types, ylabel="%", ylim=(0, 100), title=title)
    axes[0].legend(fontsize=8)
    for extension in ("png", "pdf"):
        fig.savefig(root / f"ae_mobilenet_comparison.{extension}", dpi=200)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    for ax, machine in zip(axes.flat, config.machine_types):
        for seed in config.seeds:
            history = json.loads((root / machine / f"seed_{seed}" / "history.json").read_text())
            line, = ax.plot([r["epoch"] for r in history], [r["validation_loss"] for r in history], label=f"val, seed {seed}")
            ax.plot([r["epoch"] for r in history], [r["train_loss"] for r in history], "--", color=line.get_color(), alpha=.6)
        ax.set(title=machine, xlabel="Epoch", ylabel="Reconstruction MSE")
        ax.legend(fontsize=7)
    fig.savefig(root / "learning_curves.png", dpi=180)
    plt.close(fig)
    if mobile:
        with (root / "RESULTS.md").open("a") as handle:
            handle.write("\nMobileNetV3-Small on the same splits: "
                         f"AUC {mobile['macro_auc']['mean'] * 100:.2f} ± {mobile['macro_auc']['std'] * 100:.2f}%; "
                         f"pAUC {mobile['macro_pauc']['mean'] * 100:.2f} ± {mobile['macro_pauc']['std'] * 100:.2f}%.\n"
                         "\nCompare complete detection systems: the AE uses raw log-mel reconstruction, while MobileNet uses "
                         "ImageNet pretraining, train-normal standardized inputs, known-ID scoring and healthy-validation selection. "
                         "These differences preclude attributing a gap solely to architecture.\n")
    destination = Path("docs/results/dcase2020_ae")
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("summary.json", "RESULTS.md", "protocol.json", "per_id_snr_seed.csv", "ae_mobilenet_comparison.png", "ae_mobilenet_comparison.pdf", "learning_curves.png"):
        shutil.copy2(root / name, destination / name)
    audits = {str(p.parent.relative_to(root)): json.loads(p.read_text()) for p in sorted(root.glob("*/seed_*/validation.json"))}
    (destination / "validation.json").write_text(json.dumps(audits, indent=2) + "\n")
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(destination.iterdir()) if p.name != "artifact_hashes.json"}
    (destination / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
