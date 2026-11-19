"""Validate and publish the bounded CNN comparison without overstating repeats."""

import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from .evaluation import detection_metrics


def audit(root, study, results, stores):
    protocol = json.loads((root / "protocol.json").read_text())
    for name, digest in protocol["identity"]["source_sha256"].items():
        if hashlib.sha256((root / "source_snapshot" / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Source snapshot hash mismatch")
    selection = json.loads((root / "selection.json").read_text())
    screening = json.loads((root / "screening.json").read_text())
    from .cnn_optimization import choose_trials
    if choose_trials(screening, study["architectures"], study["machine_types"], len(study["backbone_learning_rates"])) != selection:
        raise ValueError("Learning-rate choice differs from healthy-validation rule")
    for row in screening:
        directory = root / row["architecture"] / f"lr_{row['trial']}" / row["machine_type"]
        history = json.loads((directory / "history.json").read_text())[:study["screening_epochs"]]
        np.testing.assert_allclose(min(r["validation_loss"] for r in history), row["validation_loss"], rtol=0, atol=1e-12)
    expected = {(a, m) for a in study["architectures"] for m in study["machine_types"]}
    if len(results) != len(expected) or {(r["architecture"], r["machine_type"]) for r in results} != expected:
        raise ValueError("Incomplete model/machine result coverage")
    total_train, total_test = 0, 0
    for result in results:
        architecture, machine = result["architecture"], result["machine_type"]
        index = selection[architecture]["selected_trial"]
        directory = root / architecture / f"lr_{index}" / machine
        if result["trial"] != index or hashlib.sha256((directory / "model.pt").read_bytes()).hexdigest() != result["checkpoint_sha256"]:
            raise ValueError("Selected/evaluated model differs")
        store = stores[machine]
        test = {r["relative_path"]: r for r in store.split("test")}
        train = {r["relative_path"]: r for r in store.split("train")}
        with (directory / "scores.csv").open() as handle:
            scores = list(csv.DictReader(handle))
        with (directory / "train_scores.csv").open() as handle:
            train_scores = list(csv.DictReader(handle))
        for rows, records in ((scores, test), (train_scores, train)):
            if len(rows) != len(records) or {r["relative_path"] for r in rows} != set(records):
                raise ValueError("Saved scores do not cover exact split")
            if not np.isfinite([float(r["score"]) for r in rows]).all():
                raise ValueError("Nonfinite score")
            if any(r["machine_id"] != records[r["relative_path"]]["machine_id"] for r in rows):
                raise ValueError("Machine ID mismatch")
        thresholds = {mid: np.quantile([float(r["score"]) for r in train_scores if r["machine_id"] == mid], study["threshold_quantile"])
                      for mid in store.summary["machine_ids"]}
        for row in scores:
            if int(row["label"]) != test[row["relative_path"]]["label"] or float(row["threshold"]) != thresholds[row["machine_id"]]:
                raise ValueError("Label or training-only threshold mismatch")
            if int(row["prediction"]) != int(float(row["score"]) > float(row["threshold"])):
                raise ValueError("Thresholded prediction mismatch")
        expected_cells = {(r["machine_id"], r["snr_db"]) for r in test.values()}
        if len(result["cells"]) != len(expected_cells) or {(r["machine_id"], r["snr_db"]) for r in result["cells"]} != expected_cells:
            raise ValueError("Missing ID/SNR cell")
        for cell in result["cells"]:
            rows = [r for r in scores if r["machine_id"] == cell["machine_id"] and test[r["relative_path"]]["snr_db"] == cell["snr_db"]]
            actual = detection_metrics([int(r["label"]) for r in rows], [float(r["score"]) for r in rows], thresholds[cell["machine_id"]])
            for key, value in actual.items():
                if isinstance(value, (float, int)):
                    np.testing.assert_allclose(value, cell[key], rtol=0, atol=1e-12)
                elif cell[key] != value:
                    raise ValueError("Metric convention mismatch")
        for metric in ("auc", "pauc"):
            np.testing.assert_allclose(np.mean([r[metric] for r in result["cells"]]), result[f"macro_{metric}"], rtol=0, atol=1e-12)
        history = json.loads((directory / "history.json").read_text())
        if min(history, key=lambda r: r["validation_loss"])["epoch"] != result["selected_epoch"]:
            raise ValueError("Epoch selection mismatch")
        total_train += len(train_scores)
        total_test += len(scores)
    validation = {"state": "passed", "checked_models": len(results), "checked_train_scores": total_train,
                  "checked_test_scores": total_test, "healthy_validation_selection_checked": True,
                  "source_snapshots_and_model_hashes_checked": True, "thresholds_and_metrics_recomputed": True}
    (root / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    return validation


def report(root, study, results, stores, publish_dir=None):
    audit(root, study, results, stores)
    summary = {"seed": study["seed"], "n_seeds": 1, "standard_deviation": None,
               "aggregation": "equal mean of machine-type/ID/SNR cells", "models": {}}
    rows = []
    for architecture in study["architectures"]:
        selected = [r for r in results if r["architecture"] == architecture]
        summary["models"][architecture] = {"macro_auc": float(np.mean([r["macro_auc"] for r in selected])),
            "macro_pauc": float(np.mean([r["macro_pauc"] for r in selected])), "per_machine": {r["machine_type"]: r for r in selected}}
        rows.extend({"architecture": architecture, **c} for r in selected for c in r["cells"])
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (root / "results_by_machine_snr_id.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    mobile = json.loads(Path("docs/results/mobilenet_v3_small/summary.json").read_text())
    lines = ["# CNN architecture and learning-rate comparison", "", "New models use one initialization seed (42); no seed-variability estimate is available.",
             "Learning-rate selection and checkpoint selection use healthy-validation cross-entropy only.", "",
             "| Model | Machine | AUC (%) | pAUC (%) |", "|---|---|---:|---:|"]
    for architecture, metrics in summary["models"].items():
        for machine, result in metrics["per_machine"].items():
            lines.append(f"| {architecture} | {machine} | {result['macro_auc'] * 100:.2f} | {result['macro_pauc'] * 100:.2f} |")
        lines.append(f"| {architecture} | Overall | {metrics['macro_auc'] * 100:.2f} | {metrics['macro_pauc'] * 100:.2f} |")
    lines.extend(["", f"Existing MobileNetV3-Small: AUC {mobile['macro_auc']['mean'] * 100:.2f} ± {mobile['macro_auc']['std'] * 100:.2f}%; "
                  f"pAUC {mobile['macro_pauc']['mean'] * 100:.2f} ± {mobile['macro_pauc']['std'] * 100:.2f}% (three seeds, up to 20 epochs).",
                  "", "The CNNs use different epoch limits and initialization counts. Comparisons are descriptive; differences do not isolate architecture or establish statistical superiority.",
                  "The normal-ID validation objective is a proxy and does not guarantee better anomaly detection.", "",
                  "Published DCASE 2020 AE results belong to a different dataset partition/protocol and must be presented separately as literature results, not as a matched experimental baseline."])
    (root / "RESULTS.md").write_text("\n".join(lines) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    x = np.arange(len(study["machine_types"]))
    for ax, metric in zip(axes, ("auc", "pauc")):
        for offset, architecture in ((-.25, study["architectures"][0]), (0, study["architectures"][1])):
            ax.bar(x + offset, [summary["models"][architecture]["per_machine"][m][f"macro_{metric}"] * 100 for m in study["machine_types"]], .25, label=f"{architecture} (seed 42)")
        vals = [mobile["per_machine"][m][f"macro_{metric}"] for m in study["machine_types"]]
        ax.bar(x + .25, [v["mean"] * 100 for v in vals], .25, yerr=[v["std"] * 100 for v in vals], capsize=3, label="MobileNetV3 (3 seeds)")
        ax.set(xticks=x, xticklabels=study["machine_types"], ylim=(0, 100), ylabel=f"{metric.upper()} (%)")
    axes[0].legend(fontsize=7)
    for ext in ("png", "pdf"):
        fig.savefig(root / f"cnn_comparison.{ext}", dpi=200)
    plt.close(fig)
    destination = Path(publish_dir) if publish_dir is not None else Path("docs/results/cnn_optimization_2h")
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("summary.json", "RESULTS.md", "protocol.json", "selection.json", "screening.json", "final_plan.json",
                 "validation.json", "results_by_machine_snr_id.csv", "cnn_comparison.png", "cnn_comparison.pdf"):
        shutil.copy2(root / name, destination / name)
    (destination / "artifact_hashes.json").write_text(json.dumps({p.name: hashlib.sha256(p.read_bytes()).hexdigest()
         for p in destination.iterdir() if p.name != "artifact_hashes.json"}, indent=2) + "\n")
