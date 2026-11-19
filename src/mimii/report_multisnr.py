"""Audit saved CNN scores and report per-SNR, per-ID repeated-seed results."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .evaluation import detection_metrics
from .preprocess import write_json
from .training_data import FeatureStore


def statistics(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "std": float(values.std(ddof=0)),
            "variance": float(values.var(ddof=0)), "n": len(values)}


def read_csv(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def report(root):
    root = Path(root)
    summary = json.loads((root / "summary.json").read_text())
    config = json.loads((root / f"seed_{summary['seeds'][0]}" / "config.json").read_text())
    store = FeatureStore(config["feature_dir"], "train_per_mel")
    records = {r["relative_path"]: r for r in store.split("test")}
    training = {r["relative_path"]: r for r in store.split("train")}
    ids = sorted(store.summary["machine_ids"])
    snrs = sorted({r["snr_db"] for r in records.values()})
    seed_results, result_rows, failures, hashes = [], [], [], {}
    for seed in summary["seeds"]:
        directory = root / f"seed_{seed}"
        state = json.loads((directory / "status.json").read_text())
        if state["state"] != "complete":
            raise ValueError(f"Incomplete seed {seed}")
        saved = json.loads((directory / "metrics.json").read_text())
        if "float64 softplus" not in saved.get("evaluation_score_implementation", ""):
            raise ValueError("Re-evaluate saved CNN checkpoints with the stable scorer before reporting")
        scores = read_csv(directory / "scores.csv")
        train_scores = read_csv(directory / "train_scores.csv")
        if len(scores) != len(records) or {r["relative_path"] for r in scores} != set(records):
            raise ValueError("Test scores do not exactly cover the held-out manifest")
        if len(train_scores) != len(training) or {r["relative_path"] for r in train_scores} != set(training):
            raise ValueError("Training scores do not exactly cover the training manifest")
        thresholds = {}
        for mid in ids:
            values = [float(r["score"]) for r in train_scores if training[r["relative_path"]]["machine_id"] == mid]
            thresholds[mid] = float(np.quantile(values, config["threshold_quantile"]))
            if not np.isclose(thresholds[mid], saved["per_id"][mid]["threshold"], rtol=1e-12, atol=0):
                raise ValueError("Saved threshold differs from training-only quantile")
        for row in scores:
            original = records[row["relative_path"]]
            if row["machine_id"] != original["machine_id"] or int(row["label"]) != original["label"]:
                raise ValueError("Score labels disagree with manifest")
            if float(row["threshold"]) != thresholds[row["machine_id"]]:
                raise ValueError("Threshold changed for an individual test observation")
            if int(row["prediction"]) != int(float(row["score"]) > float(row["threshold"])):
                raise ValueError("Saved decision is inconsistent with threshold")
            if int(row["label"]) != int(row["prediction"]):
                failures.append({"seed": seed, "snr_db": original["snr_db"],
                                 "source_recording_key": original["source_recording_key"],
                                 "error_type": "false_negative" if int(row["label"]) else "false_positive",
                                 **row})
        per_snr = {}
        for snr in snrs:
            by_id = {}
            for mid in ids:
                selected = [r for r in scores if r["machine_id"] == mid and records[r["relative_path"]]["snr_db"] == snr]
                m = detection_metrics([int(r["label"]) for r in selected],
                                      [float(r["score"]) for r in selected], thresholds[mid])
                by_id[mid] = m
                result_rows.append({"seed": seed, "snr_db": snr, "machine_id": mid, **m})
            per_snr[str(snr)] = {"per_id": by_id,
                                 "macro_auc": float(np.mean([m["auc"] for m in by_id.values()])),
                                 "macro_pauc": float(np.mean([m["pauc"] for m in by_id.values()]))}
        history = [json.loads(line) for line in (directory / "history.jsonl").read_text().splitlines()]
        chosen = next(r for r in history if r["epoch"] == saved["selected_epoch"])
        item = {"seed": seed, "selected_epoch": saved["selected_epoch"],
                "selected_validation_id_accuracy": chosen["validation_id_accuracy"],
                "elapsed_seconds": saved["elapsed_seconds"], "per_snr": per_snr,
                "macro_auc": float(np.mean([m["macro_auc"] for m in per_snr.values()])),
                "macro_pauc": float(np.mean([m["macro_pauc"] for m in per_snr.values()]))}
        seed_results.append(item)
        write_json(directory / "metrics_by_snr.json", item)
        for filename in ("metrics.json", "scores.csv", "train_scores.csv", "model.pt"):
            path = directory / filename
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    by_snr = {}
    for snr in snrs:
        key = str(snr)
        by_snr[key] = {metric: statistics([r["per_snr"][key][metric] for r in seed_results])
                       for metric in ("macro_auc", "macro_pauc")}
        by_snr[key]["per_id"] = {
            mid: {metric: statistics([r["per_snr"][key]["per_id"][mid][metric] for r in seed_results])
                  for metric in ("auc", "pauc", "false_positive_rate", "true_positive_rate")} for mid in ids}
    aggregate = {
        "model": summary["model"], "seeds": summary["seeds"], "training_snr_db": snrs,
        "split_seed": summary["split_seed"], "feature_manifest_sha256": store.manifest_sha256,
        "macro_auc": statistics([r["macro_auc"] for r in seed_results]),
        "macro_pauc": statistics([r["macro_pauc"] for r in seed_results]),
        "aggregation": "Compute AUC/pAUC separately in each ID/SNR cell; average four IDs per SNR, then three SNRs per seed; report mean/population SD/variance over initialization seeds",
        "std_convention": "Population, ddof=0; descriptive initialization variability, not a confidence interval",
        "pauc_convention": "sklearn standardized partial AUC with max_fpr=0.1",
        "score_implementation": "float64 softplus(logsumexp(competing_logits - known_logit)); mathematically -log p(known ID)",
        "threshold_rule": "Per-ID 95th percentile of all-SNR training-normal scores; unchanged across test SNRs",
        "test_counts_by_snr": {
            str(snr): {"total": sum(r["snr_db"] == snr for r in records.values()),
                       "normal": sum(r["snr_db"] == snr and r["label"] == 0 for r in records.values()),
                       "abnormal": sum(r["snr_db"] == snr and r["label"] == 1 for r in records.values())}
            for snr in snrs},
        "per_snr": by_snr, "runs": seed_results, "source_sha256": hashes,
        "interpretation": "All three SNRs were represented in training. SNR versions are correlated; source recordings, not versions, define independent split groups.",
    }
    write_json(root / "summary_by_snr.json", aggregate)
    grouped_failures = defaultdict(set)
    failure_counts = Counter()
    for row in failures:
        grouped_failures[(row["source_recording_key"], row["error_type"])].add((row["seed"], row["snr_db"]))
        failure_counts[(row["seed"], row["snr_db"], row["machine_id"], row["error_type"])] += 1
    persistent = [{"source_recording_key": key, "error_type": kind, "failed_conditions": len(conditions)}
                  for (key, kind), conditions in sorted(grouped_failures.items())
                  if len(conditions) == len(snrs) * len(summary["seeds"])]
    write_json(root / "failure_patterns.json", {
        "definition": "Persistent = same source key misclassified at every SNR and initialization seed",
        "conditions_per_source": len(snrs) * len(summary["seeds"]),
        "persistent_source_counts": dict(Counter(r["error_type"] for r in persistent)),
        "persistent_sources": persistent,
        "counts": [{"seed": seed, "snr_db": snr, "machine_id": mid, "error_type": kind, "clips": count}
                   for (seed, snr, mid, kind), count in sorted(failure_counts.items())]})
    for filename, rows in (("results_by_seed_snr_id.csv", result_rows), ("failure_cases.csv", failures)):
        with (root / filename).open("w", newline="") as handle:
            fields = (list(rows[0]) if rows else
                      ["seed", "snr_db", "source_recording_key", "error_type", "relative_path",
                       "machine_id", "label", "score", "threshold", "prediction"])
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    create_figures(root, aggregate, ids, snrs)
    create_report(root, aggregate)
    print(json.dumps({"output": str(root), "macro_auc": aggregate["macro_auc"],
                      "macro_pauc": aggregate["macro_pauc"], "snr_results": by_snr}, indent=2))
    return aggregate


def create_figures(root, result, ids, snrs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#B74F35", "#187F8C", "#173340"]
    x = np.arange(len(ids))
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.1), layout="constrained")
    for ax, metric, label in zip(axes, ("auc", "pauc"), ("AUC (%)", "Standardized pAUC (%)")):
        for i, (snr, color) in enumerate(zip(snrs, colors)):
            cells = [result["per_snr"][str(snr)]["per_id"][mid][metric] for mid in ids]
            ax.bar(x + (i-1)*.25, [100*c["mean"] for c in cells], .24,
                   yerr=[100*c["std"] for c in cells], capsize=2, label=f"{snr:+d} dB", color=color)
        ax.axhline(50, linestyle="--", color="#888888", linewidth=.8)
        ax.set(xticks=x, xticklabels=[mid.replace("id_", "ID ") for mid in ids], ylim=(0, 105), ylabel=label)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, ncol=3, fontsize=10)
    fig.savefig(root / "per_id_snr_results.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
    for run in result["runs"]:
        history = [json.loads(line) for line in (root / f"seed_{run['seed']}" / "history.jsonl").read_text().splitlines()]
        epochs = [r["epoch"] for r in history]
        axes[0].plot(epochs, [r["validation_loss"] for r in history], label=f"Seed {run['seed']}")
        axes[1].plot(epochs, [100*r["validation_id_accuracy"] for r in history], label=f"Seed {run['seed']}")
    axes[0].set(xlabel="Epoch", ylabel="Normal validation cross-entropy", yscale="log")
    axes[1].set(xlabel="Epoch", ylabel="Normal validation ID accuracy (%)")
    axes[0].legend()
    fig.savefig(root / "all_seed_learning_curves.png", dpi=180)
    plt.close(fig)


def create_report(root, result):
    def fmt(cell):
        return f"{100*cell['mean']:.2f} ± {100*cell['std']:.2f}"

    lines = ["# Fan CNN baseline: pooled −6 / 0 / +6 dB", "",
             "Completed experiment; three independent initializations on one fixed source-recording split.", "",
             "| Test SNR | Macro AUC (%) | Macro pAUC (%) |",
             "| --- | ---: | ---: |"]
    for snr, row in result["per_snr"].items():
        lines.append(f"| {int(snr):+d} dB | {fmt(row['macro_auc'])} | {fmt(row['macro_pauc'])} |")
    lines.extend([f"| Mean across SNRs | {fmt(result['macro_auc'])} | {fmt(result['macro_pauc'])} |", "",
                  "Values are mean ± population SD over seeds 42/43/44, in percentage points. Full variances are in `summary_by_snr.json`. pAUC is standardized over FPR ≤ 0.1. The all-SNR value averages ID/SNR-cell metrics; it differs from pooling every score before computing AUC.", "",
                  "## Protocol", "",
                  "- 22,652-parameter CNN: average pooling → three Conv/BN/ReLU/MaxPool blocks (8/16/32 channels) → four frequency regions with temporal mean/std → Dense 64 → Dropout 0.2 → four ID logits.",
                  "- Input: 10 s, microphone 1, 16 kHz; 1,024-point Hann STFT, hop 512, 128 Slaney mel bands, absolute log power; 128 × 313 per clip.",
                  "- Training/validation/test: 8,565 / 1,218 / 6,867 noisy versions, corresponding to 2,855 / 406 / 2,289 distinct source groups. All anomalous clips stay in test.",
                  "- One per-band mean/std is fitted to pooled training normals. All three SNRs are represented during training; normalization is unchanged at evaluation.",
                  "- Cross-entropy; Adam lr 0.001, epsilon 1e-7; batch 32; 30 epochs per seed. Lowest normal-validation cross-entropy selects each checkpoint.",
                  "- Anomaly score: negative log probability of the known machine ID. Each ID threshold is the 95th percentile of pooled training-normal scores, fixed across SNRs.",
                  "- Numerical evaluation: float64 `softplus(logsumexp(other_logits - known_logit))` preserves small positive scores that float32 log-softmax rounded to zero. Frozen checkpoints were re-evaluated; original evaluations and the original pooled summary are archived. Epoch training-log AUC lines refer to that superseded first evaluation. No architecture, checkpoint, mathematical score or threshold rule was selected using this correction.",
                  "- GPU: NVIDIA L4-3Q. PyTorch requests deterministic algorithms in warning mode; CUDA adaptive pooling warns that its backward pass is nondeterministic. Seeding does not guarantee bitwise repeatability.", "",
                  "## Selected checkpoints", "", "| Seed | Epoch | Normal validation ID accuracy (%) | Training + evaluation time (s) |",
                  "| --- | ---: | ---: | ---: |"])
    for run in result["runs"]:
        lines.append(f"| {run['seed']} | {run['selected_epoch']} | {100*run['selected_validation_id_accuracy']:.2f} | {run['elapsed_seconds']:.1f} |")
    lines.extend(["", "## Per-ID results at 0 dB", "",
                  "| Machine ID | AUC (%) | pAUC (%) | Fault recall at fixed threshold (%) | False-positive rate (%) |",
                  "| --- | ---: | ---: | ---: | ---: |"])
    for mid, row in result["per_snr"]["0"]["per_id"].items():
        lines.append(f"| {mid} | {fmt(row['auc'])} | {fmt(row['pauc'])} | {fmt(row['true_positive_rate'])} | {fmt(row['false_positive_rate'])} |")
    case = result["runs"][0]["per_snr"]["0"]["per_id"]["id_00"]
    lines.extend(["", f"Presentation example (seed {result['runs'][0]['seed']}, ID 00, 0 dB): TN={case['tn']}, FP={case['fp']}, FN={case['fn']}, TP={case['tp']}. This seed was chosen in advance as an illustrative example, not selected for its detection score."])
    lines.extend(["", "## Interpretation and limits", "",
                  "This establishes a repeatable CNN baseline and a first comparison across known noise levels. It does not establish generalization to an unseen SNR, a new machine type, or a held-out fault category. Related SNR versions are not independent test observations. Three seeds quantify initialization variability only; the data partition is fixed.", "",
                  "Healthy-ID classification and anomaly detection measure different things. Inspect per-ID AUC and false negatives before relying on a macro average. `failure_cases.csv` preserves every false positive/negative with original recording identity, SNR, seed, score and threshold. Matrices for each seed are in `seed_*/confusion_matrices.png`; per-SNR counts are in `seed_*/metrics_by_snr.json`.", "",
                  "`failure_patterns.json` identifies source recordings misclassified at every one of the nine seed/SNR combinations. These are priorities for listening and spectral inspection; repeated misses alone do not identify a physical fault mechanism.", "",
                  "Earlier pilot test results were already inspected, so this remains exploratory. The architecture, epochs, seeds and threshold rule were fixed before this run's results; test fault labels did not train or select its checkpoints. A future final comparison should reserve a fresh validation protocol or final untouched test set.", "",
                  "## Next experiments", "",
                  "1. Train DCASE-style reconstruction autoencoders on the same source split and SNR conditions; report AUC/pAUC and training-normal thresholds with the same seeds. Keep absolute log-mel inputs for the documented AE baseline and explicitly report normalization differences. Compare per-ID and pooled training deliberately.",
                  "2. Sweep AE latent dimensions (for example 4/8/16/32) and inspect normal/fault reconstruction distributions to test the excess-capacity hypothesis.",
                  "3. Compare log-mel/MFCC/raw waveform, FFT/mel settings, and mean-only versus mean/std pooling while holding the remaining factors fixed.",
                  "4. Compare pooled-SNR training with matched single-SNR and leave-one-SNR-out training; extend to a contrasting transient machine type and investigate persistent false negatives.",
                  "5. Keep published DCASE benchmark scores separate: this uses original MIMII releases and a custom split, so it is not a numerical reproduction of DCASE 2020.", "",
                  "## Stored artifacts", "",
                  "`protocol.json` records the decisions; `source_snapshot/` freezes code/configuration; `feature_normalizer.npz` preserves pooled normalization; `training.log` records all epochs. Each `seed_*/` contains `model.pt`, training histories, provenance, train/test scores and metrics. `results_by_seed_snr_id.csv` is the tidy numeric table; `summary_by_snr.json` contains means, SDs, variances, per-seed results and artifact hashes.", ""])
    (root / "RESULTS.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="runs/fan_all_snr_temporal_stats_cnn128_gpu")
    args = parser.parse_args()
    report(args.run_dir)


if __name__ == "__main__":
    main()
