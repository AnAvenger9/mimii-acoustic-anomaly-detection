"""Report MobileNet detection metrics separately by machine, ID, SNR, and seed."""

import csv
import json
from pathlib import Path

import numpy as np

from .preprocess import write_json


def statistics(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "std": float(values.std(ddof=0)),
            "variance": float(values.var(ddof=0)), "n": len(values)}


def report(root, results, config):
    root = Path(root)
    expected = {(machine, seed) for machine in config.machine_types for seed in config.seeds}
    if {(r["machine_type"], r["seed"]) for r in results} != expected or len(results) != len(expected):
        raise ValueError("Results do not cover every declared machine/seed run")
    for result in results:
        state = json.loads((root / result["machine_type"] / f"seed_{result['seed']}" / "status.json").read_text())
        if state["state"] != "complete":
            raise ValueError("Cannot report incomplete runs")
    cells = [cell for result in results for cell in result["cells"]]
    if len(cells) != len(results) * 12:
        raise ValueError("Expected four IDs and three SNRs per run")
    per_seed = [{"seed": seed, **{f"macro_{metric}": float(np.mean(
        [r[metric] for r in cells if r["seed"] == seed])) for metric in ("auc", "pauc")}}
        for seed in config.seeds]
    by_machine, by_snr = {}, {}
    for machine in config.machine_types:
        by_machine[machine] = {f"macro_{metric}": statistics([
            float(np.mean([r[metric] for r in cells if r["machine_type"] == machine and r["seed"] == seed]))
            for seed in config.seeds]) for metric in ("auc", "pauc")}
    for snr in (-6, 0, 6):
        by_snr[str(snr)] = {f"macro_{metric}": statistics([
            float(np.mean([r[metric] for r in cells if r["snr_db"] == snr and r["seed"] == seed]))
            for seed in config.seeds]) for metric in ("auc", "pauc")}
    summary = {"model": "ImageNet-pretrained MobileNetV3-Small, fine-tuned for normal machine ID",
        "macro_auc": statistics([r["macro_auc"] for r in per_seed]),
        "macro_pauc": statistics([r["macro_pauc"] for r in per_seed]),
        "per_seed": per_seed, "per_machine": by_machine, "per_snr": by_snr,
        "seeds": list(config.seeds), "runs": results,
        "aggregation": "Average AUC/pAUC over 48 machine-type/ID/SNR cells within each seed, then mean/population SD/variance across seeds",
        "pauc_convention": "sklearn standardized partial AUC, max_fpr=0.1; chance=0.5",
        "uncertainty": "Initialization variability on one fixed split; not a confidence interval",
        "thresholds": "Per-type/ID training-normal 95th percentile; fixed across evaluation SNRs",
        "scope": "All SNRs represented in training; held-out source recordings; no unseen-type/fault-category claim"}
    write_json(root / "summary.json", summary)
    with (root / "results_by_seed_machine_snr_id.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cells[0]))
        writer.writeheader()
        writer.writerows(cells)
    create_figures(root, cells, config)
    format_metric = lambda cell: f"{100 * cell['mean']:.2f} ± {100 * cell['std']:.2f}"
    lines = ["# MobileNetV3-Small transfer-learning baseline", "",
        "All four machine types, all four IDs, and all three SNRs; three initialization seeds on the frozen full-dataset split.", "",
        "| Machine | AUC (%) | Standardized pAUC (%) |", "| --- | ---: | ---: |"]
    for machine, metrics in by_machine.items():
        lines.append(f"| {machine.title()} | {format_metric(metrics['macro_auc'])} | {format_metric(metrics['macro_pauc'])} |")
    lines += [f"| **Overall** | **{format_metric(summary['macro_auc'])}** | **{format_metric(summary['macro_pauc'])}** |", "",
        "Values are mean ± population SD over initialization seeds, in percentage points. Compute metrics per ID/SNR before averaging; do not pool unrelated ID scores.", "",
        "| SNR | AUC (%) | Standardized pAUC (%) |", "| --- | ---: | ---: |"]
    for snr, metrics in by_snr.items():
        lines.append(f"| {int(snr):+d} dB | {format_metric(metrics['macro_auc'])} | {format_metric(metrics['macro_pauc'])} |")
    lines += ["", "## Protocol", "",
        "- Official torchvision ImageNet-1K pretrained MobileNetV3-Small; four separate machine-type models, each predicting four machine IDs.",
        "- First convolution's RGB kernels summed into one channel; final 1,000-class layer replaced by four outputs. Native 128 × 313 log-mel input; no RGB transforms, resizing, or image augmentation.",
        "- Training-normal per-mel normalization; 30,948 training normals and 4,395 validation normals across the four types. All 9,900 fault versions stay in the 18,714-clip test set.",
        f"- {config.warmup_epochs} epochs of classifier warmup with frozen backbone/BatchNorm; then full-network fine-tuning, up to {config.epochs} total epochs. AdamW: head warmup lr={config.head_warmup_lr}, fine-tuning head lr={config.head_finetune_lr}, backbone lr={config.backbone_lr}, weight decay={config.weight_decay}; batch={config.batch_size}; gradient norm clipped at 5.",
        f"- Healthy-validation cross-entropy selects checkpoints. Early stopping after {config.patience} fine-tuning epochs without improvement of at least {config.min_delta}. No fault-label hyperparameter or checkpoint selection.",
        "- bfloat16 autocast during GPU training; float32 validation/scoring forward passes. Stable float64 negative log probability of the known ID supplies the anomaly score.",
        "- Per-ID thresholds use the 95th percentile of training-normal scores pooled across SNRs; thresholds remain fixed across test SNRs. pAUC uses FPR ≤ 0.1 with sklearn standardization.", "",
        "## Selected checkpoints", "",
        "| Machine | Seed | Selected / trained epochs | Validation ID accuracy (%) | Runtime (s) |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for r in results:
        lines.append(f"| {r['machine_type']} | {r['seed']} | {r['selected_epoch']} / {r['trained_epochs']} | {100*r['selected_validation_id_accuracy']:.2f} | {r['elapsed_seconds']:.1f} |")
    lines += ["", "## Interpretation and limitations", "",
        "Healthy-ID accuracy is the proxy-task result, not fault-detection accuracy. Interpret the AUC/pAUC tables and false-negative/false-positive counts independently of ID accuracy. Thresholds calibrated on training normals need not produce exactly 5% false positives on held-out audio.", "",
        "This is transfer from natural images, not an audio-pretrained backbone. All three noise levels are represented in training. The SNR versions share underlying recordings and are not independent observations. Seeds measure initialization variability; the split is fixed. Earlier fan pilot test results were inspected, so this is not a fresh blind benchmark. The protocol was frozen before this run's test evaluation.", "",
        "Each machine/seed directory contains model.pt (best checkpoint plus normalizer/feature metadata), last.pt (resumable optimizer/RNG state), history.jsonl, train_scores.csv, scores.csv, metrics.json, learning_curve.png, detection_results.png, and confusion_matrices.png. Per-SNR confusion counts are in metrics.json and the root CSV. protocol.json and source_snapshot preserve the executed configuration and code.", ""]
    (root / "RESULTS.md").write_text("\n".join(lines))
    print(json.dumps({"macro_auc": summary["macro_auc"], "macro_pauc": summary["macro_pauc"]}, indent=2), flush=True)
    return summary


def create_figures(root, cells, config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), layout="constrained")
    x = np.arange(len(config.machine_types))
    for ax, metric, title in zip(axes, ("auc", "pauc"), ("ROC-AUC (%)", "Standardized pAUC (%)")):
        for index, snr in enumerate((-6, 0, 6)):
            stats = [statistics([np.mean([r[metric] for r in cells if r["machine_type"] == machine
                and r["snr_db"] == snr and r["seed"] == seed]) for seed in config.seeds]) for machine in config.machine_types]
            ax.bar(x + (index - 1) * 0.25, [100*s["mean"] for s in stats], 0.24,
                   yerr=[100*s["std"] for s in stats], capsize=2, label=f"{snr:+d} dB")
        ax.axhline(50, color="gray", linestyle="--", linewidth=0.8)
        ax.set(xticks=x, xticklabels=[m.title() for m in config.machine_types], ylim=(0, 105), ylabel=title)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, ncol=3)
    for extension in ("png", "pdf"):
        fig.savefig(root / f"mobilenet_results.{extension}", dpi=200)
    plt.close(fig)
