"""Repeated-seed custom CNN and fixed pooling ablation on the complete corpus."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

from .acoustic_cnn import AcousticResidualStatsCNN
from .ae_experiment import write_json
from .finetune import FinetuneConfig, configure_runtime, train_seed
from .report_transfer import statistics
from .training_data import FeatureStore


@dataclass(frozen=True)
class AcousticConfig(FinetuneConfig):
    include_std: bool = True
    initialization: str = "random, no pretrained weights"


def variant_config(study, include_std):
    return AcousticConfig(feature_root=study["feature_root"], output_dir=str(Path(study["output_dir"]) / ("stats" if include_std else "mean_only")),
        pretrained_path="", machine_types=tuple(study["machine_types"]),
        seeds=tuple(study["seeds"]) if include_std else (study["ablation_seed"],),
        epochs=study["epochs"], warmup_epochs=0, patience=study["patience"], min_delta=study["min_delta"],
        batch_size=study["batch_size"], evaluation_batch_size=study["batch_size"],
        backbone_lr=study["learning_rate"], head_finetune_lr=study["learning_rate"], head_warmup_lr=study["learning_rate"],
        weight_decay=study["weight_decay"], threshold_quantile=study["threshold_quantile"], include_std=include_std)


def summarize_variant(root, config):
    results = [json.loads((root / m / f"seed_{s}" / "metrics.json").read_text()) for m in config.machine_types for s in config.seeds]
    cells = [cell for r in results for cell in r["cells"]]
    per_seed = [{"seed": seed, **{f"macro_{metric}": float(np.mean([r[metric] for r in cells if r["seed"] == seed]))
                                  for metric in ("auc", "pauc")}} for seed in config.seeds]
    summary = {"model": "AcousticResidualStatsCNN" if config.include_std else "AcousticResidualStatsCNN mean-only ablation",
               "parameters": 177724, "pretraining": "none", "seeds": list(config.seeds), "n_seeds": len(config.seeds),
               "macro_auc": statistics([r["macro_auc"] for r in per_seed]), "macro_pauc": statistics([r["macro_pauc"] for r in per_seed]),
               "per_seed": per_seed, "runs": results,
               "per_machine": {m: {f"macro_{metric}": statistics([np.mean([c[metric] for c in cells if c["machine_type"] == m and c["seed"] == seed])
                                     for seed in config.seeds]) for metric in ("auc", "pauc")} for m in config.machine_types},
               "aggregation": "equal-weight ID/SNR/type cells within each seed, then mean and population SD across seeds",
               "single_seed_warning": "A singleton's numerical SD=0 is not a variability estimate"}
    write_json(root / "summary.json", summary)
    import csv
    with (root / "results_by_seed_machine_snr_id.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cells[0]))
        writer.writeheader()
        writer.writerows(cells)
    return summary


def publish(study, summaries):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(study["output_dir"])
    destination = Path("docs/results/acoustic_residual_stats_cnn")
    destination.mkdir(parents=True, exist_ok=True)
    lines = ["# Custom acoustic CNN results", "", "A 177,724-parameter CNN trained from scratch on healthy machine IDs.",
             "The main model uses temporal mean/std pooling and three seeds. The mean-only ablation uses seed 42; its std input is zeroed while the classifier shape is retained.", "",
             "| Variant | Machine | AUC (%) | pAUC (%) |", "|---|---|---:|---:|"]
    def format_metric(value, n):
        return f"{value['mean']*100:.2f} ± {value['std']*100:.2f}" if n > 1 else f"{value['mean']*100:.2f} (seed 42)"
    for name, summary in summaries.items():
        for machine, metrics in summary["per_machine"].items():
            lines.append(f"| {name} | {machine} | {format_metric(metrics['macro_auc'],summary['n_seeds'])} | {format_metric(metrics['macro_pauc'],summary['n_seeds'])} |")
        lines.append(f"| {name} | Overall | {format_metric(summary['macro_auc'],summary['n_seeds'])} | {format_metric(summary['macro_pauc'],summary['n_seeds'])} |")
    stats42 = next(r for r in summaries["stats"]["per_seed"] if r["seed"] == study["ablation_seed"])
    mean42 = summaries["mean_only"]["per_seed"][0]
    ablation = {"seed": study["ablation_seed"], "statistics_pool": stats42, "mean_only": mean42,
                "change_percentage_points": {metric: 100 * (stats42[metric] - mean42[metric]) for metric in ("macro_auc", "macro_pauc")},
                "limit": "Single paired seed; removing std also removes that branch's effective input capacity, despite unchanged nominal parameter count."}
    lines.extend(["", "## Paired pooling ablation", "",
                  f"Seed 42: mean/std AUC/pAUC = {100*stats42['macro_auc']:.2f}/{100*stats42['macro_pauc']:.2f}%; "
                  f"mean-only = {100*mean42['macro_auc']:.2f}/{100*mean42['macro_pauc']:.2f}%.",
                  "Compare the matched seed above for the ablation, not the three-seed mean against a single run.", "",
                  "## Interpretation", "", "Architecture combines established residual, depthwise/inverted-bottleneck and channel-attention components. No architectural novelty claim is made.",
                  "Mean/std pooling is a hypothesis about temporal variability, not a guarantee of improved fault sensitivity. The anomaly score remains known-ID negative log probability.",
                  "All recordings and frozen splits are retained. Checkpoints use healthy-validation cross-entropy; thresholds use training-normal quantiles. Test outcomes do not select configurations.",
                  "The architecture was proposed after earlier test results and separability diagnostics were inspected. This is exploratory work on a reused test set, not a fresh blind benchmark.",
                  "Comparisons with ImageNet-initialized models also differ in pretraining, parameter count and training schedules."])
    write_json(root / "summary.json", {"variants": summaries, "pooling_ablation": ablation})
    (root / "RESULTS.md").write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    x = np.arange(len(study["machine_types"]))
    for ax, metric in zip(axes, ("auc", "pauc")):
        for offset, variant, label in ((-.18, "stats", "Mean/std (3 seeds)"), (.18, "mean_only", "Mean-only (seed 42)")):
            vals = [summaries[variant]["per_machine"][m][f"macro_{metric}"] for m in study["machine_types"]]
            ax.bar(x + offset, [v["mean"] * 100 for v in vals], .35, yerr=[v["std"] * 100 for v in vals] if variant == "stats" else None, capsize=3, label=label)
        ax.set(xticks=x, xticklabels=study["machine_types"], ylim=(0, 100), ylabel=f"{metric.upper()} (%)")
    axes[0].legend(fontsize=8)
    for ext in ("png", "pdf"):
        fig.savefig(root / f"custom_cnn_results.{ext}", dpi=180)
    plt.close(fig)
    for name in ("summary.json", "RESULTS.md", "protocol.json", "custom_cnn_results.png", "custom_cnn_results.pdf"):
        shutil.copy2(root / name, destination / name)
    for variant in ("stats", "mean_only"):
        for name in ("validation.json", "summary.json", "results_by_seed_machine_snr_id.csv", "protocol.json"):
            shutil.copy2(root / variant / name, destination / f"{variant}_{name}")


def run(study):
    root = Path(study["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    configs = {name: variant_config(study, name == "stats") for name in ("stats", "mean_only")}
    device = configure_runtime(configs["stats"])
    source_names = ("acoustic_cnn", "custom_cnn_experiment", "finetune", "mobilenet", "training_data", "evaluation", "features", "dataset", "train", "config", "models", "preprocess", "report_transfer", "ae_experiment", "ae_report", "ae_windows")
    sources = [Path("src/mimii") / f"{name}.py" for name in source_names] + [Path("scripts/train_acoustic_cnn.py"), Path("scripts/verify_mobilenet_results.py")]
    source_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    master = FeatureStore(study["feature_root"])
    if {r["machine_type"] for r in master.records} != set(study["machine_types"]):
        raise ValueError("Custom study must cover every machine type")
    identity = {"study": study, "source_sha256": source_hashes, "feature_manifest_sha256": master.manifest_sha256,
                "versions": {name: version(name) for name in ("torch", "numpy", "scikit-learn")}}
    path = root / "protocol.json"
    if path.exists():
        protocol = json.loads(path.read_text())
        if protocol["identity"] != identity:
            raise ValueError("Custom study code/settings/features changed")
    else:
        now = time.time()
        protocol = {"identity": identity, "started_unix": now, "deadline_unix": now + study["wall_seconds"],
                    "hypothesis": "Temporal feature variability can aid normal-only acoustic anomaly detection beyond temporal means",
                    "test_reuse": "Exploratory architecture motivated after inspection of earlier test results; no fresh blind evaluation claim"}
        write_json(path, protocol)
    for name, cfg in configs.items():
        variant_root = Path(cfg.output_dir)
        variant_root.mkdir(exist_ok=True)
        vp = {"config": json.loads(json.dumps(asdict(cfg))), "source_sha256": source_hashes,
              "feature_manifest_sha256": master.manifest_sha256,
              "normalizer_sha256": {m: hashlib.sha256((Path(study["feature_root"]) / "by_machine" / m / "normalizer.npz").read_bytes()).hexdigest() for m in study["machine_types"]},
              "selection": "lowest healthy-validation cross-entropy; no fault-label tuning"}
        if (variant_root / "protocol.json").exists():
            if json.loads((variant_root / "protocol.json").read_text()) != vp:
                raise ValueError("Custom variant protocol changed")
        else:
            write_json(variant_root / "protocol.json", vp)
            for source in sources:
                target = variant_root / "source_snapshot" / source
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
    write_json(root / "status.json", {"state": "loading_features", "planned_fits": 16,
               "deadline_utc": datetime.fromtimestamp(protocol["deadline_unix"], timezone.utc).isoformat()})
    stores, inputs = {}, {}
    for machine in study["machine_types"]:
        print(f"Loading full train/validation normals: {machine}", flush=True)
        store = FeatureStore(Path(study["feature_root"]) / "by_machine" / machine, "train_per_mel")
        train, validation = store.split("train"), store.split("validation")
        if any(r["label"] != 0 for r in train + validation):
            raise ValueError("Non-normal recording in training/validation")
        stores[machine] = store
        inputs[machine] = (torch.from_numpy(store.clip_array(train)), torch.tensor([r["id_class"] for r in train]),
                           torch.from_numpy(store.clip_array(validation)), torch.tensor([r["id_class"] for r in validation]))
    # Get a complete primary/ablation seed pair before spending time on repeats.
    schedule = [("stats", study["seeds"][0]), ("mean_only", study["ablation_seed"])] + [("stats", seed) for seed in study["seeds"][1:]]
    completed = []
    for variant, seed in schedule:
        cfg = configs[variant]
        for machine in study["machine_types"]:
            if time.time() >= protocol["deadline_unix"] - study["finalization_reserve_seconds"]:
                raise TimeoutError("Custom CNN execution allowance exhausted; completed fits preserved")
            write_json(root / "status.json", {"state": "training", "variant": variant, "seed": seed, "machine": machine,
                       "completed_fits": len(completed), "planned_fits": len(schedule) * len(study["machine_types"]),
                       "fit_status": str(Path(cfg.output_dir) / machine / f"seed_{seed}" / "status.json")})
            result = train_seed(cfg, machine, seed, stores[machine], device, inputs[machine],
                                model_factory=lambda classes: AcousticResidualStatsCNN(classes, include_std=cfg.include_std),
                                model_name=f"acoustic_residual_{variant}", deadline=protocol["deadline_unix"] - study["finalization_reserve_seconds"])
            completed.append({"variant": variant, **result})
            write_json(root / "partial_results.json", completed)
            torch.cuda.empty_cache()
    summaries = {}
    for variant, cfg in configs.items():
        summaries[variant] = summarize_variant(Path(cfg.output_dir), cfg)
        subprocess.run([sys.executable, "scripts/verify_mobilenet_results.py", "--run-dir", cfg.output_dir], check=True)
    publish(study, summaries)
    write_json(root / "status.json", {"state": "complete", "completed_fits": len(completed),
               "elapsed_seconds": time.time() - protocol["started_unix"]})
