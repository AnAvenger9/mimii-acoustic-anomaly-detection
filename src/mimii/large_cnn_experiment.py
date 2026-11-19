"""Repeated-seed wider acoustic CNN capacity study on the complete corpus."""

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

from .large_acoustic_cnn import LargeAcousticResidualStatsCNN
from .ae_experiment import write_json
from .finetune import FinetuneConfig, configure_runtime, train_seed
from .report_transfer import statistics
from .training_data import FeatureStore


@dataclass(frozen=True)
class LargeAcousticConfig(FinetuneConfig):
    architecture: str = "large_acoustic_residual_stats_v1"
    include_std: bool = True
    initialization: str = "random, no pretrained weights"


def variant_config(study):
    return LargeAcousticConfig(feature_root=study["feature_root"], output_dir=str(Path(study["output_dir"]) / "stats"),
        pretrained_path="", machine_types=tuple(study["machine_types"]),
        seeds=tuple(study["seeds"]),
        epochs=study["epochs"], warmup_epochs=0, patience=study["patience"], min_delta=study["min_delta"],
        batch_size=study["batch_size"], evaluation_batch_size=study["batch_size"],
        backbone_lr=study["learning_rate"], head_finetune_lr=study["learning_rate"], head_warmup_lr=study["learning_rate"],
        weight_decay=study["weight_decay"], threshold_quantile=study["threshold_quantile"], include_std=True)


def summarize_variant(root, config):
    results = [json.loads((root / m / f"seed_{s}" / "metrics.json").read_text()) for m in config.machine_types for s in config.seeds]
    cells = [cell for r in results for cell in r["cells"]]
    per_seed = [{"seed": seed, **{f"macro_{metric}": float(np.mean([r[metric] for r in cells if r["seed"] == seed]))
                                  for metric in ("auc", "pauc")}} for seed in config.seeds]
    summary = {"model": "LargeAcousticResidualStatsCNN",
               "parameters": sum(p.numel() for p in LargeAcousticResidualStatsCNN().parameters()), "pretraining": "none", "seeds": list(config.seeds), "n_seeds": len(config.seeds),
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


def operating_point(cells):
    counts = {key: sum(c[key] for c in cells) for key in ("tp", "fn", "fp", "tn")}
    return {**counts, "fault_recall": counts["tp"] / (counts["tp"] + counts["fn"]),
            "false_positive_rate": counts["fp"] / (counts["fp"] + counts["tn"])}


def publish(study, summaries):
    root = Path(study["output_dir"])
    summary = summaries["stats"]
    cells = [c for run in summary["runs"] for c in run["cells"]]
    for row in summary["per_seed"]:
        row["operating_point"] = operating_point([c for c in cells if c["seed"] == row["seed"]])
    summary["operating_point"] = {metric: statistics([r["operating_point"][metric] for r in summary["per_seed"]])
                                  for metric in ("fault_recall", "false_positive_rate")}
    summary["operating_point_aggregation"] = "Clip-weighted within each seed; mean and population SD across seeds. Fixed training-normal 95th percentile thresholds."
    for machine, metrics in summary["per_machine"].items():
        points = [operating_point([c for c in cells if c["seed"] == seed and c["machine_type"] == machine]) for seed in summary["seeds"]]
        metrics["operating_point"] = {metric: statistics([p[metric] for p in points]) for metric in ("fault_recall", "false_positive_rate")}
    small_path = Path("runs/acoustic_residual_stats_cnn/stats/summary.json")
    small = json.loads(small_path.read_text())
    paired = []
    for large_row in summary["per_seed"]:
        small_row = next(r for r in small["per_seed"] if r["seed"] == large_row["seed"])
        paired.append({"seed": large_row["seed"], **{metric: 100 * (large_row[metric] - small_row[metric]) for metric in ("macro_auc", "macro_pauc")}})
    summary["capacity_comparison"] = {"small_summary_sha256": hashlib.sha256(small_path.read_bytes()).hexdigest(),
        "paired_change_percentage_points": paired,
        "mean_change_percentage_points": {metric: statistics([r[metric] for r in paired]) for metric in ("macro_auc", "macro_pauc")},
        "interpretation": "Exploratory width/head-capacity comparison with identical data, depth, pooling and training settings. Test data were inspected in earlier experiments; no blind benchmark claim."}
    write_json(root / "stats/summary.json", summary)
    write_json(root / "summary.json", summary)
    def fmt(v):
        return f"{100*v['mean']:.2f} ± {100*v['std']:.2f}"
    lines = ["# Larger acoustic CNN results", "", f"Parameters: {summary['parameters']:,}; scratch initialization; seeds {summary['seeds']}.", "",
             "| Machine | AUC (%) | pAUC (%) | Fault recall (%) | False alarms (%) |",
             "|---|---:|---:|---:|---:|"]
    for machine, metrics in list(summary["per_machine"].items()) + [("Overall", summary)]:
        op = metrics["operating_point"]
        lines.append(f"| {machine} | {fmt(metrics['macro_auc'])} | {fmt(metrics['macro_pauc'])} | {fmt(op['fault_recall'])} | {fmt(op['false_positive_rate'])} |")
    lines += ["", "Values are mean ± population SD across seeds. AUC/pAUC use equal-weight type/ID/SNR cells; recall/false alarms count individual clips. Standardized pAUC uses max FPR=0.1.",
              "", "This detects abnormal sound, not a specific fault mechanism. Machine-ID accuracy is a separate auxiliary task.",
              "Checkpoints minimize healthy-validation cross-entropy; decisions use per-ID training-normal thresholds. No abnormal clips are used for training or threshold selection.",
              "", "## Matched small-to-large comparison", "", "| Seed | AUC change (percentage points) | pAUC change (percentage points) |", "|---|---:|---:|"]
    for row in paired:
        lines.append(f"| {row['seed']} | {row['macro_auc']:+.2f} | {row['macro_pauc']:+.2f} |")
    lines += ["", summary["capacity_comparison"]["interpretation"], "Three seeds describe initialization variability, not independent recordings or deployment uncertainty."]
    (root / "RESULTS.md").write_text("\n".join(lines) + "\n")
    destination = Path("docs/results/large_acoustic_residual_stats_cnn")
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("summary.json", "RESULTS.md", "protocol.json"):
        shutil.copy2(root / name, destination / name)
    for name in ("validation.json", "results_by_seed_machine_snr_id.csv", "protocol.json"):
        shutil.copy2(root / "stats" / name, destination / f"stats_{name}")


def run(study):
    root = Path(study["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    configs = {"stats": variant_config(study)}
    device = configure_runtime(configs["stats"])
    source_names = ("acoustic_cnn", "large_acoustic_cnn", "large_cnn_experiment", "finetune", "mobilenet", "training_data", "evaluation", "features", "dataset", "train", "config", "models", "preprocess", "report_transfer", "ae_experiment", "ae_report", "ae_windows")
    sources = [Path("src/mimii") / f"{name}.py" for name in source_names] + [Path("scripts/train_large_acoustic_cnn.py"), Path("scripts/verify_mobilenet_results.py")]
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
                    "hypothesis": "Increasing feature and classifier width may improve normal-only acoustic anomaly detection",
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
    write_json(root / "status.json", {"state": "loading_features", "planned_fits": len(study["seeds"]) * len(study["machine_types"]),
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
    # Complete all four machine types for one seed before the next repeat.
    schedule = [("stats", seed) for seed in study["seeds"]]
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
                                model_factory=lambda classes: LargeAcousticResidualStatsCNN(classes),
                                model_name="large_acoustic_residual_stats", deadline=protocol["deadline_unix"] - study["finalization_reserve_seconds"])
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
