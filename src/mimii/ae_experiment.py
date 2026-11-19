"""Full-corpus, resumable DCASE-style autoencoder experiment.

Primary evaluation uses epoch 100; best healthy-validation MSE is a separately
labelled secondary analysis. No anomaly labels select weights or settings.
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import csv
import gc
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from .ae_windows import CudaGraphStep, WindowBank, score_bank, train_epoch
from .ae_report import audit_run, publish_report
from .evaluation import detection_metrics, evaluate_scores
from .models import DenseAutoencoder
from .training_data import FeatureStore


@dataclass(frozen=True)
class AEConfig:
    feature_root: str = "features/full_dataset_logmel128_seed42"
    output_dir: str = "runs/dcase2020_ae_full_dataset"
    machine_types: tuple = ("fan", "pump", "slider", "valve")
    seeds: tuple = (42, 43, 44)
    epochs: int = 100
    batch_size: int = 512
    evaluation_batch_size: int = 8192
    learning_rate: float = 0.001
    adam_epsilon: float = 1e-7
    normalization: str = "none"
    latent_dimensions: int = 8
    threshold_quantile: float = 0.95
    device: str = "cuda"
    cuda_graphs: bool = True
    cpu_threads: int = 2

    def __post_init__(self):
        if min(self.epochs, self.evaluation_batch_size, self.latent_dimensions, self.cpu_threads) < 1:
            raise ValueError("Positive dimensions and schedule required")
        if self.batch_size < 2 or min(self.learning_rate, self.adam_epsilon) <= 0:
            raise ValueError("Invalid optimizer settings")
        if self.normalization != "none" or not 0 < self.threshold_quantile < 1:
            raise ValueError("This baseline requires raw log-mel and a valid quantile")
        if self.device not in {"cpu", "cuda"} or (self.cuda_graphs and self.device != "cuda"):
            raise ValueError("Invalid device/graph combination")
        for values in (self.machine_types, self.seeds):
            if not values or len(values) != len(set(values)):
                raise ValueError("Machine types and seeds must be nonempty and distinct")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save_checkpoint(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def state_copy(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assert_normal_splits(train, validation):
    for rows, expected in ((train, "train"), (validation, "validation")):
        if not rows or any(row["label"] != 0 or row["split"] != expected for row in rows):
            raise ValueError("AE fitting/selection require normal recordings in the correct split")


def score_rows(model, store, rows, device, batch_size, chunk_clips=64):
    scores = []
    for start in range(0, len(rows), chunk_clips):
        bank = WindowBank.from_store(store, rows[start:start + chunk_clips], device)
        scores.append(score_bank(model, bank, batch_size))
    return np.concatenate(scores)


def evaluate(model, store, train, train_bank, config, output):
    test = store.split("test")
    # Compute the quantile in float64 so it is exactly reproducible from CSV.
    training_scores = score_bank(model, train_bank, config.evaluation_batch_size).astype(np.float64)
    test_scores = score_rows(model, store, test, config.device, config.evaluation_batch_size)
    pooled = evaluate_scores(test, test_scores, train, training_scores, output, config.threshold_quantile)
    cells = []
    for snr in sorted({r["snr_db"] for r in test}):
        for mid in sorted({r["machine_id"] for r in test}):
            mask = np.array([r["snr_db"] == snr and r["machine_id"] == mid for r in test])
            labels = [r["label"] for r, selected in zip(test, mask) if selected]
            metrics = detection_metrics(labels, test_scores[mask], pooled["per_id"][mid]["threshold"])
            cells.append({"snr_db": snr, "machine_id": mid, **metrics})
    result = {"cells": cells, "macro_auc": float(np.mean([c["auc"] for c in cells])),
              "macro_pauc": float(np.mean([c["pauc"] for c in cells])),
              "test_clips": len(test), "train_clips": len(train),
              "aggregation": "equal-weight machine-ID x SNR cells",
              "pauc_convention": "sklearn standardized partial AUC, max_fpr=0.1"}
    write_json(output / "metrics_by_snr.json", result)
    return result


def train_seed(config, machine, seed, store, train_bank, validation_bank, root):
    output = root / machine / f"seed_{seed}"
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "result.json"
    if result_path.exists():
        audit_run(output, store, config)
        return json.loads(result_path.read_text())
    train, validation = store.split("train"), store.split("validation")
    assert_normal_splits(train, validation)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = DenseAutoencoder(store.shape[0] * store.context_frames, config.latent_dimensions).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, eps=config.adam_epsilon,
                                 fused=config.device == "cuda", capturable=config.device == "cuda")
    generator = torch.Generator().manual_seed(seed)
    history, best_loss, best_epoch, best_state = [], float("inf"), 0, None
    resume = output / "resume.pt"
    if resume.exists():
        checkpoint = torch.load(resume, map_location=config.device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        generator.set_state(checkpoint["shuffle_rng"].cpu())
        history = checkpoint["history"]
        best_loss, best_epoch, best_state = checkpoint["best_loss"], checkpoint["best_epoch"], checkpoint["best_model"]
        del checkpoint
    graph = CudaGraphStep(model, optimizer, train_bank, config.batch_size) if config.cuda_graphs else None
    print(f"{machine} seed {seed}: {len(train)} normal clips, {len(train_bank)} windows/epoch; resume epoch {len(history)}", flush=True)
    for epoch in range(len(history) + 1, config.epochs + 1):
        start = time.monotonic()
        loss = train_epoch(model, optimizer, train_bank, config.batch_size, generator, graph)
        validation_loss = float(score_bank(model, validation_bank, config.evaluation_batch_size).astype(np.float64).mean())
        if validation_loss < best_loss:
            best_loss, best_epoch, best_state = validation_loss, epoch, state_copy(model)
        row = {"epoch": epoch, "train_loss": loss, "validation_loss": validation_loss,
               "seconds": time.monotonic() - start}
        history.append(row)
        save_checkpoint(resume, {"model": state_copy(model), "optimizer": optimizer.state_dict(),
                                "shuffle_rng": generator.get_state(), "history": history,
                                "best_model": best_state, "best_loss": best_loss, "best_epoch": best_epoch})
        write_json(output / "history.json", history)
        write_json(root / "status.json", {"state": "training", "machine": machine, "seed": seed,
                   "epoch": epoch, "epochs": config.epochs,
                   "completed_runs": len(list(root.glob("*/seed_*/result.json"))),
                   "planned_runs": len(config.machine_types) * len(config.seeds),
                   "updated_utc": datetime.now(timezone.utc).isoformat(), **row})
        print(f"{machine} seed={seed} epoch={epoch}/{config.epochs} train={loss:.5f} val={validation_loss:.5f} seconds={row['seconds']:.1f}", flush=True)
    del graph
    gc.collect()
    metadata = {"config": asdict(config), "machine_type": machine, "seed": seed,
                "manifest_sha256": store.manifest_sha256, "feature_config": store.config,
                "input_dimensions": store.shape[0] * store.context_frames,
                "normalization": "none", "score": "mean squared reconstruction error over all windows and features"}
    save_checkpoint(output / "final.pt", {**metadata, "model": state_copy(model), "epoch": config.epochs,
                                          "selection": "fixed final epoch"})
    save_checkpoint(output / "best_validation.pt", {**metadata, "model": best_state, "epoch": best_epoch,
                                                    "selection": "minimum healthy-validation MSE"})
    results = {}
    for name in ("final", "best_validation"):
        if name == "best_validation":
            model.load_state_dict(best_state)
        results[name] = evaluate(model, store, train, train_bank, config, output / name)
        print(f"{machine} seed={seed} {name}: AUC={results[name]['macro_auc']:.5f} pAUC={results[name]['macro_pauc']:.5f}", flush=True)
    result = {"machine_type": machine, "seed": seed, "epochs": config.epochs,
              "best_validation_epoch": best_epoch, "training_seconds": sum(r["seconds"] for r in history),
              "parameters": sum(p.numel() for p in model.parameters()), **results}
    write_json(result_path, result)
    audit_run(output, store, config)
    return result


def summarize(root, config):
    """Generate explicit partial/final tables; do not compare incomplete seed sets."""
    results = [json.loads(p.read_text()) for p in sorted(root.glob("*/seed_*/result.json"))]
    complete = len(results) == len(config.machine_types) * len(config.seeds)
    summary = {"complete": complete, "completed_runs": len(results), "primary_checkpoint": "final",
               "standard_deviation": "population SD across initialization seeds (ddof=0), matching MobileNet; fixed data split", "views": {}}
    rows = []
    for view in ("final", "best_validation"):
        by_machine = {}
        for machine in config.machine_types:
            selected = [r for r in results if r["machine_type"] == machine]
            if not selected:
                continue
            by_machine[machine] = {"seeds": [r["seed"] for r in selected]}
            for metric in ("auc", "pauc"):
                values = [r[view][f"macro_{metric}"] for r in selected]
                by_machine[machine][metric] = {"mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=0)) if len(values) > 1 else None}
            for r in selected:
                rows.extend({"checkpoint": view, "machine_type": machine, "seed": r["seed"], **c} for c in r[view]["cells"])
        overall = {}
        if complete:
            for metric in ("auc", "pauc"):
                values = [np.mean([r[view][f"macro_{metric}"] for r in results if r["seed"] == seed]) for seed in config.seeds]
                overall[metric] = {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=0)) if len(values) > 1 else None}
        summary["views"][view] = {"by_machine": by_machine, "overall": overall}
    write_json(root / "summary.json", summary)
    if rows:
        with (root / "per_id_snr_seed.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = ["# DCASE 2020–style autoencoder results", "", f"Completed fits: {len(results)}/{len(config.machine_types) * len(config.seeds)}.",
             "", "Primary: fixed final epoch. Secondary: minimum healthy-validation reconstruction MSE.",
             "These are custom original-MIMII splits, not official DCASE challenge scores.", "",
             "| Checkpoint | Machine | Seeds | AUC (%) | pAUC (%) |", "|---|---|---|---:|---:|"]
    def value(cell):
        suffix = f" ± {cell['std'] * 100:.2f}" if cell["std"] is not None else " (one seed)"
        return f"{cell['mean'] * 100:.2f}{suffix}"
    for view, data in summary["views"].items():
        for machine, metrics in data["by_machine"].items():
            lines.append(f"| {view} | {machine} | {metrics['seeds']} | {value(metrics['auc'])} | {value(metrics['pauc'])} |")
        if data["overall"]:
            lines.append(f"| {view} | Overall | all | {value(data['overall']['auc'])} | {value(data['overall']['pauc'])} |")
    (root / "RESULTS.md").write_text("\n".join(lines) + "\n")
    return summary


def run(config):
    root = Path(config.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config.cpu_threads)
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; do not silently change the experiment")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    files = [Path(__file__), Path(__file__).with_name("ae_windows.py"), Path(__file__).with_name("models.py"),
             Path(__file__).with_name("training_data.py"), Path(__file__).with_name("evaluation.py"),
             Path(__file__).with_name("dataset.py"), Path(__file__).with_name("features.py"),
             Path(__file__).with_name("ae_report.py"),
             Path("scripts/train_autoencoder.py")]
    provenance = {"config": json.loads(json.dumps(asdict(config))), "source_sha256": {str(p): sha256(p) for p in files},
                  "manifests": {machine: sha256(Path(config.feature_root) / "by_machine" / machine / "manifest.jsonl") for machine in config.machine_types},
                  "versions": {name: version(name) for name in ("torch", "numpy", "scikit-learn")},
                  "gpu": torch.cuda.get_device_name() if config.device == "cuda" else None}
    protocol_path = root / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != provenance:
            raise ValueError("Experiment code, runtime, features or settings changed; use a new output directory")
    else:
        write_json(protocol_path, provenance)
        snapshot = root / "source_snapshot"
        snapshot.mkdir(exist_ok=True)
        for path in files:
            shutil.copy2(path, snapshot / path.name)
        write_json(snapshot / "config.json", asdict(config))
    for machine in config.machine_types:
        if all((root / machine / f"seed_{seed}" / "result.json").exists() for seed in config.seeds):
            store = FeatureStore(Path(config.feature_root) / "by_machine" / machine, "none")
            for seed in config.seeds:
                audit_run(root / machine / f"seed_{seed}", store, config)
            continue
        store = FeatureStore(Path(config.feature_root) / "by_machine" / machine, "none")
        assert_normal_splits(store.split("train"), store.split("validation"))
        print(f"Loading {machine} feature banks on {config.device}", flush=True)
        train_bank = WindowBank.from_store(store, store.split("train"), config.device)
        validation_bank = WindowBank.from_store(store, store.split("validation"), config.device)
        for seed in config.seeds:
            train_seed(config, machine, seed, store, train_bank, validation_bank, root)
            summarize(root, config)
        del train_bank, validation_bank
        gc.collect()
        if config.device == "cuda":
            torch.cuda.empty_cache()
    summary = summarize(root, config)
    publish_report(root, summary, config)
    write_json(root / "status.json", {"state": "complete", "completed_runs": summary["completed_runs"],
                                     "updated_utc": datetime.now(timezone.utc).isoformat()})
    return summary
