"""Reproducible normal-only MobileNet transfer learning across the full corpus."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
from torch import nn

from .evaluation import detection_metrics, evaluate_scores
from .mobilenet import WEIGHTS_URL, build_mobilenet, make_optimizer, set_training_stage, weights_sha256
from .preprocess import write_json
from .train import batches, choose_device, plot_history, score_recordings
from .training_data import FeatureStore


@dataclass(frozen=True)
class FinetuneConfig:
    feature_root: str = "features/full_dataset_logmel128_seed42"
    output_dir: str = "runs/mobilenet_v3_small_full_dataset"
    pretrained_path: str = "checkpoints/mobilenet_v3_small-047dcff4.pth"
    machine_types: tuple = ("fan", "pump", "slider", "valve")
    seeds: tuple = (42, 43, 44)
    epochs: int = 20
    warmup_epochs: int = 2
    patience: int = 5
    min_delta: float = 1e-4
    batch_size: int = 32
    evaluation_batch_size: int = 32
    head_warmup_lr: float = 1e-3
    head_finetune_lr: float = 3e-4
    backbone_lr: float = 1e-4
    weight_decay: float = 1e-4
    threshold_quantile: float = 0.95
    device: str = "cuda"
    cpu_threads: int = 2
    mixed_precision: bool = True

    def __post_init__(self):
        if not 0 <= self.warmup_epochs < self.epochs or self.patience < 1:
            raise ValueError("Invalid training schedule")
        if min(self.batch_size, self.evaluation_batch_size, self.cpu_threads) < 1:
            raise ValueError("Invalid batch size or CPU threads")
        if min(self.head_warmup_lr, self.head_finetune_lr, self.backbone_lr) <= 0:
            raise ValueError("Learning rates must be positive")
        if self.weight_decay < 0 or self.min_delta < 0 or not 0 < self.threshold_quantile < 1:
            raise ValueError("Invalid optimizer/stopping/threshold settings")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Initialization seeds must be distinct")
        if not self.machine_types or len(set(self.machine_types)) != len(self.machine_types):
            raise ValueError("Machine types must be distinct")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("Select CPU or CUDA explicitly")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configure_runtime(config):
    torch.set_num_threads(config.cpu_threads)
    device = choose_device(config.device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    if config.mixed_precision and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Configured bfloat16 training is unsupported on this GPU")
    return device


def epoch_pass(model, inputs, targets, config, device, optimizer=None, warmup=False):
    training = optimizer is not None
    if training:
        set_training_stage(model, warmup)
    else:
        model.eval()
    count, loss_sum, correct = 0, 0.0, 0
    size = config.batch_size if training else config.evaluation_batch_size
    with torch.set_grad_enabled(training):
        for indices in batches(len(inputs), size, shuffle=training):
            x = inputs[indices].to(device=device, memory_format=torch.channels_last)
            y = targets[indices].to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            # Validation and final scoring always use float32 forward passes.
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=training and config.mixed_precision and device.type == "cuda"):
                logits = model(x)
                loss = nn.functional.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite cross-entropy")
            if training:
                loss.backward()
                norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                if not torch.isfinite(norm):
                    raise RuntimeError("Nonfinite gradients")
                optimizer.step()
            count += len(indices)
            loss_sum += float(loss.detach()) * len(indices)
            correct += int((logits.argmax(1) == y).sum())
    return loss_sum / count, correct / count


def atomic_save(value, path):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_seed(config, machine, seed, store, device, inputs, model_factory=None,
               model_name="mobilenet_v3_small_imagenet", deadline=None):
    output = Path(config.output_dir) / machine / f"seed_{seed}"
    output.mkdir(parents=True, exist_ok=True)
    identity = json.loads(json.dumps({**asdict(config), "machine_type": machine,
        "initialization_seed": seed, "feature_manifest_sha256": store.manifest_sha256,
        "normalizer_sha256": sha256(store.root / "normalizer.npz"),
        "pretrained_sha256": weights_sha256(config.pretrained_path) if model_factory is None else None}))
    if (output / "config.json").exists():
        if json.loads((output / "config.json").read_text()) != identity:
            raise ValueError(f"Run configuration/provenance changed: {output}")
        status = json.loads((output / "status.json").read_text()) if (output / "status.json").exists() else {}
        if status.get("state") == "complete":
            if sha256(output / "model.pt") != status["checkpoint_sha256"]:
                raise ValueError("Completed checkpoint changed")
            return json.loads((output / "metrics.json").read_text())
    write_json(output / "config.json", identity)
    write_json(output / "status.json", {"state": "starting", "machine": machine, "seed": seed})
    seed_everything(seed)
    model = (build_mobilenet(len(store.summary["machine_ids"]), config.pretrained_path) if model_factory is None
             else model_factory(len(store.summary["machine_ids"]))).to(device=device, memory_format=torch.channels_last)
    train_x, train_y, val_x, val_y = inputs
    train_rows, test_rows = store.split("train"), store.split("test")
    write_json(output / "provenance.json", {
        "feature_manifest_sha256": store.manifest_sha256, "pretrained_sha256": identity["pretrained_sha256"],
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "device": str(device), "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
        "train_clips": len(train_x), "validation_clips": len(val_x), "test_clips": len(test_rows),
        "train_precision": "bfloat16 autocast" if config.mixed_precision and device.type == "cuda" else "float32",
        "evaluation_precision": "float32 forward; float64 stable negative-log probability",
        "determinism": "Seeded, deterministic algorithms requested in warning mode; no bitwise guarantee",
    })
    history, best_loss, best_epoch, best_state = [], float("inf"), 0, None
    stale, stopping_best, previous_seconds = 0, float("inf"), 0.0
    first_epoch, optimizer, optimizer_warmup = 1, None, None
    if (output / "last.pt").exists():
        saved = torch.load(output / "last.pt", map_location=device, weights_only=False)
        if saved["run_identity"] != identity:
            raise ValueError("Resume checkpoint belongs to a different run")
        model.load_state_dict(saved["model_state"])
        history, best_loss, best_epoch = saved["history"], saved["best_loss"], saved["best_epoch"]
        best_state = {k: v.cpu() for k, v in saved["best_state"].items()}
        stale, stopping_best, previous_seconds = saved["stale"], saved["stopping_best"], saved["elapsed_seconds"]
        first_epoch = saved["epoch"] + 1
        optimizer_warmup = saved["warmup"]
        optimizer = make_optimizer(model, config, optimizer_warmup)
        optimizer.load_state_dict(saved["optimizer_state"])
        torch.set_rng_state(saved["torch_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([state.cpu() for state in saved["cuda_rng"]])
        (output / "history.jsonl").write_text("".join(json.dumps(row) + "\n" for row in history))
        print(f"RESUME {machine} seed={seed} after epoch {saved['epoch']}", flush=True)
    start = time.monotonic()
    try:
        for epoch in range(first_epoch, config.epochs + 1):
            if deadline is not None and time.time() >= deadline:
                raise TimeoutError("Training deadline reached; epoch checkpoint retained")
            if stale >= config.patience:
                break
            warmup = epoch <= config.warmup_epochs
            if optimizer is None or optimizer_warmup != warmup:
                optimizer = make_optimizer(model, config, warmup)
                optimizer_warmup = warmup
                stale, stopping_best = 0, float("inf")
            before = time.monotonic()
            train_loss, train_accuracy = epoch_pass(model, train_x, train_y, config, device, optimizer, warmup)
            val_loss, val_accuracy = epoch_pass(model, val_x, val_y, config, device)
            history.append({"epoch": epoch, "stage": "head_warmup" if warmup else "finetune",
                "train_loss": train_loss, "validation_loss": val_loss,
                "train_id_accuracy": train_accuracy, "validation_id_accuracy": val_accuracy,
                "seconds": round(time.monotonic() - before, 3)})
            if val_loss < best_loss:
                best_loss, best_epoch = val_loss, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if not warmup:
                if val_loss < stopping_best - config.min_delta:
                    stopping_best, stale = val_loss, 0
                else:
                    stale += 1
            atomic_save({"run_identity": identity, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "history": history, "best_state": best_state, "best_loss": best_loss, "best_epoch": best_epoch,
                "epoch": epoch, "warmup": warmup, "stale": stale, "stopping_best": stopping_best,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                "elapsed_seconds": previous_seconds + time.monotonic() - start}, output / "last.pt")
            (output / "history.jsonl").write_text("".join(json.dumps(row) + "\n" for row in history))
            write_json(output / "status.json", {"state": "training", "machine": machine, "seed": seed,
                "epoch": epoch, "epochs": config.epochs, "stage": history[-1]["stage"],
                "validation_loss": val_loss, "validation_id_accuracy": val_accuracy,
                "best_epoch": best_epoch, "stale_epochs": stale})
            print(f"{machine} seed={seed} epoch={epoch}/{config.epochs} {history[-1]['stage']} "
                  f"train={train_loss:.4f} val={val_loss:.4f} val_ID_acc={val_accuracy:.4f} "
                  f"{history[-1]['seconds']:.1f}s", flush=True)
        model.load_state_dict(best_state)
        atomic_save({"model_state": best_state, "epoch": best_epoch, "config": identity,
                     "normalizer_mean": torch.from_numpy(store.mean), "normalizer_std": torch.from_numpy(store.std),
                     "machine_ids": store.summary["machine_ids"], "feature_config": store.config}, output / "model.pt")
        write_json(output / "status.json", {"state": "evaluating", "machine": machine, "seed": seed,
                                            "selected_epoch": best_epoch})
        train_scores = score_recordings(model, store, train_rows, device, "mobilenet_v3_small", config.evaluation_batch_size)
        test_scores = score_recordings(model, store, test_rows, device, "mobilenet_v3_small", config.evaluation_batch_size)
        pooled = evaluate_scores(test_rows, test_scores, train_rows, train_scores, output, config.threshold_quantile)
        cells = []
        for snr in sorted({r["snr_db"] for r in test_rows}):
            for mid in sorted(store.summary["machine_ids"]):
                mask = np.array([r["snr_db"] == snr and r["machine_id"] == mid for r in test_rows])
                labels = [r["label"] for r in test_rows if r["snr_db"] == snr and r["machine_id"] == mid]
                cells.append({"machine_type": machine, "seed": seed, "snr_db": snr, "machine_id": mid,
                    **detection_metrics(labels, test_scores[mask], pooled["per_id"][mid]["threshold"])})
        selected = next(r for r in history if r["epoch"] == best_epoch)
        result = {"model": model_name, "machine_type": machine, "seed": seed,
            "selected_epoch": best_epoch, "trained_epochs": len(history),
            "selected_validation_id_accuracy": selected["validation_id_accuracy"],
            "selected_validation_loss": selected["validation_loss"],
            "macro_auc": float(np.mean([r["auc"] for r in cells])),
            "macro_pauc": float(np.mean([r["pauc"] for r in cells])), "cells": cells,
            "pooled_snr_per_id_metrics": pooled, "threshold_quantile": config.threshold_quantile,
            "elapsed_seconds": round(previous_seconds + time.monotonic() - start, 2),
            "feature_manifest_sha256": store.manifest_sha256,
            "score": "float64 softplus(logsumexp(other_logits - known_logit)); -log P(known ID)",
            "aggregation": "Equal mean over 4 ID x 3 SNR cells; standardized pAUC at max_fpr=0.1"}
        write_json(output / "metrics.json", result)
        plot_history(history, output)
        write_json(output / "status.json", {"state": "complete", "machine": machine, "seed": seed,
            "selected_epoch": best_epoch, "checkpoint_sha256": sha256(output / "model.pt")})
        print(f"COMPLETE {machine} seed={seed}: AUC={result['macro_auc']:.4f}, pAUC={result['macro_pauc']:.4f}", flush=True)
        return result
    except BaseException as error:
        write_json(output / "status.json", {"state": "failed", "error": repr(error)})
        raise


def run(config):
    device = configure_runtime(config)
    master = FeatureStore(config.feature_root)
    if set(config.machine_types) != {r["machine_type"] for r in master.records}:
        raise ValueError("This experiment must cover every machine type in the full dataset")
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config_dict = json.loads(json.dumps(asdict(config)))
    files = [*sorted(Path("src/mimii").glob("*.py")), Path("scripts/train_mobilenet.py"), Path("pyproject.toml")]
    source_hashes = {str(p): sha256(p) for p in files}
    protocol = {"config": config_dict, "source_sha256": source_hashes,
        "pretrained_url": WEIGHTS_URL, "pretrained_sha256": weights_sha256(config.pretrained_path),
        "feature_manifest_sha256": master.manifest_sha256,
        "normalizer_sha256": {m: sha256(Path(config.feature_root) / "by_machine" / m / "normalizer.npz")
                              for m in config.machine_types}}
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        previous = json.loads(protocol_path.read_text())
        if any(previous[k] != v for k, v in protocol.items()):
            raise ValueError("Protocol, source code, pretrained weights, or features changed; use a new output directory")
    else:
        protocol.update({"created_utc": datetime.now(timezone.utc).isoformat(),
            "environment": {name: version(name) for name in ("torch", "torchvision", "numpy", "scikit-learn")},
            "training": "Four separate four-ID classifiers; healthy audio only; pooled -6/0/+6 dB",
            "selection": "Lowest healthy-validation cross-entropy; no fault-label tuning",
            "threshold": "95th percentile of training-normal scores per ID, fixed across test SNRs",
            "development_limitation": "Earlier fan pilot test results have been inspected; this is not a fresh blind benchmark",
            "adaptation": "Sum RGB first-convolution weights; retain 128x313 shape; training-normal per-mel normalization",
            "statistics": "Three initialization seeds on a fixed source-group split; population SD, not a confidence interval"})
        write_json(protocol_path, protocol)
        for path in files:
            destination = output / "source_snapshot" / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
        write_json(output / "source_snapshot" / "experiment_config.json", config_dict)
    results = []
    write_json(output / "status.json", {"state": "training", "planned_runs": len(config.machine_types) * len(config.seeds)})
    for machine in config.machine_types:
        store = FeatureStore(Path(config.feature_root) / "by_machine" / machine, "train_per_mel")
        train, validation = store.split("train"), store.split("validation")
        if any(r["label"] != 0 for r in train + validation):
            raise ValueError("Abnormal audio entered training/validation")
        print(f"Loading {machine}: {len(train)} train normals, {len(validation)} validation normals", flush=True)
        inputs = (torch.from_numpy(store.clip_array(train)), torch.tensor([r["id_class"] for r in train]),
                  torch.from_numpy(store.clip_array(validation)), torch.tensor([r["id_class"] for r in validation]))
        machine_output = output / machine
        machine_output.mkdir(exist_ok=True)
        for name in ("config.json", "summary.json", "normalizer.npz", "normalizer_provenance.json"):
            shutil.copy2(store.root / name, machine_output / f"feature_{name}")
        for seed in config.seeds:
            results.append(train_seed(config, machine, seed, store, device, inputs))
            write_json(output / "partial_results.json", results)
            write_json(output / "status.json", {"state": "training", "completed_runs": len(results),
                "planned_runs": len(config.machine_types) * len(config.seeds)})
        del inputs
    from .report_transfer import report
    summary = report(output, results, config)
    write_json(output / "status.json", {"state": "complete", "completed_runs": len(results)})
    return summary


def benchmark(config):
    device = configure_runtime(config)
    seed_everything(42)
    model = build_mobilenet(4, config.pretrained_path).to(device=device, memory_format=torch.channels_last)
    optimizer = make_optimizer(model, config, False)
    count = config.batch_size * 10
    x, y = torch.randn(count, 1, 128, 313), torch.arange(count) % 4
    epoch_pass(model, x[:config.batch_size * 2], y[:config.batch_size * 2], config, device, optimizer)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    epoch_pass(model, x, y, config, device, optimizer)
    if device.type == "cuda":
        torch.cuda.synchronize()
    result = {"batch_size": config.batch_size, "seconds_per_training_batch": (time.monotonic() - start) / 10,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else None,
        "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
        "parameters": sum(p.numel() for p in model.parameters())}
    write_json("runs/mobilenet_benchmark.json", result)
    print(json.dumps(result, indent=2))
