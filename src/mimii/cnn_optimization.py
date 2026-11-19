"""Bounded healthy-validation CNN optimization with test evaluation after selection."""

from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from .ae_experiment import write_json
from .evaluation import detection_metrics, evaluate_scores
from .finetune import FinetuneConfig, atomic_save, configure_runtime, epoch_pass, seed_everything
from .mobilenet import make_optimizer
from .train import known_id_anomaly_scores, plot_history
from .training_data import FeatureStore
from .transfer_models import build_transfer, pretrained_hash


class BudgetExpired(RuntimeError):
    pass


def trial_name(index):
    return f"lr_{index}"


def make_config(study, architecture, rate):
    return FinetuneConfig(feature_root=study["feature_root"], output_dir=study["output_dir"],
        pretrained_path="", seeds=(study["seed"],), epochs=study["final_epochs"],
        warmup_epochs=study["warmup_epochs"], patience=100, min_delta=0,
        batch_size=study["batch_sizes"][architecture], evaluation_batch_size=study["batch_sizes"][architecture],
        backbone_lr=rate, head_finetune_lr=rate * study["head_lr_multiplier"],
        head_warmup_lr=0.001, weight_decay=study["weight_decay"], threshold_quantile=study["threshold_quantile"],
        device=study.get("device", "cuda"))


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def fit(study, architecture, index, machine, store, inputs, target_epoch, deadline):
    cfg = make_config(study, architecture, study["backbone_learning_rates"][index])
    device = torch.device(cfg.device)
    output = Path(study["output_dir"]) / architecture / trial_name(index) / machine
    output.mkdir(parents=True, exist_ok=True)
    path = output / "last.pt"
    seed_everything(study["seed"])
    model = build_transfer(architecture, len(store.summary["machine_ids"])).to(device=device, memory_format=torch.channels_last)
    history, best_loss, best_epoch, best_state = [], float("inf"), 0, None
    optimizer, previous_warmup = None, None
    if path.exists():
        saved = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(saved["model_state"])
        history, best_loss, best_epoch = saved["history"], saved["best_loss"], saved["best_epoch"]
        best_state = {k: v.cpu() for k, v in saved["best_state"].items()}
        previous_warmup = saved["warmup"]
        optimizer = make_optimizer(model, cfg, previous_warmup)
        optimizer.load_state_dict(saved["optimizer_state"])
        torch.set_rng_state(saved["torch_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([s.cpu() for s in saved["cuda_rng"]])
        del saved
    train_x, train_y, val_x, val_y = inputs
    for epoch in range(len(history) + 1, target_epoch + 1):
        if time.time() >= deadline:
            raise BudgetExpired("Training reserve reached; saved checkpoints preserved")
        warmup = epoch <= cfg.warmup_epochs
        if optimizer is None or warmup != previous_warmup:
            optimizer, previous_warmup = make_optimizer(model, cfg, warmup), warmup
        started = time.monotonic()
        train_loss, train_accuracy = epoch_pass(model, train_x, train_y, cfg, device, optimizer, warmup)
        val_loss, val_accuracy = epoch_pass(model, val_x, val_y, cfg, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss,
                        "train_id_accuracy": train_accuracy, "validation_id_accuracy": val_accuracy,
                        "seconds": time.monotonic() - started, "stage": "head_warmup" if warmup else "finetune"})
        if val_loss < best_loss:
            best_loss, best_epoch, best_state = val_loss, epoch, cpu_state(model)
        atomic_save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                     "history": history, "best_loss": best_loss, "best_epoch": best_epoch, "best_state": best_state,
                     "warmup": warmup, "torch_rng": torch.get_rng_state(),
                     "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}, path)
        write_json(output / "history.json", history)
        write_json(Path(study["output_dir"]) / "status.json", {"state": "training", "architecture": architecture,
                   "trial": index, "machine": machine, "epoch": epoch, "target_epoch": target_epoch,
                   "validation_loss": val_loss, "validation_id_accuracy": val_accuracy,
                   "updated_utc": datetime.now(timezone.utc).isoformat()})
        print(f"{architecture} trial={index} {machine} epoch={epoch}/{target_epoch} train={train_loss:.4f} val={val_loss:.4f} IDacc={val_accuracy:.4f} {history[-1]['seconds']:.1f}s", flush=True)
    selected = next(r for r in history if r["epoch"] == best_epoch)
    atomic_save({"model_state": best_state, "architecture": architecture, "seed": study["seed"],
                 "epoch": best_epoch, "config": asdict(cfg), "machine_ids": store.summary["machine_ids"],
                 "normalizer_mean": torch.from_numpy(store.mean), "normalizer_std": torch.from_numpy(store.std),
                 "feature_config": store.config, "manifest_sha256": store.manifest_sha256}, output / "model.pt")
    result = {"architecture": architecture, "trial": index, "machine_type": machine, "seed": study["seed"],
              "selected_epoch": best_epoch, "trained_epochs": len(history), "validation_loss": best_loss,
              "validation_id_accuracy": selected["validation_id_accuracy"], "backbone_lr": cfg.backbone_lr,
              "training_seconds": sum(r["seconds"] for r in history), "parameters": sum(p.numel() for p in model.parameters())}
    write_json(output / "fit.json", result)
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return result


def choose_trials(screening, architectures, machines, trial_count):
    selections = {}
    for architecture in architectures:
        candidates = []
        for index in range(trial_count):
            rows = [r for r in screening if r["architecture"] == architecture and r["trial"] == index]
            if len(rows) != len(machines) or {r["machine_type"] for r in rows} != set(machines):
                raise ValueError("Learning-rate selection requires every machine type")
            candidates.append({"trial": index, "macro_validation_loss": float(np.mean([r["validation_loss"] for r in rows]))})
        selections[architecture] = {"selected_trial": min(candidates, key=lambda x: (x["macro_validation_loss"], x["trial"]))["trial"],
                                   "candidates": candidates, "criterion": "mean best healthy-validation cross-entropy over machine types"}
    return selections


def score_inputs(model, inputs, labels, size):
    scores = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(inputs), size):
            device = next(model.parameters()).device
            x = inputs[start:start + size].to(device=device, memory_format=torch.channels_last)
            targets = labels[start:start + size].to(device)
            scores.extend(known_id_anomaly_scores(model(x), targets).cpu().tolist())
    return np.asarray(scores)


def evaluate(study, architecture, index, machine, store, inputs):
    output = Path(study["output_dir"]) / architecture / trial_name(index) / machine
    model_path = output / "model.pt"
    checkpoint_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    metrics_path = output / "metrics.json"
    if metrics_path.exists():
        result = json.loads(metrics_path.read_text())
        if result["checkpoint_sha256"] != checkpoint_hash:
            raise ValueError("Evaluated checkpoint changed")
        return result
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model = build_transfer(architecture, len(store.summary["machine_ids"]), pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device=study.get("device", "cuda"), memory_format=torch.channels_last)
    size = study["batch_sizes"][architecture]
    train, test = store.split("train"), store.split("test")
    training_scores = score_inputs(model, inputs[0], inputs[1], size)
    # Scoring test data begins only after all architecture/rate/epoch choices are frozen.
    test_scores = []
    for start in range(0, len(test), size):
        rows = test[start:start + size]
        test_scores.extend(score_inputs(model, torch.from_numpy(store.clip_array(rows)),
                                       torch.tensor([r["id_class"] for r in rows]), size))
    test_scores = np.asarray(test_scores)
    pooled = evaluate_scores(test, test_scores, train, training_scores, output, study["threshold_quantile"])
    cells = []
    for snr in (-6, 0, 6):
        for mid in sorted(store.summary["machine_ids"]):
            mask = np.array([r["snr_db"] == snr and r["machine_id"] == mid for r in test])
            labels = [r["label"] for r, selected in zip(test, mask) if selected]
            cells.append({"machine_type": machine, "machine_id": mid, "snr_db": snr, "seed": study["seed"],
                          **detection_metrics(labels, test_scores[mask], pooled["per_id"][mid]["threshold"])})
    result = {**json.loads((output / "fit.json").read_text()), "cells": cells,
              "macro_auc": float(np.mean([c["auc"] for c in cells])), "macro_pauc": float(np.mean([c["pauc"] for c in cells])),
              "checkpoint_sha256": checkpoint_hash}
    write_json(metrics_path, result)
    plot_history(json.loads((output / "history.json").read_text()), output)
    print(f"RESULT {architecture} {machine}: AUC={result['macro_auc']:.5f} pAUC={result['macro_pauc']:.5f}", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run(study):
    root = Path(study["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    configure_runtime(make_config(study, study["architectures"][0], study["backbone_learning_rates"][0]))
    sources = [*sorted(Path("src/mimii").glob("*.py")), Path("scripts/train_cnn_optimization.py")]
    master = FeatureStore(study["feature_root"])
    if {r["machine_type"] for r in master.records} != set(study["machine_types"]):
        raise ValueError("Study must cover the full dataset")
    identity = {"study": study, "master_manifest_sha256": master.manifest_sha256,
                "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                "pretrained_sha256": {a: pretrained_hash(a) for a in study["architectures"]},
                "normalizer_sha256": {m: hashlib.sha256((Path(study["feature_root"]) / "by_machine" / m / "normalizer.npz").read_bytes()).hexdigest() for m in study["machine_types"]},
                "versions": {name: version(name) for name in ("torch", "torchvision", "numpy", "scikit-learn")}}
    protocol_path = root / "protocol.json"
    if protocol_path.exists():
        protocol = json.loads(protocol_path.read_text())
        if protocol["identity"] != identity:
            raise ValueError("Frozen study code/data/settings changed")
    else:
        protocol = {"identity": identity, "started_unix": time.time(), "deadline_unix": time.time() + study["wall_seconds"],
                    "selection": "healthy validation only; test remains unused until choices are frozen",
                    "statistical_limit": "one initialization seed; no variability estimate for new CNNs"}
        write_json(protocol_path, protocol)
        for source in sources:
            destination = root / "source_snapshot" / source
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    deadline = protocol["deadline_unix"]
    train_deadline = deadline - study["evaluation_reserve_seconds"]
    write_json(root / "status.json", {"state": "loading_features", "deadline_utc": datetime.fromtimestamp(deadline, timezone.utc).isoformat()})
    stores, cache = {}, {}
    for machine in study["machine_types"]:
        print(f"Loading all normal train/validation features: {machine}", flush=True)
        store = FeatureStore(Path(study["feature_root"]) / "by_machine" / machine, "train_per_mel")
        train, validation = store.split("train"), store.split("validation")
        if any(r["label"] != 0 for r in train + validation):
            raise ValueError("Abnormal recording in fitting/selection data")
        stores[machine] = store
        cache[machine] = (torch.from_numpy(store.clip_array(train)), torch.tensor([r["id_class"] for r in train]),
                          torch.from_numpy(store.clip_array(validation)), torch.tensor([r["id_class"] for r in validation]))
    screening = []
    selection_path = root / "selection.json"
    if not selection_path.exists():
        for architecture in study["architectures"]:
            for index in range(len(study["backbone_learning_rates"])):
                for machine in study["machine_types"]:
                    screening.append(fit(study, architecture, index, machine, stores[machine], cache[machine], study["screening_epochs"], train_deadline))
                    write_json(root / "screening.json", screening)
        selection = choose_trials(screening, study["architectures"], study["machine_types"], len(study["backbone_learning_rates"]))
        write_json(selection_path, selection)
    else:
        selection = json.loads(selection_path.read_text())
        screening = json.loads((root / "screening.json").read_text())
    # Choose a common final training horizon from observed throughput, not test scores.
    plan_path = root / "final_plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
    else:
        seconds_per_round = 0
        for architecture in study["architectures"]:
            index = selection[architecture]["selected_trial"]
            for machine in study["machine_types"]:
                history = json.loads((root / architecture / trial_name(index) / machine / "history.json").read_text())
                seconds_per_round += max(r["seconds"] for r in history)
        remaining = max(0, train_deadline - time.time() - 120)
        extra = min(study["final_epochs"] - study["screening_epochs"], int(remaining / (seconds_per_round * 1.25)))
        plan = {"target_epochs": study["screening_epochs"] + extra, "selection": selection,
                "reason": "common horizon constrained by measured throughput and reserved evaluation time"}
        write_json(plan_path, plan)
    for architecture in study["architectures"]:
        for machine in study["machine_types"]:
            fit(study, architecture, selection[architecture]["selected_trial"], machine, stores[machine], cache[machine], plan["target_epochs"], train_deadline)
    write_json(root / "status.json", {"state": "evaluating", "test_selection": "all settings frozen before evaluation"})
    results = []
    for architecture in study["architectures"]:
        for machine in study["machine_types"]:
            if time.time() >= deadline:
                raise BudgetExpired("Wall-clock budget reached during evaluation; completed results preserved")
            results.append(evaluate(study, architecture, selection[architecture]["selected_trial"], machine, stores[machine], cache[machine]))
            write_json(root / "partial_results.json", results)
    from .report_cnn_optimization import report
    report(root, study, results, stores)
    write_json(root / "status.json", {"state": "complete", "completed_fits": len(results), "seed": study["seed"],
               "elapsed_seconds": time.time() - protocol["started_unix"]})
