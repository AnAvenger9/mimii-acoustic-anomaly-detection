"""Train and evaluate normal-only AE/CNN models with reproducible manifests."""

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn

from .evaluation import evaluate_scores
from .features import context_vectors
from .models import CNN_MODELS, DenseAutoencoder
from .preprocess import write_json
from .training_data import FeatureStore


@dataclass(frozen=True)
class TrainingConfig:
    feature_dir: str = "features/fan_0db_128mel_seed42"
    output_dir: str = "runs/fan_0db_ae128"
    model: str = "autoencoder"
    seeds: tuple = (42, 43, 44)
    epochs: int = 100
    batch_size: int = 512
    learning_rate: float = 0.001
    adam_epsilon: float = 1e-7
    latent_dimensions: int = 8
    normalization: str = "none"
    checkpoint_selection: str = "last"
    threshold_quantile: float = 0.95
    device: str = "auto"
    cpu_threads: int = 2

    def __post_init__(self):
        if self.model not in {"autoencoder", *CNN_MODELS}:
            raise ValueError("Unknown model")
        if self.epochs < 1 or self.batch_size < 2 or self.cpu_threads < 1:
            raise ValueError("Invalid epochs, batch size or CPU threads")
        if self.learning_rate <= 0 or self.adam_epsilon <= 0 or self.latent_dimensions < 1:
            raise ValueError("Invalid optimizer or latent dimensions")
        if self.checkpoint_selection not in {"last", "best_validation"}:
            raise ValueError("Unknown checkpoint selection")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Supply distinct initialization seeds")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu or cuda")
        if not 0 < self.threshold_quantile < 1:
            raise ValueError("Invalid threshold quantile")


def choose_device(requested):
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but inaccessible in this process; run from a GPU-enabled terminal")
    return torch.device("cuda" if requested != "cpu" and torch.cuda.is_available() else "cpu")


def batches(count, batch_size, shuffle=False):
    indices = torch.randperm(count) if shuffle else torch.arange(count)
    parts = list(indices.split(batch_size))
    if len(parts) > 1 and len(parts[-1]) == 1:
        parts[-2] = torch.cat((parts[-2], parts[-1]))
        parts.pop()
    return parts


def epoch_pass(model, inputs, targets, batch_size, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total, correct, seen = 0.0, 0, 0
    criterion = nn.MSELoss() if targets is None else nn.CrossEntropyLoss()
    with torch.set_grad_enabled(training):
        for indices in batches(len(inputs), batch_size, shuffle=training):
            x = inputs[indices].to(device)
            target = x if targets is None else targets[indices].to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            loss = criterion(prediction, target)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            if training:
                loss.backward()
                optimizer.step()
            n = len(indices)
            total += float(loss.detach()) * n
            seen += n
            if targets is not None:
                correct += int((prediction.argmax(dim=1) == target).sum())
    return total / seen, (correct / seen if targets is not None else None)


def known_id_anomaly_scores(logits, targets):
    """Stable -log p(known ID), including very confident correct predictions.

    log_softmax can round log(1 + tiny_odds) to zero. softplus(logsumexp
    competing logit differences)) preserves those small positive scores.
    """
    values = logits.to(torch.float64)
    known = values.gather(1, targets[:, None])
    differences = (values - known).scatter(1, targets[:, None], -torch.inf)
    return torch.nn.functional.softplus(torch.logsumexp(differences, dim=1))


def score_recordings(model, store, rows, device, model_type, clip_batch_size=16):
    scores = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), clip_batch_size):
            part = rows[start:start + clip_batch_size]
            if model_type == "autoencoder":
                x = np.stack([context_vectors(store.load(r), store.context_frames) for r in part])
                clips, windows, dimensions = x.shape
                tensor = torch.from_numpy(x.reshape(-1, dimensions)).to(device)
                error = (model(tensor) - tensor).square().mean(dim=1)
                scores.extend(error.reshape(clips, windows).mean(dim=1).cpu().tolist())
            else:
                tensor = torch.from_numpy(store.clip_array(part)).to(device)
                logits = model(tensor)
                targets = torch.tensor([r["id_class"] for r in part], device=device)
                scores.extend(known_id_anomaly_scores(logits, targets).cpu().tolist())
    return np.asarray(scores)


def train_seed(config, seed, store, device):
    output = Path(config.output_dir) / f"seed_{seed}"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Run already exists: {output}; choose another output_dir")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "status.json", {"state": "running", "seed": seed})
    write_json(output / "config.json", {**asdict(config), "initialization_seed": seed,
                                       "feature_manifest_sha256": store.manifest_sha256})
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    start = time.monotonic()
    try:
        train_rows, validation_rows = store.split("train"), store.split("validation")
        if any(r["label"] for r in train_rows + validation_rows):
            raise ValueError("Training and validation must contain normal audio only")
        train_targets = validation_targets = None
        if config.model == "autoencoder":
            model = DenseAutoencoder(store.shape[0] * store.context_frames, config.latent_dimensions)
            train_inputs = torch.from_numpy(store.vector_array(train_rows))
            validation_inputs = torch.from_numpy(store.vector_array(validation_rows))
        else:
            model = CNN_MODELS[config.model](len(store.summary["machine_ids"]))
            train_inputs = torch.from_numpy(store.clip_array(train_rows))
            validation_inputs = torch.from_numpy(store.clip_array(validation_rows))
            train_targets = torch.tensor([r["id_class"] for r in train_rows])
            validation_targets = torch.tensor([r["id_class"] for r in validation_rows])
        model.to(device)
        write_json(output / "provenance.json", {
            "device": str(device), "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "feature_manifest_sha256": store.manifest_sha256,
            "feature_config": store.config, "parameter_count": sum(p.numel() for p in model.parameters()),
            "train_clips": len(train_rows), "validation_clips": len(validation_rows),
            "train_examples": len(train_inputs), "normalization": config.normalization,
            "determinism": "seeded; deterministic algorithms requested with warn_only=True",
            "protocol": ("Official DCASE development test partition; PyTorch implementation and recording-level validation"
                         if all("official_partition" in r for r in store.records)
                         else "Custom MIMII recording split; see feature_config for SNR scope; not an official DCASE replication"),
        })
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, eps=config.adam_epsilon)
        best_loss, selected_epoch = float("inf"), 0
        history = []
        for epoch in range(1, config.epochs + 1):
            epoch_start = time.monotonic()
            train_loss, train_accuracy = epoch_pass(model, train_inputs, train_targets, config.batch_size, device, optimizer)
            validation_loss, validation_accuracy = epoch_pass(model, validation_inputs, validation_targets, config.batch_size * 2, device)
            record = {"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss,
                      "train_id_accuracy": train_accuracy, "validation_id_accuracy": validation_accuracy,
                      "seconds": round(time.monotonic() - epoch_start, 3)}
            history.append(record)
            with (output / "history.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            improved = validation_loss < best_loss
            if improved:
                best_loss = validation_loss
            if config.checkpoint_selection == "last" or improved:
                selected_epoch = epoch
                torch.save({"model_state": model.state_dict(), "epoch": epoch, "seed": seed,
                            "config": asdict(config), "feature_manifest_sha256": store.manifest_sha256},
                           output / "model.pt")
            write_json(output / "status.json", {"state": "training", "seed": seed, "epoch": epoch,
                                                "epochs": config.epochs, "last_validation_loss": validation_loss})
            print(f"{config.model} seed={seed} epoch={epoch}/{config.epochs} "
                  f"train={train_loss:.5f} val={validation_loss:.5f} "
                  f"val_id_acc={validation_accuracy} {record['seconds']:.1f}s", flush=True)
        checkpoint = torch.load(output / "model.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        del train_inputs, validation_inputs
        test_rows = store.split("test")
        train_scores = score_recordings(model, store, train_rows, device, config.model)
        test_scores = score_recordings(model, store, test_rows, device, config.model)
        metrics = evaluate_scores(test_rows, test_scores, train_rows, train_scores, output, config.threshold_quantile)
        metrics.update({"seed": seed, "model": config.model, "selected_epoch": selected_epoch,
                        "elapsed_seconds": round(time.monotonic() - start, 2),
                        "evaluation_score_implementation": (
                            "mean squared reconstruction error" if config.model == "autoencoder" else
                            "float64 softplus(logsumexp(competing_logits - known_logit)); mathematically -log p(known ID)")})
        write_json(output / "metrics.json", metrics)
        plot_history(history, output)
        write_json(output / "status.json", {"state": "complete", "seed": seed, "selected_epoch": selected_epoch})
        print(f"COMPLETE {config.model} seed={seed}: AUC={metrics['macro_auc']:.4f} pAUC={metrics['macro_pauc']:.4f}", flush=True)
        return metrics
    except BaseException as error:
        write_json(output / "status.json", {"state": "failed", "seed": seed, "error": repr(error)})
        raise


def plot_history(history, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    ax.plot([r["epoch"] for r in history], [r["train_loss"] for r in history], label="Training normals")
    ax.plot([r["epoch"] for r in history], [r["validation_loss"] for r in history], label="Validation normals")
    ax.set(xlabel="Epoch", ylabel="Loss", title="Learning curves")
    ax.legend()
    fig.savefig(output / "learning_curve.png", dpi=170)
    plt.close(fig)


def run(config):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mimii-matplotlib")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(config.cpu_threads)
    device = choose_device(config.device)
    store = FeatureStore(config.feature_dir, config.normalization)
    print(f"Using {device}; fixed feature split {store.manifest_sha256[:12]}", flush=True)
    results = [train_seed(config, seed, store, device) for seed in config.seeds]
    summary = {"model": config.model, "seeds": list(config.seeds), "split_seed": store.config["seed"],
               "feature_manifest_sha256": store.manifest_sha256,
               "macro_auc_mean": float(np.mean([r["macro_auc"] for r in results])),
               "macro_auc_std": float(np.std([r["macro_auc"] for r in results])) if len(results) > 1 else None,
               "macro_pauc_mean": float(np.mean([r["macro_pauc"] for r in results])),
               "macro_pauc_std": float(np.std([r["macro_pauc"] for r in results])) if len(results) > 1 else None,
               "std_convention": "population standard deviation across initialization seeds",
               "runs": results}
    write_json(Path(config.output_dir) / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_ae.json")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    values = json.loads(Path(args.config).read_text())
    for key in ("device", "seeds", "epochs", "output_dir"):
        value = getattr(args, key)
        if value is not None:
            values[key] = value
    run(TrainingConfig(**values))


if __name__ == "__main__":
    main()
