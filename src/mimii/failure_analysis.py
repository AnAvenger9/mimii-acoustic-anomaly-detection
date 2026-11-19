"""Saved-score failure analysis and a paired, fixed-sample microphone stress test."""

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import fields
import csv
import hashlib
import html
import json
from pathlib import Path

import numpy as np

from .audio import load_audio
from .config import PreprocessConfig
from .evaluation import detection_metrics
from .features import log_mel_spectrogram
from .training_data import FeatureStore


MACHINES = ("fan", "pump", "slider", "valve")
MODELS = ("mobilenet_v3_small", "resnet18", "efficientnet_b0")
FEATURE_ROOT = Path("features/full_dataset_logmel128_seed42/by_machine")
OUTPUT = Path("runs/failure_analysis")


def saved_directory(model, machine):
    if model == "mobilenet_v3_small":
        return Path("runs/mobilenet_v3_small_full_dataset") / machine / "seed_42"
    root = Path("runs/cnn_optimization_12epochs")
    selection = json.loads((root / "selection.json").read_text())
    return root / model / f"lr_{selection[model]['selected_trial']}" / machine


def read_csv(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def enrich_scores(model, machine, store):
    records = {r["relative_path"]: r for r in store.split("test")}
    rows = read_csv(saved_directory(model, machine) / "scores.csv")
    if len(rows) != len(records) or {r["relative_path"] for r in rows} != set(records):
        raise ValueError("Saved predictions do not cover the frozen test manifest")
    return [{**records[r["relative_path"]], "score": float(r["score"]), "threshold": float(r["threshold"]),
             "prediction": int(r["prediction"]), "model": model} for r in rows]


def choose_microphone_sample(rows, count=40):
    """Select source groups by hash, before looking at model scores; pair all SNRs."""
    selected = set()
    for mid in sorted({r["machine_id"] for r in rows}):
        for label in (0, 1):
            candidates = [r for r in rows if r["machine_id"] == mid and r["label"] == label and r["snr_db"] == 0]
            candidates.sort(key=lambda r: hashlib.sha256(("microphone-shift-v1:" + r["source_recording_key"]).encode()).hexdigest())
            if len(candidates) < count:
                raise ValueError("Insufficient test recordings for predefined balanced sample")
            selected.update(r["source_recording_key"] for r in candidates[:count])
    result = sorted((r for r in rows if r["source_recording_key"] in selected), key=lambda r: r["relative_path"])
    grouped = defaultdict(set)
    for r in result:
        if r["split"] != "test":
            raise ValueError("Microphone stress test must use held-out test recordings")
        grouped[r["source_recording_key"]].add(r["snr_db"])
    if any(snrs != {-6, 0, 6} for snrs in grouped.values()):
        raise ValueError("All selected source groups must have three SNR versions")
    return result


def extract_second_channel(item):
    row, config = item
    cfg = PreprocessConfig(**config)
    waveform, metadata = load_audio(Path(row["source_path"]), cfg)
    return log_mel_spectrogram(waveform, cfg), metadata["waveform_sha256"]


def prepare_microphone_sample():
    root = OUTPUT / "microphone_shift"
    root.mkdir(parents=True, exist_ok=True)
    protocol = {"selection": "40 held-out source groups per machine type/ID/condition, sorted by fixed SHA256 key, paired across all three SNRs",
                "count_per_id_condition_snr": 40, "training_channel_zero_based": 0, "evaluation_channel_zero_based": 1,
                "thresholds_and_normalizers": "Original channel-1 training values, unchanged",
                "models": list(MODELS), "seed": 42, "interpretation": "Same-array microphone-channel stress test, not session-disjoint or unseen-SNR evaluation"}
    write_json(root / "protocol.json", protocol)
    for machine in MACHINES:
        store = FeatureStore(FEATURE_ROOT / machine)
        rows = choose_microphone_sample(store.split("test"))
        config = {f.name: store.config[f.name] for f in fields(PreprocessConfig) if f.name in store.config}
        config.update(channel=1, data_root="data", output_dir=str(root), workers=2)
        path = root / f"{machine}_channel2.npy"
        metadata_path = root / f"{machine}_manifest.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            if metadata["manifest_sha256"] != store.manifest_sha256 or metadata["config"] != config or metadata["rows"] != rows:
                raise ValueError("Microphone sample settings changed")
            if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["feature_sha256"]:
                raise ValueError("Microphone feature cache changed")
            continue
        print(f"Extracting microphone channel 2: {machine}, {len(rows)} paired test clips", flush=True)
        array = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(len(rows), *store.shape))
        hashes = []
        with ProcessPoolExecutor(max_workers=2) as executor:
            for index, (features, digest) in enumerate(executor.map(extract_second_channel, ((r, config) for r in rows), chunksize=8)):
                array[index] = features
                hashes.append(digest)
        array.flush()
        del array
        write_json(metadata_path, {"manifest_sha256": store.manifest_sha256, "config": config, "rows": rows,
                                  "waveform_sha256_channel2": hashes, "feature_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})


def analyze_saved_errors():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import soundfile as sf

    OUTPUT.mkdir(parents=True, exist_ok=True)
    media = OUTPUT / "media"
    media.mkdir(exist_ok=True)
    summaries, cases, gallery = [], [], ["<!doctype html><meta charset='utf-8'><title>Acoustic failure review</title><h1>Representative error review</h1><p>Cases are selected by median score/threshold ratio among errors, preferentially at 0 dB. Audio is microphone 1. These cases illustrate errors; they do not identify mechanical fault causes.</p>"]
    for model in MODELS:
        for machine in MACHINES:
            store = FeatureStore(FEATURE_ROOT / machine)
            rows = enrich_scores(model, machine, store)
            cells = []
            for snr in (-6, 0, 6):
                for mid in sorted(store.summary["machine_ids"]):
                    part = [r for r in rows if r["snr_db"] == snr and r["machine_id"] == mid]
                    cells.append({"snr_db": snr, "machine_id": mid, **detection_metrics([r["label"] for r in part], [r["score"] for r in part], part[0]["threshold"])})
            missed = [r for r in rows if r["label"] == 1 and r["prediction"] == 0]
            alarms = [r for r in rows if r["label"] == 0 and r["prediction"] == 1]
            grouped = defaultdict(list)
            for row in rows:
                grouped[row["source_recording_key"]].append(row)
            patterns = {str(label): Counter("".join(str(r["prediction"]) for r in sorted(group, key=lambda r: r["snr_db"]))
                       for group in grouped.values() if group[0]["label"] == label) for label in (0, 1)}
            summary = {"model": model, "machine_type": machine, "false_positives": len(alarms), "false_negatives": len(missed),
                "macro_fpr": float(np.mean([c["false_positive_rate"] for c in cells])),
                "macro_recall": float(np.mean([c["true_positive_rate"] for c in cells])),
                "missed_faults_with_known_id_probability_over_099": int(sum(r["score"] < -np.log(.99) for r in missed)),
                "snr_prediction_patterns_minus6_0_plus6": patterns, "cells": cells}
            summaries.append(summary)
            train_scores = read_csv(saved_directory(model, machine) / "train_scores.csv")
            train_records = {r["relative_path"]: r for r in store.split("train")}
            for error_name, errors in (("false_positive", alarms), ("false_negative", missed)):
                preferred = [r for r in errors if r["snr_db"] == 0] or errors
                if not preferred:
                    continue
                preferred.sort(key=lambda r: (r["score"] / max(r["threshold"], np.finfo(float).tiny), r["relative_path"]))
                row = preferred[len(preferred) // 2]
                reference = [r for r in train_scores if r["machine_id"] == row["machine_id"] and train_records[r["relative_path"]]["snr_db"] == row["snr_db"]]
                reference.sort(key=lambda r: (float(r["score"]), r["relative_path"]))
                normal = train_records[reference[len(reference) // 2]["relative_path"]]
                name = f"{model}_{machine}_{error_name}"
                matrices = [store.load(normal), store.load(row)]
                low, high = np.percentile(np.concatenate([x.ravel() for x in matrices]), [1, 99])
                fig, axes = plt.subplots(1, 3, figsize=(12, 3.3), layout="constrained")
                for ax, matrix, title in zip(axes[:2], matrices, ("Training-normal reference", error_name.replace("_", " ").title())):
                    ax.imshow(matrix, origin="lower", aspect="auto", extent=(0, 10, 0, 128), vmin=low, vmax=high, cmap="magma")
                    ax.set(xlabel="Time (s)", ylabel="Mel band", title=title)
                for matrix, label in zip(matrices, ("Training normal", "Error clip")):
                    axes[2].plot(matrix.mean(axis=1), label=label)
                axes[2].set(xlabel="Mel band", ylabel="Mean log-mel power (dB)", title="Average spectrum")
                axes[2].legend(fontsize=8)
                fig.suptitle(f"{model} · {machine} {row['machine_id']} · {row['snr_db']:+d} dB · score/threshold={row['score']/max(row['threshold'],1e-300):.3g}")
                fig.savefig(media / f"{name}.png", dpi=130)
                plt.close(fig)
                for suffix, recording in (("error", row), ("reference", normal)):
                    samples, rate = sf.read(recording["source_path"], dtype="float32", always_2d=True)
                    sf.write(media / f"{name}_{suffix}.wav", samples[:, 0], rate, subtype="PCM_16")
                cases.append({"model": model, "machine_type": machine, "error_type": error_name, "relative_path": row["relative_path"],
                              "source_path": row["source_path"], "machine_id": row["machine_id"], "snr_db": row["snr_db"],
                              "score": row["score"], "threshold": row["threshold"], "normal_reference": normal["relative_path"], "figure": f"media/{name}.png"})
                gallery.append(f"<h2>{html.escape(name)}</h2><p>{html.escape(row['relative_path'])}</p><img style='max-width:100%' src='media/{name}.png'><p>Error clip:</p><audio controls src='media/{name}_error.wav'></audio><p>Healthy training reference:</p><audio controls src='media/{name}_reference.wav'></audio>")
    write_json(OUTPUT / "summary.json", {"seed": 42, "summaries": summaries,
               "interpretation": "Descriptive analysis of previously inspected test data. Micro and macro error rates differ. No fault-mechanism labels or session labels are inferred."})
    write_json(OUTPUT / "cases.json", cases)
    gallery.append("<p>Audio: original MIMII dataset, Hitachi Ltd., CC BY-SA 4.0; excerpts here are channel-1 copies for local review.</p>")
    (OUTPUT / "gallery.html").write_text("\n".join(gallery))
    lines = ["# Failure analysis of the completed CNNs", "", "All models use seed 42 here for a paired descriptive comparison. Thresholds remain the original training-normal 95th percentiles.", "",
             "| Model | Machine | False alarms | Missed faults | Macro FPR (%) | Macro fault recall (%) |", "|---|---|---:|---:|---:|---:|"]
    for row in summaries:
        lines.append(f"| {row['model']} | {row['machine_type']} | {row['false_positives']} | {row['false_negatives']} | {100*row['macro_fpr']:.2f} | {100*row['macro_recall']:.2f} |")
    lines.extend(["", "Counts pool clips; rates are equal-weight means over ID/SNR cells. SNR versions share source recordings.",
                  "The HTML listening gallery in runs/failure_analysis/gallery.html includes deterministic median-error examples and healthy training references from the same ID/SNR.",
                  "Case selection is illustrative, not a prevalence estimate or evidence of a particular physical fault mechanism."])
    (OUTPUT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    docs = Path("docs/results/failure_analysis")
    docs.mkdir(parents=True, exist_ok=True)
    for name in ("summary.json", "cases.json", "RESULTS.md"):
        (docs / name).write_bytes((OUTPUT / name).read_bytes())
    print("Saved failure counts, paired-SNR error patterns, representative figures and listening gallery", flush=True)


def evaluate_microphone_shift():
    import torch
    from .cnn_optimization import score_inputs
    from .finetune import FinetuneConfig, configure_runtime
    from .mobilenet import build_mobilenet
    from .transfer_models import build_transfer

    configure_runtime(FinetuneConfig())
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required for the microphone shift evaluation")
    root = OUTPUT / "microphone_shift"
    results = []
    for machine in MACHINES:
        metadata = json.loads((root / f"{machine}_manifest.json").read_text())
        rows = metadata["rows"]
        feature_path = root / f"{machine}_channel2.npy"
        if hashlib.sha256(feature_path.read_bytes()).hexdigest() != metadata["feature_sha256"]:
            raise ValueError("Microphone cache checksum changed")
        raw = np.load(feature_path, allow_pickle=False)
        store = FeatureStore(FEATURE_ROOT / machine)
        if store.manifest_sha256 != metadata["manifest_sha256"] or rows != choose_microphone_sample(store.split("test")):
            raise ValueError("Microphone sample or source manifest changed")
        if raw.shape != (len(rows), *store.shape) or not np.isfinite(raw).all():
            raise ValueError("Invalid microphone feature bank")
        for name in MODELS:
            checkpoint_path = saved_directory(name, machine) / "model.pt"
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            model = build_mobilenet(4) if name == "mobilenet_v3_small" else build_transfer(name, pretrained=False)
            model.load_state_dict(checkpoint["model_state"])
            model.to(device="cuda", memory_format=torch.channels_last)
            mean, std = checkpoint["normalizer_mean"].numpy(), checkpoint["normalizer_std"].numpy()
            tensor = torch.from_numpy(((raw - mean[None, :, None]) / std[None, :, None]).astype(np.float32))[:, None]
            scores = score_inputs(model, tensor, torch.tensor([r["id_class"] for r in rows]), 32)
            original = {r["relative_path"]: r for r in enrich_scores(name, machine, store)}
            cells = []
            for snr in (-6, 0, 6):
                for mid in sorted(store.summary["machine_ids"]):
                    mask = np.array([r["snr_db"] == snr and r["machine_id"] == mid for r in rows])
                    selected = [r for r, keep in zip(rows, mask) if keep]
                    threshold = original[selected[0]["relative_path"]]["threshold"]
                    labels = [r["label"] for r in selected]
                    for channel, values in ((1, [original[r["relative_path"]]["score"] for r in selected]), (2, scores[mask])):
                        cells.append({"channel": channel, "machine_id": mid, "snr_db": snr, **detection_metrics(labels, values, threshold)})
            result = {"model": name, "machine_type": machine, "clips": len(rows), "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(), "cells": cells,
                      "channels": {str(channel): {metric: float(np.mean([c[metric] for c in cells if c["channel"] == channel]))
                                                 for metric in ("auc", "pauc", "false_positive_rate", "true_positive_rate")} for channel in (1, 2)}}
            results.append(result)
            with (root / f"{name}_{machine}_scores.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["relative_path", "machine_id", "snr_db", "label", "channel1_score", "channel2_score", "threshold"])
                writer.writerows((r["relative_path"], r["machine_id"], r["snr_db"], r["label"], original[r["relative_path"]]["score"], float(score), original[r["relative_path"]]["threshold"]) for r, score in zip(rows, scores))
            print(f"Microphone shift {name}/{machine}: AUC {result['channels']['1']['auc']:.4f} → {result['channels']['2']['auc']:.4f}", flush=True)
            del model, tensor, checkpoint
            torch.cuda.empty_cache()
    summary = {"protocol": json.loads((root / "protocol.json").read_text()), "results": results,
               "limitation": "40 normal and 40 anomalous clips per ID/SNR cell; pAUC has limited low-FPR resolution. Same source groups across SNR; not independent samples or a new-session test."}
    # Independently recompute every reported cell from the saved paired CSVs.
    from sklearn.metrics import roc_auc_score
    verified_rows = 0
    for result in results:
        saved = read_csv(root / f"{result['model']}_{result['machine_type']}_scores.csv")
        if len(saved) != result["clips"]:
            raise ValueError("Incomplete paired score file")
        for cell in result["cells"]:
            part = [r for r in saved if r["machine_id"] == cell["machine_id"] and int(r["snr_db"]) == cell["snr_db"]]
            labels = np.array([int(r["label"]) for r in part])
            values = np.array([float(r[f"channel{cell['channel']}_score"]) for r in part])
            for metric, limit in (("auc", None), ("pauc", .1)):
                np.testing.assert_allclose(roc_auc_score(labels, values, max_fpr=limit), cell[metric], rtol=0, atol=1e-12)
            prediction = values > np.array([float(r["threshold"]) for r in part])
            np.testing.assert_allclose(prediction[labels == 0].mean(), cell["false_positive_rate"], rtol=0, atol=1e-12)
            np.testing.assert_allclose(prediction[labels == 1].mean(), cell["true_positive_rate"], rtol=0, atol=1e-12)
        verified_rows += len(saved)
    summary["audit"] = {"state": "passed", "paired_score_rows_checked": verified_rows,
                        "feature_checksums_verified": True, "metrics_recomputed_from_csv": True}
    write_json(root / "summary.json", summary)
    docs = Path("docs/results/failure_analysis")
    write_json(docs / "microphone_shift.json", summary)
    lines = ["# Paired microphone-channel stress test", "", "Same 3,840 held-out clips: original channel 1 versus channel 2. Original normalizers and thresholds retained; no retraining.", "",
             "| Model | Channel | AUC (%) | pAUC (%) | FPR (%) | Fault recall (%) |", "|---|---:|---:|---:|---:|---:|"]
    for name in MODELS:
        for channel in ("1", "2"):
            values = [r["channels"][channel] for r in results if r["model"] == name]
            averages = [100 * np.mean([v[key] for v in values]) for key in ("auc", "pauc", "false_positive_rate", "true_positive_rate")]
            lines.append(f"| {name} | {channel} | " + " | ".join(f"{v:.2f}" for v in averages) + " |")
    lines.extend(["", summary["limitation"], "This measures sensitivity to another channel of the same microphone array. It does not establish generalization to a new factory, session, microphone device or unseen SNR."])
    (docs / "microphone_shift.md").write_text("\n".join(lines) + "\n")
