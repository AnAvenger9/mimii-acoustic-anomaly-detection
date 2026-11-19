"""Continue selected CNN fits in a separate experiment, preserving prior results."""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))
for key, value in {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "OPENBLAS_NUM_THREADS": "1",
                   "OMP_NUM_THREADS": "2", "MPLCONFIGDIR": "/tmp/mimii-matplotlib"}.items():
    os.environ.setdefault(key, value)

import torch
from mimii.ae_experiment import write_json
from mimii.cnn_optimization import BudgetExpired, evaluate, fit, make_config
from mimii.finetune import configure_runtime
from mimii.report_cnn_optimization import report
from mimii.training_data import FeatureStore
from mimii.transfer_models import pretrained_hash


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(parent, output, target_epochs, minutes):
    parent_protocol = json.loads((parent / "protocol.json").read_text())
    if json.loads((parent / "status.json").read_text())["state"] != "complete":
        raise ValueError("Continuation requires a completed parent study")
    if json.loads((parent / "validation.json").read_text())["state"] != "passed":
        raise ValueError("Parent study has not passed its result audit")
    parent_plan = json.loads((parent / "final_plan.json").read_text())
    if target_epochs <= parent_plan["target_epochs"]:
        raise ValueError("Target epoch must extend the parent schedule")
    study = {**parent_protocol["identity"]["study"], "output_dir": str(output),
             "final_epochs": target_epochs, "wall_seconds": minutes * 60,
             "evaluation_reserve_seconds": 12 * 60}
    selection = json.loads((parent / "selection.json").read_text())
    parent_files = [parent / name for name in ("protocol.json", "summary.json", "validation.json", "selection.json", "screening.json", "final_plan.json")]
    parent_files.extend(parent.glob("*/lr_*/*/history.json"))
    for architecture, chosen in selection.items():
        for machine in study["machine_types"]:
            directory = parent / architecture / f"lr_{chosen['selected_trial']}" / machine
            parent_files.extend(directory / name for name in ("last.pt", "model.pt", "fit.json"))
    sources = [*sorted(Path("src/mimii").glob("*.py")), Path("scripts/train_cnn_optimization.py"), Path(__file__).relative_to(ROOT)]
    identity = {**parent_protocol["identity"], "study": study,
                "source_sha256": {str(p): digest(p) for p in sources},
                "versions": {name: version(name) for name in ("torch", "torchvision", "numpy", "scikit-learn")},
                "pretrained_sha256": {a: pretrained_hash(a) for a in study["architectures"]},
                "parent_dir": str(parent), "parent_artifact_sha256": {str(p.relative_to(parent)): digest(p) for p in parent_files}}
    for key in ("versions", "pretrained_sha256"):
        if identity[key] != parent_protocol["identity"][key]:
            raise ValueError(f"Continuation changed {key}")
    if digest(Path(study["feature_root"]) / "manifest.jsonl") != identity["master_manifest_sha256"]:
        raise ValueError("Master feature manifest changed")
    for machine, expected in identity["normalizer_sha256"].items():
        if digest(Path(study["feature_root"]) / "by_machine" / machine / "normalizer.npz") != expected:
            raise ValueError("Training normalizer changed")
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        protocol = json.loads(protocol_path.read_text())
        if protocol["identity"] != identity:
            raise ValueError("Continuation code, parent artifacts, data or settings changed")
        return study, selection, protocol
    output.mkdir(parents=True, exist_ok=True)
    # Copy, never hard-link: extended checkpoints must not modify the parent.
    for path in parent_files:
        relative = path.relative_to(parent)
        if relative.parts[0] in study["architectures"]:
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            if digest(destination) != identity["parent_artifact_sha256"][str(relative)]:
                raise ValueError("Initial checkpoint/history copy failed integrity check")
    for name in ("selection.json", "screening.json"):
        shutil.copy2(parent / name, output / name)
    for path in sources:
        destination = output / "source_snapshot" / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    now = time.time()
    protocol = {"identity": identity, "started_unix": now, "deadline_unix": now + minutes * 60,
                "selection": "Original healthy-validation-selected rates retained; minimum healthy-validation loss over all epochs selects weights",
                "motivation": "Five of eight fits had their minimum healthy-validation loss at epoch 6",
                "test_reuse": "Parent test results were already inspected. This is a documented exploratory continuation, not a fresh blind test.",
                "initial_state": "Resume last epoch, optimizer and Torch RNG states; retain the previous best-validation checkpoint",
                "statistical_limit": "One initialization seed (42), no seed-variability estimate"}
    write_json(output / "config.json", study)
    write_json(output / "final_plan.json", {"target_epochs": target_epochs, "selection": selection,
               "reason": "User-authorized continuation; fixed horizon chosen before extended test evaluation"})
    write_json(protocol_path, protocol)
    return study, selection, protocol


def run(parent, output, target_epochs, minutes, publish_dir):
    study, selection, protocol = prepare(parent, output, target_epochs, minutes)
    if (output / "status.json").exists() and json.loads((output / "status.json").read_text()).get("state") == "complete":
        print("Continuation is already complete.", flush=True)
        return
    configure_runtime(make_config(study, study["architectures"][0], study["backbone_learning_rates"][0]))
    deadline = protocol["deadline_unix"]
    write_json(output / "status.json", {"state": "loading_features", "target_epochs": target_epochs,
               "deadline_utc": datetime.fromtimestamp(deadline, timezone.utc).isoformat()})
    stores, inputs = {}, {}
    for machine in study["machine_types"]:
        print(f"Loading {machine}; continuing saved epoch-6 models to epoch {target_epochs}", flush=True)
        store = FeatureStore(Path(study["feature_root"]) / "by_machine" / machine, "train_per_mel")
        train, validation = store.split("train"), store.split("validation")
        if any(r["label"] != 0 for r in train + validation):
            raise ValueError("Non-normal clip in fitting/selection data")
        for architecture, chosen in selection.items():
            original = torch.load(parent / architecture / f"lr_{chosen['selected_trial']}" / machine / "model.pt", map_location="cpu", weights_only=False)
            if original["manifest_sha256"] != store.manifest_sha256:
                raise ValueError("Per-machine manifest changed since the parent fit")
        stores[machine] = store
        inputs[machine] = (torch.from_numpy(store.clip_array(train)), torch.tensor([r["id_class"] for r in train]),
                           torch.from_numpy(store.clip_array(validation)), torch.tensor([r["id_class"] for r in validation]))
    for architecture, chosen in selection.items():
        for machine in study["machine_types"]:
            fit(study, architecture, chosen["selected_trial"], machine, stores[machine], inputs[machine], target_epochs,
                deadline - study["evaluation_reserve_seconds"])
    write_json(output / "status.json", {"state": "evaluating", "trained_epochs": target_epochs})
    results = []
    for architecture, chosen in selection.items():
        for machine in study["machine_types"]:
            if time.time() >= deadline:
                raise BudgetExpired("Continuation budget reached during evaluation; artifacts preserved")
            results.append(evaluate(study, architecture, chosen["selected_trial"], machine, stores[machine], inputs[machine]))
            write_json(output / "partial_results.json", results)
    report(output, study, results, stores, publish_dir=publish_dir)
    before = json.loads((parent / "summary.json").read_text())
    after = json.loads((output / "summary.json").read_text())
    comparison = {"parent_epochs": 6, "extended_epochs": target_epochs,
                  "selection": "healthy-validation loss only; anomaly metrics below are descriptive, not selection criteria",
                  "models": {a: {metric: {"parent": before["models"][a][metric], "extended": after["models"][a][metric],
                              "change_percentage_points": 100 * (after["models"][a][metric] - before["models"][a][metric])}
                              for metric in ("macro_auc", "macro_pauc")} for a in study["architectures"]}}
    write_json(output / "epoch_comparison.json", comparison)
    note = ("\nThis experiment continues the preserved six-epoch study from its last checkpoints. "
            "The same learning rates, seed and splits were retained. Parent test results had already been inspected; "
            "this is an exploratory continuation. Checkpoints were selected only by healthy-validation loss.\n")
    with (output / "RESULTS.md").open("a") as handle:
        handle.write(note)
        handle.write("\n| Model | Six-epoch AUC / pAUC (%) | Extended AUC / pAUC (%) |\n|---|---:|---:|\n")
        for a in study["architectures"]:
            b, e = before["models"][a], after["models"][a]
            handle.write(f"| {a} | {100*b['macro_auc']:.2f} / {100*b['macro_pauc']:.2f} | {100*e['macro_auc']:.2f} / {100*e['macro_pauc']:.2f} |\n")
    destination = Path(publish_dir)
    for name in ("RESULTS.md", "epoch_comparison.json"):
        shutil.copy2(output / name, destination / name)
    write_json(destination / "artifact_hashes.json", {p.name: digest(p) for p in destination.iterdir() if p.name != "artifact_hashes.json"})
    # Verify that parent artifacts remained unchanged throughout continuation.
    for relative, expected in protocol["identity"]["parent_artifact_sha256"].items():
        if digest(parent / relative) != expected:
            raise ValueError("Parent artifact changed during the continuation")
    write_json(output / "status.json", {"state": "complete", "completed_fits": len(results), "target_epochs": target_epochs,
               "elapsed_seconds": time.time() - protocol["started_unix"], "parent_artifacts_unchanged": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", default="runs/cnn_optimization_2h")
    parser.add_argument("--output", default="runs/cnn_optimization_12epochs")
    parser.add_argument("--publish-dir", default="docs/results/cnn_optimization_12epochs")
    parser.add_argument("--target-epochs", type=int, default=12)
    parser.add_argument("--minutes", type=int, default=75)
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    parent, output = Path(args.parent), Path(args.output)
    if output.resolve() == parent.resolve() or Path(args.publish_dir).resolve() == Path("docs/results/cnn_optimization_2h").resolve():
        raise ValueError("Continuation must preserve the parent output and published results")
    if args.minutes <= 12:
        raise ValueError("Budget must leave time beyond the evaluation reserve")
    output.mkdir(parents=True, exist_ok=True)
    if args.detach:
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--parent", str(parent), "--output", str(output),
                   "--publish-dir", args.publish_dir, "--target-epochs", str(args.target_epochs), "--minutes", str(args.minutes)]
        with (output / "training.log").open("a") as log:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write_json(output / "worker.json", {"pid": process.pid})
        print(f"Started continuation PID {process.pid}; log: {output / 'training.log'}")
        return
    with (output / "worker.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another worker owns this continuation")
        try:
            run(parent, output, args.target_epochs, args.minutes, args.publish_dir)
        except BudgetExpired as error:
            write_json(output / "status.json", {"state": "budget_exhausted", "reason": str(error)})
        except BaseException as error:
            write_json(output / "status.json", {"state": "failed", "error": repr(error)})
            raise


if __name__ == "__main__":
    main()
