"""CNN adaptation and validation-only optimization/resume invariants."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from mimii.cnn_optimization import BudgetExpired, choose_trials, evaluate, fit, score_inputs
from mimii.report_cnn_optimization import audit
from mimii.mobilenet import set_training_stage
from mimii.transfer_models import build_transfer, grayscale_conv


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(1, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU())
        self.classifier = nn.Sequential(nn.Dropout(.2), nn.Linear(4, 2))

    def forward(self, x):
        return self.classifier(self.features(x).mean((2, 3)))


class CNNOptimizationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_grayscale_adaptation_matches_rgb_replication(self):
        original = nn.Conv2d(3, 8, 3, stride=2, padding=1, bias=True)
        x = torch.randn(2, 1, 19, 23)
        torch.testing.assert_close(grayscale_conv(original)(x), original(x.repeat(1, 3, 1, 1)))

    def test_backbones_accept_native_spectrograms_and_freeze_batchnorm(self):
        for architecture, parameters in (("resnet18", 11172292), ("efficientnet_b0", 4012096)):
            model = build_transfer(architecture, pretrained=False)
            self.assertEqual(sum(p.numel() for p in model.parameters()), parameters)
            set_training_stage(model, True)
            self.assertFalse(model.features.training)
            self.assertTrue(all(not p.requires_grad for p in model.features.parameters()))
            self.assertEqual(model(torch.randn(2, 1, 128, 313)).shape, (2, 4))
            set_training_stage(model, False)
            self.assertTrue(all(p.requires_grad for p in model.features.parameters()))

    def test_selection_uses_macro_validation_not_fault_metrics(self):
        rows = [{"architecture": "resnet18", "trial": trial, "machine_type": machine,
                 "validation_loss": loss, "macro_auc": 1 - loss}
                for trial, losses in ((0, (.1, .8)), (1, (.3, .3))) for machine, loss in zip(("fan", "pump"), losses)]
        selected = choose_trials(rows, ["resnet18"], ["fan", "pump"], 2)
        self.assertEqual(selected["resnet18"]["selected_trial"], 1)
        for row in rows:
            row["macro_auc"] = -100 * row["macro_auc"]
        self.assertEqual(choose_trials(rows, ["resnet18"], ["fan", "pump"], 2), selected)
        with self.assertRaises(ValueError):
            choose_trials(rows[:-1], ["resnet18"], ["fan", "pump"], 2)

    def test_staged_fit_resume_exact_and_no_test_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            study = {"feature_root": "unused", "output_dir": str(root / "full"), "seed": 42,
                     "backbone_learning_rates": [1e-4], "head_lr_multiplier": 3, "final_epochs": 3,
                     "warmup_epochs": 1, "batch_sizes": {"resnet18": 4}, "weight_decay": 1e-4,
                     "threshold_quantile": .95, "device": "cpu"}
            store = SimpleNamespace(summary={"machine_ids": {"id_00": 0, "id_02": 1}},
                                    mean=np.zeros(8, np.float32), std=np.ones(8, np.float32), config={}, manifest_sha256="fixture")
            torch.manual_seed(13)
            inputs = (torch.randn(12, 1, 8, 16), torch.arange(12) % 2, torch.randn(6, 1, 8, 16), torch.arange(6) % 2)
            with patch("mimii.cnn_optimization.build_transfer", side_effect=lambda *a, **k: TinyCNN()), contextlib.redirect_stdout(io.StringIO()):
                full = fit(study, "resnet18", 0, "fan", store, inputs, 3, time.time() + 100)
                resumed_study = {**study, "output_dir": str(root / "resumed")}
                fit(resumed_study, "resnet18", 0, "fan", store, inputs, 1, time.time() + 100)
                resumed = fit(resumed_study, "resnet18", 0, "fan", store, inputs, 3, time.time() + 100)
                with self.assertRaises(BudgetExpired):
                    fit({**study, "output_dir": str(root / "expired")}, "resnet18", 0, "fan", store, inputs, 3, time.time() - 1)
            self.assertEqual(full["validation_loss"], resumed["validation_loss"])
            self.assertEqual(full["selected_epoch"], resumed["selected_epoch"])
            left = torch.load(root / "full/resnet18/lr_0/fan/last.pt", weights_only=False)
            right = torch.load(root / "resumed/resnet18/lr_0/fan/last.pt", weights_only=False)
            for key in left["model_state"]:
                torch.testing.assert_close(left["model_state"][key], right["model_state"][key], rtol=0, atol=0)
            self.assertFalse((root / "full/resnet18/lr_0/fan/scores.csv").exists())

    def test_known_id_scoring_is_batch_size_independent(self):
        model = TinyCNN().eval()
        x, y = torch.randn(7, 1, 8, 16), torch.arange(7) % 2
        np.testing.assert_allclose(score_inputs(model, x, y, 2), score_inputs(model, x, y, 7), rtol=1e-6)

    def test_selected_fit_scoring_and_saved_result_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            study = {"feature_root": "unused", "output_dir": str(root), "seed": 42,
                     "architectures": ["resnet18"], "machine_types": ["fan"],
                     "backbone_learning_rates": [1e-4], "head_lr_multiplier": 3, "final_epochs": 2,
                     "screening_epochs": 2, "warmup_epochs": 1, "batch_sizes": {"resnet18": 4},
                     "weight_decay": 1e-4, "threshold_quantile": .95, "device": "cpu"}
            rng = np.random.default_rng(8)
            records, arrays = [], {}
            for snr in (-6, 0, 6):
                for cls, mid in enumerate(("id_00", "id_02")):
                    for index, (split, label) in enumerate((("train", 0), ("validation", 0), ("test", 0), ("test", 1))):
                        path = f"{snr}/{mid}/{index}.wav"
                        records.append({"relative_path": path, "machine_id": mid, "id_class": cls,
                                        "split": split, "label": label, "snr_db": snr})
                        arrays[path] = rng.normal(size=(8, 16)).astype(np.float32)
            store = SimpleNamespace(summary={"machine_ids": {"id_00": 0, "id_02": 1}},
                                    mean=np.zeros(8, np.float32), std=np.ones(8, np.float32), config={}, manifest_sha256="fixture")
            store.split = lambda split: [r for r in records if r["split"] == split]
            store.clip_array = lambda rows: np.stack([arrays[r["relative_path"]] for r in rows])[:, None]
            inputs = tuple(value for split in ("train", "validation") for value in
                           (torch.from_numpy(store.clip_array(store.split(split))), torch.tensor([r["id_class"] for r in store.split(split)])))
            with patch("mimii.cnn_optimization.build_transfer", side_effect=lambda *a, **k: TinyCNN()), \
                    patch("mimii.evaluation.create_evaluation_figures"), patch("mimii.cnn_optimization.plot_history"), \
                    contextlib.redirect_stdout(io.StringIO()):
                fitted = fit(study, "resnet18", 0, "fan", store, inputs, 2, time.time() + 100)
                result = evaluate(study, "resnet18", 0, "fan", store, inputs)
            selection = choose_trials([fitted], ["resnet18"], ["fan"], 1)
            for name, value in (("protocol.json", {"identity": {"source_sha256": {}}}), ("selection.json", selection), ("screening.json", [fitted])):
                (root / name).write_text(json.dumps(value))
            checked = audit(root, study, [result], {"fan": store})
            self.assertEqual(checked["state"], "passed")
            self.assertEqual(checked["checked_test_scores"], 12)
            scores = root / "resnet18/lr_0/fan/scores.csv"
            scores.write_text(scores.read_text().replace("id_00", "id_99", 1))
            with self.assertRaisesRegex(ValueError, "Saved scores do not cover exact split|Machine ID mismatch"):
                audit(root, study, [result], {"fan": store})
