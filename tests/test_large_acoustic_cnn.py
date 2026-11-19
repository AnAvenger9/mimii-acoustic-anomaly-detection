"""Validate wider-model training coverage and comparison metric semantics."""

from dataclasses import asdict
import json
from pathlib import Path
import unittest

import torch

from mimii.custom_cnn_experiment import variant_config as small_config
from mimii.large_acoustic_cnn import LargeAcousticResidualStatsCNN
from mimii.large_cnn_experiment import operating_point, variant_config
from mimii.mobilenet import make_optimizer


class LargeAcousticTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_forward_backward_optimizer_coverage_and_checkpoint(self):
        model = LargeAcousticResidualStatsCNN()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 1286964)
        study = json.loads(Path("configs/train_large_acoustic_cnn.json").read_text())
        optimizer = make_optimizer(model, variant_config(study), False)
        optimized = [p for group in optimizer.param_groups for p in group["params"]]
        self.assertEqual(len(optimized), len(set(map(id, optimized))))
        self.assertEqual(set(map(id, optimized)), set(map(id, model.parameters())))
        x = torch.randn(2, 1, 128, 313)
        self.assertEqual(model.features(x).shape, (2, 256, 4, 20))
        logits = model(x)
        self.assertEqual(logits.shape, (2, 4))
        torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        optimizer.step()
        restored = LargeAcousticResidualStatsCNN()
        restored.load_state_dict(model.state_dict())
        model.eval()
        restored.eval()
        with torch.no_grad():
            torch.testing.assert_close(model(x), restored(x), rtol=0, atol=0)

    def test_training_settings_match_small_model(self):
        small = asdict(small_config(json.loads(Path("configs/train_acoustic_cnn.json").read_text()), True))
        large = asdict(variant_config(json.loads(Path("configs/train_large_acoustic_cnn.json").read_text())))
        large.pop("architecture")
        small.pop("output_dir")
        large.pop("output_dir")
        self.assertEqual(small, large)

    def test_operating_point_counts_clips_instead_of_averaging_cells(self):
        actual = operating_point([{"tp": 1, "fn": 0, "fp": 0, "tn": 1},
                                  {"tp": 0, "fn": 9, "fp": 9, "tn": 0}])
        self.assertEqual(actual["fault_recall"], .1)
        self.assertEqual(actual["false_positive_rate"], .9)
