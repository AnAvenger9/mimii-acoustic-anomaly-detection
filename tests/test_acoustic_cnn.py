"""Custom-model shape, pooling hypothesis, scratch-training and resume checks."""

import contextlib
from dataclasses import replace
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from mimii.acoustic_cnn import AcousticResidualStatsCNN, FrequencyStatisticsPool
from mimii.custom_cnn_experiment import AcousticConfig
from mimii.failure_analysis import choose_microphone_sample
from mimii.finetune import train_seed


class AcousticCNNTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_native_shape_and_finite_gradients_in_both_variants(self):
        for enabled in (True, False):
            model = AcousticResidualStatsCNN(include_std=enabled)
            self.assertEqual(sum(p.numel() for p in model.parameters()), 177724)
            x = torch.randn(2, 1, 128, 313)
            self.assertEqual(model.features(x).shape, (2, 80, 4, 20))
            logits = model(x)
            self.assertEqual(logits.shape, (2, 4))
            torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_std_pool_distinguishes_variation_with_equal_mean(self):
        steady = torch.ones(2, 3, 4, 8, requires_grad=True)
        varying = torch.tensor([0., 2.] * 4).expand(2, 3, 4, 8)
        full, mean_only = FrequencyStatisticsPool(True), FrequencyStatisticsPool(False)
        a, b = full(steady), full(varying)
        torch.testing.assert_close(a[:, :3], b[:, :3])
        self.assertTrue((b[:, 3:] > a[:, 3:]).all())
        torch.testing.assert_close(mean_only(steady), mean_only(varying))
        self.assertEqual(torch.count_nonzero(mean_only(steady)[:, 3:]).item(), 0)
        a.sum().backward()
        self.assertTrue(torch.isfinite(steady.grad).all())

    def test_sample_selection_is_score_independent_and_source_paired(self):
        rows = [{"machine_id": mid, "label": label, "snr_db": snr, "split": "test",
                 "source_recording_key": f"{mid}/{label}/{i}", "relative_path": f"{snr}/{mid}/{label}/{i}"}
                for mid in ("id_00", "id_02") for label in (0, 1) for i in range(6) for snr in (-6, 0, 6)]
        a = choose_microphone_sample(rows, 3)
        b = choose_microphone_sample(list(reversed(rows)), 3)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 36)
        with self.assertRaises(ValueError):
            choose_microphone_sample(rows, 7)

    def test_random_factory_has_no_pretrained_dependency_and_resumes(self):
        # Exercise the production trainer's new factory path on real custom layers.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            np.savez(root / "normalizer.npz", mean=np.zeros(128), std=np.ones(128))
            rng = np.random.default_rng(9)
            records, arrays = [], {}
            for snr in (-6, 0, 6):
                for cls, mid in enumerate(("id_00", "id_02")):
                    for i, (split, label) in enumerate((("train", 0), ("validation", 0), ("test", 0), ("test", 1))):
                        path = f"{snr}/{mid}/{i}.wav"
                        records.append({"relative_path": path, "machine_id": mid, "id_class": cls, "snr_db": snr, "split": split, "label": label})
                        arrays[path] = rng.normal(size=(128, 64)).astype(np.float32)
            store = SimpleNamespace(root=root, manifest_sha256="fixture", config={}, summary={"machine_ids": {"id_00": 0, "id_02": 1}},
                                    mean=np.zeros(128, np.float32), std=np.ones(128, np.float32))
            store.split = lambda split: [r for r in records if r["split"] == split]
            store.clip_array = lambda rows: np.stack([arrays[r["relative_path"]] for r in rows])[:, None]
            inputs = tuple(v for split in ("train", "validation") for v in
                           (torch.from_numpy(store.clip_array(store.split(split))), torch.tensor([r["id_class"] for r in store.split(split)])))
            cfg = AcousticConfig(output_dir=str(root / "run"), pretrained_path="absent", seeds=(42,), epochs=2,
                                 warmup_epochs=0, patience=5, batch_size=3, evaluation_batch_size=3, device="cpu", mixed_precision=False)
            with patch("mimii.finetune.plot_history"), patch("mimii.evaluation.create_evaluation_figures"), contextlib.redirect_stdout(io.StringIO()):
                result = train_seed(cfg, "fan", 42, store, torch.device("cpu"), inputs,
                                    model_factory=lambda n: AcousticResidualStatsCNN(n), model_name="custom_test")
                repeated = train_seed(cfg, "fan", 42, store, torch.device("cpu"), inputs,
                                      model_factory=lambda n: AcousticResidualStatsCNN(n), model_name="custom_test")
            self.assertEqual(result, repeated)
            self.assertEqual(result["model"], "custom_test")
            checkpoint = torch.load(root / "run/fan/seed_42/model.pt", weights_only=False)
            self.assertIsNone(checkpoint["config"]["pretrained_sha256"])
            self.assertEqual(checkpoint["config"]["initialization"], "random, no pretrained weights")
