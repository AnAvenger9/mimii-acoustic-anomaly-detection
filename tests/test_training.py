"""Model, scoring and evaluation invariants for normal-only experiments."""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from mimii.evaluation import detection_metrics, evaluate_scores
from mimii.models import CNN_MODELS, DenseAutoencoder, MachineIdCNN, ResidualBlock, TemporalStatisticsPool
from mimii.train import TrainingConfig, batches, run, score_recordings, known_id_anomaly_scores


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_official_ae_parameter_count_and_shapes(self):
        model = DenseAutoencoder()
        trainable = sum(p.numel() for p in model.parameters())
        running_statistics = sum(layer.running_mean.numel() + layer.running_var.numel()
                                 for layer in model.modules() if isinstance(layer, nn.BatchNorm1d))
        self.assertEqual(trainable, 267928)
        self.assertEqual(trainable + running_statistics, 269992)
        x = torch.randn(5, 640)
        self.assertEqual(model(x).shape, x.shape)
        self.assertEqual(MachineIdCNN()(torch.randn(5, 1, 128, 313)).shape, (5, 4))

    def test_metric_convention_and_direction(self):
        perfect = detection_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], 0.5)
        self.assertEqual(perfect["auc"], 1)
        self.assertEqual(perfect["pauc"], 1)
        self.assertEqual((perfect["tn"], perfect["fp"], perfect["fn"], perfect["tp"]), (2, 0, 0, 2))
        chance = detection_metrics([0, 0, 1, 1], [1, 1, 1, 1], 0.5)
        self.assertEqual(chance["auc"], 0.5)
        self.assertEqual(chance["pauc"], 0.5)
        with self.assertRaises(ValueError):
            detection_metrics([0, 0], [0.1, 0.2], 0.5)

    def test_cnn_variants_support_both_frontends_and_backpropagation(self):
        for name, constructor in CNN_MODELS.items():
            for mel_bands in (64, 128):
                with self.subTest(model=name, mel_bands=mel_bands):
                    model = constructor(classes=4)
                    logits = model(torch.randn(2, 1, mel_bands, 313))
                    self.assertEqual(logits.shape, (2, 4))
                    nn.CrossEntropyLoss()(logits, torch.tensor([0, 1])).backward()
                    self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_residual_path_transmits_signal_with_zero_body(self):
        block = ResidualBlock(8, 8)
        for parameter in block.body.parameters():
            nn.init.zeros_(parameter)
        x = torch.rand(2, 8, 4, 4)
        torch.testing.assert_close(block(x), x)

    def test_temporal_statistics_distinguish_variation_and_have_finite_gradients(self):
        pool = TemporalStatisticsPool()
        steady = torch.ones(1, 1, 4, 4, requires_grad=True)
        varying = torch.tensor([0., 2., 0., 2.]).reshape(1, 1, 1, 4).expand(1, 1, 4, 4)
        a, b = pool(steady), pool(varying)
        torch.testing.assert_close(a[:, :1], b[:, :1])
        self.assertTrue((b[:, 1:] > a[:, 1:]).all())
        a.sum().backward()
        self.assertTrue(torch.isfinite(steady.grad).all())

    def test_threshold_uses_training_scores_only(self):
        test = [{"relative_path": f"{i}.wav", "machine_id": "id_00", "split": "test", "label": i}
                for i in (0, 1)]
        train = [{"relative_path": f"train{i}.wav", "machine_id": "id_00", "split": "train", "label": 0}
                 for i in (0, 1)]
        with tempfile.TemporaryDirectory() as directory, patch("mimii.evaluation.create_evaluation_figures"):
            a = evaluate_scores(test, [1., 20.], train, [0., 2.], directory)
            b = evaluate_scores(test, [3., 200.], train, [0., 2.], directory)
            self.assertAlmostEqual(a["per_id"]["id_00"]["threshold"], 1.9)
            self.assertEqual(a["per_id"]["id_00"]["threshold"], b["per_id"]["id_00"]["threshold"])
            train[0]["label"] = 1
            with self.assertRaises(ValueError):
                evaluate_scores(test, [1., 20.], train, [0., 2.], directory)

    def test_ae_averages_window_errors_to_one_clip_score(self):
        class Store:
            context_frames = 2
            def load(self, row):
                return np.full((2, 4), row["value"], dtype=np.float32)
        class ZeroReconstruction(nn.Module):
            def forward(self, x):
                return torch.zeros_like(x)
        scores = score_recordings(ZeroReconstruction(), Store(), [{"value": 2}, {"value": 3}],
                                  torch.device("cpu"), "autoencoder")
        np.testing.assert_allclose(scores, [4, 9])

    def test_cnn_scores_known_id_even_when_wrong_class_is_confident(self):
        class Store:
            def clip_array(self, rows):
                return np.zeros((len(rows), 1, 2, 2), dtype=np.float32)
        class WrongConfidentClassifier(nn.Module):
            def forward(self, x):
                return torch.tensor([[-5., 5.]]).repeat(len(x), 1)
        scores = score_recordings(WrongConfidentClassifier(), Store(), [{"id_class": 0}, {"id_class": 1}],
                                  torch.device("cpu"), "machine_id_cnn")
        self.assertGreater(scores[0], 9)
        self.assertLess(scores[1], 0.001)

    def test_known_id_score_preserves_tiny_probabilities_without_overflow(self):
        logits = torch.tensor([[50., 0., 0., 0.], [0., 1000., 0., 0.]])
        scores = known_id_anomaly_scores(logits, torch.tensor([0, 0]))
        self.assertGreater(scores[0], 0)
        self.assertAlmostEqual(float(scores[0] / (3 * np.exp(-50))), 1.0, places=12)
        self.assertTrue(torch.isfinite(scores).all())
        self.assertAlmostEqual(float(scores[1]), 1000)
        ordinary = torch.randn(6, 4, dtype=torch.float64)
        targets = torch.tensor([0, 1, 2, 3, 0, 1])
        expected = -ordinary.log_softmax(dim=1)[torch.arange(6), targets]
        torch.testing.assert_close(known_id_anomaly_scores(ordinary, targets), expected)

    def test_singleton_batch_is_merged_without_dropping_samples(self):
        indices = batches(9, 4)
        self.assertEqual([len(part) for part in indices], [4, 5])
        torch.testing.assert_close(torch.cat(indices), torch.arange(9))

    def test_training_end_to_end_on_small_feature_store(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feature_root = root / "features"
            feature_root.mkdir()
            rng = np.random.default_rng(8)
            rows = []
            for mid, id_class in (("id_00", 0), ("id_02", 1)):
                for index, (split, label) in enumerate((("train", 0), ("train", 0), ("train", 0),
                                                        ("validation", 0), ("test", 0), ("test", 1))):
                    name = f"{mid}_{index}.npy"
                    np.save(feature_root / name, rng.normal(size=(16, 32)).astype(np.float32))
                    rows.append({"relative_path": name.replace(".npy", ".wav"), "feature_path": name,
                                 "machine_id": mid, "id_class": id_class, "label": label, "split": split,
                                 "waveform_sha256": name, "feature_shape": [16, 32]})
            manifest = "".join(json.dumps(r) + "\n" for r in rows)
            (feature_root / "manifest.jsonl").write_text(manifest)
            digest = hashlib.sha256(manifest.encode()).hexdigest()
            (feature_root / "status.json").write_text(json.dumps({"state": "complete", "manifest_sha256": digest}))
            (feature_root / "config.json").write_text(json.dumps({"n_mels": 16, "context_frames": 5, "seed": 42}))
            (feature_root / "summary.json").write_text(json.dumps({"machine_ids": {"id_00": 0, "id_02": 1}}))
            config = TrainingConfig(feature_dir=str(feature_root), output_dir=str(root / "run"),
                                    seeds=(1,), epochs=2, batch_size=32, cpu_threads=1, device="cpu")
            with patch("mimii.evaluation.create_evaluation_figures"), patch("mimii.train.plot_history"), contextlib.redirect_stdout(io.StringIO()):
                result = run(config)
            self.assertTrue(np.isfinite(result["macro_auc_mean"]))
            output = root / "run" / "seed_1"
            self.assertEqual(json.loads((output / "status.json").read_text())["state"], "complete")
            self.assertTrue((output / "model.pt").exists())
            self.assertEqual(len((output / "scores.csv").read_text().splitlines()), 5)
            with self.assertRaises(FileExistsError), contextlib.redirect_stdout(io.StringIO()):
                run(config)


if __name__ == "__main__":
    unittest.main()
