"""Window isolation, full coverage, clip scoring and exact CPU resume."""

import copy
import contextlib
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from mimii.ae_experiment import AEConfig, assert_normal_splits, train_seed, summarize
from mimii.ae_report import audit_run, publish_report
from mimii.ae_windows import WindowBank, batch_slices, score_bank, train_epoch
from mimii.features import context_vectors
from mimii.models import DenseAutoencoder


class AEWindowTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_all_windows_match_reference_without_crossing_clips(self):
        features = torch.arange(3 * 4 * 9, dtype=torch.float32).reshape(3, 4, 9)
        bank = WindowBank(features, 5)
        expected = np.concatenate([context_vectors(x.numpy(), 5) for x in features])
        order = torch.tensor([14, 0, 5, 4, 9, 10, 3, 7, 2, 11, 1, 6, 8, 12, 13])
        np.testing.assert_array_equal(bank.batch(order).numpy(), expected[order.numpy()])

    def test_full_coverage_including_singleton_tail(self):
        for count in (2, 3, 511, 512, 513, 1024, 1025, 1537):
            slices = list(batch_slices(count, 512))
            self.assertTrue(all(stop - start >= 2 for start, stop in slices))
            self.assertEqual([i for start, stop in slices for i in range(start, stop)], list(range(count)))

    def test_scores_average_features_then_all_windows(self):
        features = torch.arange(2 * 3 * 8, dtype=torch.float32).reshape(2, 3, 8)
        bank = WindowBank(features, 5)
        zero = torch.nn.Linear(15, 15, bias=False)
        with torch.no_grad():
            zero.weight.zero_()
        scores = score_bank(zero, bank, batch_size=3)
        expected = np.array([(context_vectors(x.numpy(), 5) ** 2).mean() for x in features])
        np.testing.assert_allclose(scores, expected, rtol=1e-6)

    def test_epoch_resume_restores_shuffle_optimizer_and_batchnorm(self):
        torch.manual_seed(7)
        bank = WindowBank(torch.randn(3, 2, 9), 5)
        model = DenseAutoencoder(10, 3)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, eps=1e-7)
        generator = torch.Generator().manual_seed(42)
        train_epoch(model, optimizer, bank, 4, generator)
        model_state, optimizer_state, rng = copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict()), generator.get_state()
        expected_loss = train_epoch(model, optimizer, bank, 4, generator)
        restored = DenseAutoencoder(10, 3)
        restored.load_state_dict(model_state)
        restored_optimizer = torch.optim.Adam(restored.parameters(), lr=0.001, eps=1e-7)
        restored_optimizer.load_state_dict(optimizer_state)
        restored_generator = torch.Generator().set_state(rng)
        loss = train_epoch(restored, restored_optimizer, bank, 4, restored_generator)
        self.assertEqual(loss, expected_loss)
        for key, tensor in model.state_dict().items():
            torch.testing.assert_close(tensor, restored.state_dict()[key], rtol=0, atol=0)

    def test_only_healthy_training_and_validation_are_accepted(self):
        train = [{"label": 0, "split": "train"}]
        validation = [{"label": 0, "split": "validation"}]
        assert_normal_splits(train, validation)
        for invalid in ([{"label": 1, "split": "train"}], [{"label": 0, "split": "test"}], []):
            with self.assertRaises(ValueError):
                assert_normal_splits(invalid, validation)

    def test_frozen_baseline_settings(self):
        config = AEConfig()
        self.assertEqual((config.epochs, config.batch_size, config.latent_dimensions), (100, 512, 8))
        for kwargs in ({"normalization": "train_per_mel"}, {"device": "cpu"}, {"batch_size": 1}, {"seeds": [42, 42]}):
            with self.assertRaises(ValueError):
                AEConfig(**kwargs)

    def test_runner_resume_and_both_evaluation_views(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rng = np.random.default_rng(5)
            records, arrays = [], {}
            for snr in (-6, 0, 6):
                for i, (split, label) in enumerate((("train", 0), ("validation", 0), ("test", 0), ("test", 1))):
                    path = f"{snr}/{i}.wav"
                    records.append({"relative_path": path, "machine_id": "id_00", "split": split, "label": label, "snr_db": snr})
                    arrays[path] = rng.normal(size=(2, 9)).astype(np.float32)
            store = SimpleNamespace(shape=(2, 9), context_frames=5, config={}, manifest_sha256="fixture")
            store.load = lambda row: arrays[row["relative_path"]]
            store.split = lambda split: [r for r in records if r["split"] == split]
            train = WindowBank.from_store(store, store.split("train"), "cpu")
            validation = WindowBank.from_store(store, store.split("validation"), "cpu")
            config = AEConfig(output_dir=str(root / "complete"), machine_types=("fan",), seeds=(42,), epochs=2,
                              device="cpu", cuda_graphs=False, batch_size=4, evaluation_batch_size=3)
            other = replace(config, output_dir=str(root / "resumed"))
            calls = 0
            def interrupt(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("simulated interruption")
                return train_epoch(*args, **kwargs)
            with patch("mimii.evaluation.create_evaluation_figures"), contextlib.redirect_stdout(io.StringIO()):
                full = train_seed(config, "fan", 42, store, train, validation, Path(config.output_dir))
                with patch("mimii.ae_experiment.train_epoch", side_effect=interrupt), self.assertRaisesRegex(RuntimeError, "simulated"):
                    train_seed(other, "fan", 42, store, train, validation, Path(other.output_dir))
                resumed = train_seed(other, "fan", 42, store, train, validation, Path(other.output_dir))
            for view in ("final", "best_validation"):
                self.assertEqual(full[view], resumed[view])
                checkpoint = torch.load(root / f"resumed/fan/seed_42/{view}.pt", weights_only=False)
                self.assertEqual(checkpoint["normalization"], "none")
            report = summarize(Path(other.output_dir), other)
            self.assertTrue(report["complete"])
            self.assertEqual(report["views"]["final"]["overall"]["auc"]["mean"], full["final"]["macro_auc"])
            self.assertEqual(train_seed(other, "fan", 42, store, train, validation, Path(other.output_dir)), resumed)
            incomplete = summarize(Path(other.output_dir), replace(other, seeds=(42, 43)))
            self.assertFalse(incomplete["complete"])
            self.assertEqual(incomplete["views"]["final"]["overall"], {})
            report = summarize(Path(other.output_dir), other)
            previous_directory = Path.cwd()
            try:
                os.chdir(root)
                (Path(other.output_dir) / "protocol.json").write_text("{}\n")
                publish_report(Path(other.output_dir), report, other)
                self.assertTrue(Path("docs/results/dcase2020_ae/artifact_hashes.json").exists())
                self.assertTrue(Path("docs/results/dcase2020_ae/ae_mobilenet_comparison.pdf").exists())
            finally:
                os.chdir(previous_directory)
            score_path = Path(other.output_dir) / "fan/seed_42/final/scores.csv"
            score_path.write_text(score_path.read_text().replace("id_00", "id_99", 1))
            with self.assertRaisesRegex(ValueError, "machine ID mismatch"):
                audit_run(Path(other.output_dir) / "fan/seed_42", store, other)
