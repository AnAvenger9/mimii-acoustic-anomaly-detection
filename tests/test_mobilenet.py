"""Transfer-learning invariants: RGB adaptation, freezing, and exact CPU resume."""

import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torchvision.models import mobilenet_v3_small

from mimii.finetune import FinetuneConfig, epoch_pass, train_seed
from mimii.mobilenet import build_mobilenet, convert_first_convolution, make_optimizer, weights_sha256


class TinyTransfer(nn.Module):
    def __init__(self, classes=2):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(1, 4, 3, padding=1, bias=False), nn.BatchNorm2d(4), nn.ReLU())
        self.classifier = nn.Sequential(nn.Linear(4, classes))

    def forward(self, x):
        return self.classifier(self.features(x).mean(dim=(2, 3)))


class MobileNetTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_summed_kernels_match_replicated_input_and_native_shape(self):
        model = mobilenet_v3_small(weights=None)
        x = torch.randn(2, 1, 128, 313)
        expected = model.features[0][0](x.repeat(1, 3, 1, 1))
        convert_first_convolution(model)
        torch.testing.assert_close(model.features[0][0](x), expected, rtol=1e-5, atol=1e-5)
        adapted = build_mobilenet(4)
        self.assertEqual(adapted(x).shape, (2, 4))
        self.assertEqual(sum(p.numel() for p in adapted.parameters()), 1521668)

    def test_warmup_freezes_backbone_parameters_and_running_statistics(self):
        model = TinyTransfer()
        config = FinetuneConfig(device="cpu", batch_size=4, mixed_precision=False)
        x, y = torch.randn(8, 1, 8, 16), torch.arange(8) % 2
        before = {k: v.clone() for k, v in model.features.state_dict().items()}
        head_before = model.classifier[0].weight.detach().clone()
        optimizer = make_optimizer(model, config, True)
        epoch_pass(model, x, y, config, torch.device("cpu"), optimizer, True)
        for key, value in model.features.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        self.assertFalse(torch.equal(head_before, model.classifier[0].weight))
        optimizer = make_optimizer(model, config, False)
        epoch_pass(model, x, y, config, torch.device("cpu"), optimizer, False)
        self.assertFalse(torch.equal(before["0.weight"], model.features[0].weight))
        self.assertFalse(torch.equal(before["1.running_mean"], model.features[1].running_mean))

    def test_weight_integrity_rejects_wrong_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "weights.pth"
            path.write_bytes(b"not official weights")
            with self.assertRaisesRegex(ValueError, "official PyTorch hash"):
                weights_sha256(path)

    def test_interrupted_training_resumes_across_unfreeze_boundary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            np.savez(root / "normalizer.npz", mean=np.zeros(8), std=np.ones(8))
            rng = np.random.default_rng(3)
            records, arrays = [], {}
            for id_class, mid in enumerate(("id_00", "id_02")):
                for snr in (-6, 0, 6):
                    for index, (split, label) in enumerate((("train", 0), ("train", 0), ("validation", 0), ("test", 0), ("test", 1))):
                        path = f"{snr}/{mid}/{index}.wav"
                        records.append({"relative_path": path, "machine_id": mid, "id_class": id_class,
                                        "split": split, "label": label, "snr_db": snr})
                        arrays[path] = rng.normal(size=(8, 16)).astype(np.float32)
            store = SimpleNamespace(root=root, manifest_sha256="fixture", config={},
                summary={"machine_ids": {"id_00": 0, "id_02": 1}}, mean=np.zeros(8, np.float32), std=np.ones(8, np.float32))
            store.split = lambda split: [r for r in records if r["split"] == split]
            store.load = lambda row: arrays[row["relative_path"]]
            store.clip_array = lambda rows: np.stack([store.load(r) for r in rows])[:, None]
            train, val = store.split("train"), store.split("validation")
            inputs = (torch.from_numpy(store.clip_array(train)), torch.tensor([r["id_class"] for r in train]),
                      torch.from_numpy(store.clip_array(val)), torch.tensor([r["id_class"] for r in val]))
            config = FinetuneConfig(output_dir=str(root / "complete"), machine_types=("fan",), seeds=(7,),
                epochs=2, warmup_epochs=1, patience=2, batch_size=4, evaluation_batch_size=4,
                device="cpu", mixed_precision=False)
            resumed_config = replace(config, output_dir=str(root / "resumed"))
            calls = 0
            def interrupt(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 3:
                    raise RuntimeError("simulated interruption")
                return epoch_pass(*args, **kwargs)
            with patch("mimii.finetune.build_mobilenet", side_effect=lambda classes, path: TinyTransfer(classes)), \
                    patch("mimii.finetune.weights_sha256", return_value="fixture"), \
                    patch("mimii.finetune.plot_history"), patch("mimii.evaluation.create_evaluation_figures"), \
                    contextlib.redirect_stdout(io.StringIO()):
                complete = train_seed(config, "fan", 7, store, torch.device("cpu"), inputs)
                with patch("mimii.finetune.epoch_pass", side_effect=interrupt), self.assertRaisesRegex(RuntimeError, "simulated"):
                    train_seed(resumed_config, "fan", 7, store, torch.device("cpu"), inputs)
                resumed = train_seed(resumed_config, "fan", 7, store, torch.device("cpu"), inputs)
                self.assertEqual(complete["cells"], resumed["cells"])
                self.assertEqual(complete["selected_epoch"], resumed["selected_epoch"])
                first = torch.load(root / "complete/fan/seed_7/last.pt", weights_only=False)
                second = torch.load(root / "resumed/fan/seed_7/last.pt", weights_only=False)
                for key in first["model_state"]:
                    torch.testing.assert_close(first["model_state"][key], second["model_state"][key], rtol=0, atol=0)
                repeated = train_seed(resumed_config, "fan", 7, store, torch.device("cpu"), inputs)
                self.assertEqual(repeated, resumed)


if __name__ == "__main__":
    unittest.main()
