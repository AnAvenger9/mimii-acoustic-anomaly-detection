"""Continuation preserves parent files and rejects changed provenance."""

import hashlib
import importlib.util
from importlib.metadata import version
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/extend_cnn_study.py"
spec = importlib.util.spec_from_file_location("extend_cnn_study", SCRIPT)
continuation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(continuation)


class ContinuationTests(unittest.TestCase):
    def test_copy_preserves_parent_and_resumes_with_original_deadline(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent, output, features = root / "parent", root / "extended", root / "features"
            parent.mkdir()
            normalizer = features / "by_machine/fan/normalizer.npz"
            normalizer.parent.mkdir(parents=True)
            normalizer.write_bytes(b"normalization fixture")
            (features / "manifest.jsonl").write_bytes(b"manifest fixture")
            study = {"feature_root": str(features), "output_dir": str(parent), "architectures": ["resnet18"],
                     "machine_types": ["fan"], "final_epochs": 6, "seed": 42}
            identity = {"study": study, "source_sha256": {},
                        "versions": {name: version(name) for name in ("torch", "torchvision", "numpy", "scikit-learn")},
                        "pretrained_sha256": {"resnet18": "pretrained fixture"},
                        "master_manifest_sha256": continuation.digest(features / "manifest.jsonl"),
                        "normalizer_sha256": {"fan": continuation.digest(normalizer)}}
            for name, value in (("protocol.json", {"identity": identity}), ("status.json", {"state": "complete"}),
                                ("validation.json", {"state": "passed"}), ("final_plan.json", {"target_epochs": 6}),
                                ("selection.json", {"resnet18": {"selected_trial": 1}}), ("screening.json", []), ("summary.json", {})):
                (parent / name).write_text(json.dumps(value))
            for index in (0, 1):
                directory = parent / f"resnet18/lr_{index}/fan"
                directory.mkdir(parents=True)
                (directory / "history.json").write_text("[]\n")
            directory = parent / "resnet18/lr_1/fan"
            for name in ("last.pt", "model.pt", "fit.json"):
                (directory / name).write_bytes(b"parent checkpoint fixture")
            with patch.object(continuation, "pretrained_hash", return_value="pretrained fixture"):
                first = continuation.prepare(parent, output, 12, 75)
                second = continuation.prepare(parent, output, 12, 75)
                self.assertEqual(first[2]["deadline_unix"], second[2]["deadline_unix"])
                self.assertEqual(first[0]["final_epochs"], 12)
                self.assertEqual(first[2]["deadline_unix"] - first[2]["started_unix"], 4500)
                copied = output / "resnet18/lr_1/fan/last.pt"
                self.assertNotEqual(copied.stat().st_ino, (directory / "last.pt").stat().st_ino)
                copied.write_bytes(b"continued checkpoint")
                self.assertEqual((directory / "last.pt").read_bytes(), b"parent checkpoint fixture")
                continuation.prepare(parent, output, 12, 75)
                self.assertEqual(copied.read_bytes(), b"continued checkpoint")
                with self.assertRaisesRegex(ValueError, "settings changed"):
                    continuation.prepare(parent, output, 13, 75)
                with self.assertRaisesRegex(ValueError, "extend the parent"):
                    continuation.prepare(parent, root / "invalid", 6, 75)
