"""Full-corpus cache validation and cross-machine/SNR split isolation."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from mimii.config import PreprocessConfig
from mimii.prepare_full import SNR_NAMES, audit_cache, run
from mimii.training_data import FeatureStore


class FullPreprocessingTests(unittest.TestCase):
    def test_full_pipeline_resume_and_corrupt_cache_detection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, output = root / "data", root / "output"
            rng = np.random.default_rng(17)
            for machine in ("fan", "valve"):
                for snr, name in SNR_NAMES.items():
                    for index in range(6):
                        condition = "abnormal" if index == 5 else "normal"
                        path = data / f"{machine}_{name}" / machine / "id_00" / condition / f"{index:08}.wav"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        sf.write(path, rng.normal(0, 0.1, (1600, 2)), 16000, subtype="PCM_16")
            config = {"data_dir": str(data), "output_dir": str(output),
                      "machine_types": ["fan", "valve"], "snr_levels_db": [0, -6, 6],
                      "frontend": {"duration_seconds": 0.1, "n_mels": 16, "context_frames": 3, "workers": 1}}
            with patch("mimii.visualization.create_figures"), contextlib.redirect_stdout(io.StringIO()):
                summary = run(config)
            self.assertEqual(summary["recordings"], 36)
            self.assertEqual(summary["unique_source_recordings"], 12)
            self.assertEqual(summary["split_counts"], {"train": 18, "validation": 6, "test": 12})
            self.assertEqual(summary["machine_ids"], {"fan/id_00": 0, "valve/id_00": 1})
            store = FeatureStore(output, "train_per_mel")
            self.assertEqual({r["id_class"] for r in store.records}, {0, 1})
            self.assertTrue(all(r["split"] == "test" for r in store.records if r["label"]))
            for row in store.records:
                self.assertTrue(Path(row["source_path"]).is_file())
                self.assertTrue(np.isfinite(store.load(row)).all())
            # Re-running verifies completed caches without re-extracting audio.
            with patch("mimii.prepare_full.extract_subset", side_effect=AssertionError("unexpected extraction")), \
                    contextlib.redirect_stdout(io.StringIO()):
                resumed = run(config)
            self.assertEqual(resumed["manifest_sha256"], summary["manifest_sha256"])
            with self.assertRaisesRegex(ValueError, "configuration changed"):
                run({**config, "frontend": {**config["frontend"], "seed": 99}})
            subset = output / "subsets" / "fan_0db"
            frontend = PreprocessConfig(**json.loads((subset / "config.json").read_text()))
            row = FeatureStore(subset).records[0]
            target = subset / row["feature_path"]
            np.save(target, np.full((16, frontend.expected_frames), np.nan, dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "Invalid feature array"):
                audit_cache(subset, frontend)


if __name__ == "__main__":
    unittest.main()
