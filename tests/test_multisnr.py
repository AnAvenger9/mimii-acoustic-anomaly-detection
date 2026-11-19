"""Prevent leakage through differently noised copies of a source recording."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest

from mimii.dataset import validate_splits
from mimii.prepare_multisnr import FRONTEND_KEYS, combine_records
from mimii.report_multisnr import statistics


class MultiSnrTests(unittest.TestCase):
    def test_seed_variability_is_population_variance_not_standard_error(self):
        result = statistics([0.6, 0.7, 0.8])
        self.assertAlmostEqual(result["mean"], 0.7)
        self.assertAlmostEqual(result["variance"], 0.02 / 3)
        self.assertAlmostEqual(result["std"] ** 2, result["variance"])
        self.assertEqual(result["n"], 3)

    def stores(self):
        stores = []
        for snr in (-6, 0, 6):
            config = {key: 1 for key in FRONTEND_KEYS}
            config["snr_db"] = snr
            row = {"relative_path": "id_00/normal/000.wav", "machine_id": "id_00", "label": 0,
                   "id_class": 0, "split": "train", "waveform_sha256": str(snr),
                   "snr_db": snr, "feature_path": "logmel/id_00/normal/000.npy"}
            stores.append(SimpleNamespace(config=config, records=[row], root=Path(f"features/{snr}")))
        return stores

    def test_versions_retain_group_identity_and_unique_paths(self):
        rows = combine_records(self.stores(), Path("features/combined"))
        self.assertEqual(len({r["relative_path"] for r in rows}), 3)
        self.assertEqual(len({r["source_recording_key"] for r in rows}), 1)
        rows[1]["split"] = "test"
        with self.assertRaisesRegex(ValueError, "Noise versions"):
            validate_splits(rows)

    def test_mismatched_splits_labels_or_frontends_are_rejected(self):
        for target, key, value in (("records", "split", "test"), ("records", "label", 1),
                                   ("config", "n_fft", 2048)):
            stores = deepcopy(self.stores())
            if target == "records":
                stores[1].records[0][key] = value
            else:
                stores[1].config[key] = value
            with self.assertRaises(ValueError):
                combine_records(stores, Path("features/combined"))
