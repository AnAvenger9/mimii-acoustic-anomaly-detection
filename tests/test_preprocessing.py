"""Small synthetic fixtures; compatible with pytest and stdlib unittest."""

from dataclasses import replace
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from mimii.audio import load_audio
from mimii.config import PreprocessConfig
from mimii.dataset import assign_reference_splits, assign_splits, discover_recordings, parse_recording, validate_splits
from mimii.features import context_vectors, fit_normalizer, log_mel_spectrogram, normalize


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config = PreprocessConfig(duration_seconds=0.1, context_frames=3, workers=1)

    def tearDown(self):
        self.temporary.cleanup()

    def write_audio(self, data, sample_rate=16000, name="fixture.wav"):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, data, sample_rate, subtype="PCM_16")
        return path

    def test_first_microphone_preserves_amplitude(self):
        audio = np.column_stack((np.full(1600, 0.25), np.full(1600, -0.5)))
        path = self.write_audio(audio)
        first, meta = load_audio(path, self.config)
        second, _ = load_audio(path, replace(self.config, channel=1))
        np.testing.assert_allclose(first, 0.25)
        np.testing.assert_allclose(second, -0.5)
        self.assertEqual(meta["channels"], 2)
        self.assertAlmostEqual(meta["rms"], 0.25)

    def test_malformed_audio_is_rejected(self):
        for path, config in (
            (self.write_audio(np.ones(1600), sample_rate=8000, name="wrong_rate.wav"), self.config),
            (self.write_audio(np.ones(800), name="short.wav"), self.config),
            (self.write_audio(np.ones(1600), name="mono.wav"), replace(self.config, channel=1)),
        ):
            with self.assertRaises(ValueError):
                load_audio(path, config)

    def test_labels_and_id_parsing(self):
        for condition, label in (("normal", 0), ("abnormal", 1)):
            row = parse_recording(self.root / "id_02" / condition / "original.wav", self.root)
            self.assertEqual(row["machine_id"], "id_02")
            self.assertEqual(row["label"], label)
            self.assertEqual(row["filename"], "original.wav")
        for relative in ("id_2/normal/x.wav", "id_02/unknown/x.wav", "normal/x.wav"):
            with self.assertRaises(ValueError):
                parse_recording(self.root / relative, self.root)
        with self.assertRaises(ValueError):
            discover_recordings(self.root)

    def test_logmel_shape_and_silence_floor(self):
        config = PreprocessConfig()
        silent = log_mel_spectrogram(np.zeros(160000, dtype=np.float32), config)
        self.assertEqual(silent.shape, (64, 313))
        self.assertEqual(context_vectors(silent, 5).shape, (309, 320))
        self.assertTrue(np.isfinite(silent).all())
        np.testing.assert_allclose(silent, 10 * np.log10(np.finfo(np.float64).eps))

    def test_logmel_retains_absolute_energy(self):
        t = np.arange(1600) / 16000
        y = (0.2 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
        a = log_mel_spectrogram(y, self.config)
        b = log_mel_spectrogram(y * 0.5, self.config)
        active = a > -60
        np.testing.assert_allclose((a - b)[active], 20 * np.log10(2), atol=0.01)

    def test_context_order_and_validation(self):
        feature = np.array([[1, 2, 3], [10, 20, 30]], dtype=np.float32)
        np.testing.assert_array_equal(context_vectors(feature, 2), [[1, 10, 2, 20], [2, 20, 3, 30]])
        for frames in (0, 4):
            with self.assertRaises(ValueError):
                context_vectors(feature, frames)

    def synthetic_records(self):
        rows = []
        for mid in ("id_00", "id_02"):
            for label, count in ((0, 20), (1, 3)):
                for index in range(count):
                    path = f'{mid}/{"abnormal" if label else "normal"}/{index}.wav'
                    rows.append({"relative_path": path, "machine_id": mid, "label": label,
                                 "waveform_sha256": f"{mid}:{label}:{index}"})
        return rows

    def test_split_isolation_and_determinism(self):
        original = self.synthetic_records()
        duplicate = dict(original[0], relative_path="id_00/normal/duplicate.wav")
        original.append(duplicate)
        first = assign_splits([dict(r) for r in original], seed=42)
        second = assign_splits([dict(r) for r in reversed(original)], seed=42)
        mapping = lambda rows: {r["relative_path"]: r["split"] for r in rows}
        self.assertEqual(mapping(first), mapping(second))
        self.assertEqual(first[0]["split"], first[-1]["split"])
        self.assertTrue(all(r["split"] == "test" for r in first if r["label"] == 1))
        third = assign_splits([dict(r) for r in original], seed=99)
        self.assertNotEqual(mapping(first), mapping(third))
        validate_splits(first)

    def test_duplicate_label_conflict_rejected(self):
        rows = self.synthetic_records()
        rows[-1]["waveform_sha256"] = rows[0]["waveform_sha256"]
        with self.assertRaises(ValueError):
            assign_splits(rows)

    def test_reference_split_survives_changed_waveforms_at_other_snr(self):
        reference = assign_splits(self.synthetic_records())
        for row in reference:
            row.update(machine_type="fan", condition="abnormal" if row["label"] else "normal")
        target = [dict(row, waveform_sha256="new_snr:" + row["waveform_sha256"]) for row in reference]
        for row in target:
            row.pop("split")
        assign_reference_splits(target, reference)
        self.assertEqual([r["split"] for r in target], [r["split"] for r in reference])
        with self.assertRaises(ValueError):
            assign_reference_splits(target[:-1], reference)
        target[0]["label"] = 1
        with self.assertRaises(ValueError):
            assign_reference_splits(target, reference)

    def test_normalizer_ignores_validation_and_test(self):
        rows = []
        for index, (split, value, label) in enumerate((("train", 1., 0), ("train", 3., 0),
                                                      ("validation", 1000., 0), ("test", -1000., 1))):
            path = f"{index}.npy"
            np.save(self.root / path, np.full((2, 3), value, dtype=np.float32))
            rows.append({"split": split, "label": label, "feature_path": path})
        stats = fit_normalizer(rows, self.root)
        np.testing.assert_array_equal(stats["mean"], [2, 2])
        np.testing.assert_array_equal(stats["std"], [1, 1])
        self.assertEqual(stats["n_frames"], 6)
        self.assertEqual(stats["n_clips"], 2)
        np.testing.assert_array_equal(normalize(np.ones((2, 3)), stats["mean"], stats["std"]), -np.ones((2, 3)))

    def test_config_rejects_invalid_parameters(self):
        for kwargs in ({"n_fft": 0}, {"channel": -1}, {"fmax": 9000},
                       {"validation_fraction": 0.8, "test_normal_fraction": 0.3}):
            with self.assertRaises(ValueError):
                PreprocessConfig(**kwargs)

    def test_end_to_end_manifest_and_output_guard(self):
        from mimii.preprocess import run

        data_root = self.root / "fan"
        rng = np.random.default_rng(10)
        for mid in ("id_00", "id_02"):
            for index in range(6):
                condition = "abnormal" if index == 5 else "normal"
                self.write_audio(rng.normal(0, 0.1, (1600, 2)),
                                 name=f"fan/{mid}/{condition}/{index:08}.wav")
        output = self.root / "output"
        config = replace(self.config, data_root=str(data_root), output_dir=str(output))
        with patch("mimii.visualization.create_figures"), contextlib.redirect_stdout(io.StringIO()):
            summary = run(config)
        self.assertEqual(summary["recordings"], 12)
        self.assertEqual(summary["split_counts"], {"train": 6, "validation": 2, "test": 4})
        self.assertEqual(json.loads((output / "status.json").read_text())["state"], "complete")
        rows = [json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines()]
        validate_splits(rows)
        self.assertEqual(len(list((output / "logmel").rglob("*.npy"))), 12)
        self.assertTrue((output / "normalizer.npz").exists())
        with self.assertRaises(FileExistsError):
            run(config)


if __name__ == "__main__":
    unittest.main()
