"""Verify feature provenance and load recording-isolated model inputs."""

import hashlib
import json
from pathlib import Path

import numpy as np

from .dataset import validate_splits
from .features import context_vectors, normalize


class FeatureStore:
    def __init__(self, root, normalization="none"):
        self.root = Path(root)
        status = json.loads((self.root / "status.json").read_text())
        if status["state"] != "complete":
            raise ValueError("Feature extraction is incomplete")
        manifest = (self.root / "manifest.jsonl").read_bytes()
        self.manifest_sha256 = hashlib.sha256(manifest).hexdigest()
        if self.manifest_sha256 != status["manifest_sha256"]:
            raise ValueError("Feature manifest changed after preprocessing")
        self.records = [json.loads(line) for line in manifest.decode().splitlines()]
        validate_splits(self.records)
        self.config = json.loads((self.root / "config.json").read_text())
        self.summary = json.loads((self.root / "summary.json").read_text())
        self.shape = (self.config["n_mels"], self.records[0]["feature_shape"][1])
        self.context_frames = self.config["context_frames"]
        self.normalization = normalization
        self.mean, self.std = None, None
        if normalization == "train_per_mel":
            provenance = json.loads((self.root / "normalizer_provenance.json").read_text())
            if provenance["manifest_sha256"] != self.manifest_sha256 or provenance["fit_split"] != "train":
                raise ValueError("Normalizer provenance does not match the feature manifest")
            with np.load(self.root / "normalizer.npz", allow_pickle=False) as stats:
                self.mean, self.std = stats["mean"].copy(), stats["std"].copy()
        elif normalization != "none":
            raise ValueError("Unknown normalization")

    def split(self, name):
        rows = [r for r in self.records if r["split"] == name]
        if not rows:
            raise ValueError(f"No {name} recordings")
        return rows

    def load(self, row):
        array = np.load(self.root / row["feature_path"], allow_pickle=False)
        if array.shape != self.shape or not np.isfinite(array).all():
            raise ValueError(f"Invalid feature array: {row['feature_path']}")
        return normalize(array, self.mean, self.std) if self.mean is not None else array

    def clip_array(self, rows):
        result = np.empty((len(rows), 1, *self.shape), dtype=np.float32)
        for index, row in enumerate(rows):
            result[index, 0] = self.load(row)
        return result

    def vector_array(self, rows):
        vectors_per_clip = self.shape[1] - self.context_frames + 1
        result = np.empty((len(rows) * vectors_per_clip, self.shape[0] * self.context_frames), dtype=np.float32)
        for index, row in enumerate(rows):
            result[index * vectors_per_clip:(index + 1) * vectors_per_clip] = context_vectors(self.load(row), self.context_frames)
        return result
