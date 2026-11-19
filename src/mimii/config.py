"""Explicit, validated preprocessing and recording split configuration."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class PreprocessConfig:
    data_root: str = "data/fan_0db/fan"
    output_dir: str = "features/fan_0db_logmel_seed42"
    snr_db: int | None = 0
    seed: int = 42
    validation_fraction: float = 0.1
    test_normal_fraction: float = 0.2
    sample_rate: int = 16000
    duration_seconds: float = 10.0
    channel: int = 0
    n_fft: int = 1024
    hop_length: int = 512
    n_mels: int = 64
    fmin: float = 0.0
    fmax: float = 8000.0
    center: bool = True
    pad_mode: str = "reflect"
    context_frames: int = 5
    workers: int = 2
    split_reference_manifest: str | None = None

    def __post_init__(self):
        if not 0 < self.validation_fraction < 1 or not 0 < self.test_normal_fraction < 1:
            raise ValueError("Validation and test fractions must be between 0 and 1")
        if self.validation_fraction + self.test_normal_fraction >= 1:
            raise ValueError("Split fractions leave no training data")
        for key in ("sample_rate", "n_fft", "hop_length", "n_mels", "context_frames", "workers"):
            value = getattr(self, key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if not isinstance(self.channel, int) or self.channel < 0:
            raise ValueError("channel must be a nonnegative, zero-based index")
        if self.duration_seconds <= 0 or not 0 <= self.fmin < self.fmax <= self.sample_rate / 2:
            raise ValueError("Invalid duration or frequency limits")
        if self.pad_mode not in {"reflect", "constant"}:
            raise ValueError("pad_mode must be reflect or constant")
        if self.expected_frames < self.context_frames:
            raise ValueError("Audio is too short for the configured context")

    @property
    def expected_samples(self):
        return round(self.sample_rate * self.duration_seconds)

    @property
    def expected_frames(self):
        padding = 2 * (self.n_fft // 2) if self.center else 0
        return 1 + (self.expected_samples + padding - self.n_fft) // self.hop_length

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_json(cls, path):
        return cls(**json.loads(Path(path).read_text()))
