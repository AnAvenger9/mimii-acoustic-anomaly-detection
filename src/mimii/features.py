"""Log-mel extraction, AE context vectors, and training-only normalization."""

import numpy as np
from .config import PreprocessConfig


def log_mel_spectrogram(waveform, config: PreprocessConfig):
    import librosa

    mel_power = librosa.feature.melspectrogram(
        y=waveform, sr=config.sample_rate, n_fft=config.n_fft,
        hop_length=config.hop_length, win_length=config.n_fft, window="hann",
        center=config.center, pad_mode=config.pad_mode, power=2.0,
        n_mels=config.n_mels, fmin=config.fmin, fmax=config.fmax,
        htk=False, norm="slaney",
    )
    # Explicit absolute power reference; no per-clip maximum or top-dB clipping.
    features = (10 * np.log10(mel_power.astype(np.float64) + np.finfo(np.float64).eps)).astype(np.float32)
    expected = (config.n_mels, config.expected_frames)
    if features.shape != expected or not np.isfinite(features).all():
        raise ValueError(f"Invalid log-mel output: expected {expected}, got {features.shape}")
    return features


def context_vectors(log_mel, frames=5):
    """Return (time - frames + 1, frames * mel), frame-major order."""
    if log_mel.ndim != 2 or not isinstance(frames, int) or frames < 1 or log_mel.shape[1] < frames:
        raise ValueError("Expected (mel, time) features with enough context frames")
    windows = np.lib.stride_tricks.sliding_window_view(log_mel, frames, axis=1)
    return np.ascontiguousarray(windows.transpose(1, 2, 0).reshape(-1, frames * log_mel.shape[0]))


def fit_normalizer(records, output_dir):
    """Accumulate stable per-band moments over training frames only."""
    from pathlib import Path

    count, mean, m2, clips = 0, None, None, 0
    for row in records:
        if row["split"] != "train":
            continue
        if row["label"] != 0:
            raise ValueError("Normalizer must be fit on normal training data only")
        x = np.load(Path(output_dir) / row["feature_path"], allow_pickle=False).astype(np.float64)
        batch_count = x.shape[1]
        batch_mean, batch_m2 = x.mean(axis=1), x.var(axis=1) * batch_count
        if count == 0:
            mean, m2 = batch_mean, batch_m2
        else:
            delta = batch_mean - mean
            m2 += batch_m2 + delta ** 2 * count * batch_count / (count + batch_count)
            mean += delta * batch_count / (count + batch_count)
        count += batch_count
        clips += 1
    if not count:
        raise ValueError("No normal training frames available")
    std = np.maximum(np.sqrt(m2 / count), 1e-6)
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32),
            "n_frames": np.int64(count), "n_clips": np.int64(clips)}


def normalize(log_mel, mean, std):
    return ((log_mel - np.asarray(mean)[:, None]) / np.asarray(std)[:, None]).astype(np.float32)
