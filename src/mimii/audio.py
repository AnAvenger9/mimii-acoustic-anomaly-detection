"""Load one microphone without changing amplitude, sample rate or duration."""

from pathlib import Path
import hashlib
import numpy as np
import soundfile as sf

from .config import PreprocessConfig


def load_audio(path: Path, config: PreprocessConfig):
    with sf.SoundFile(path) as audio:
        if audio.samplerate != config.sample_rate:
            raise ValueError(f"{path}: sample rate {audio.samplerate}, expected {config.sample_rate}")
        if audio.frames != config.expected_samples:
            raise ValueError(f"{path}: {audio.frames} samples, expected {config.expected_samples}")
        if config.channel >= audio.channels:
            raise ValueError(f"{path}: channel {config.channel} missing from {audio.channels} channels")
        metadata = {"sample_rate": audio.samplerate, "channels": audio.channels,
                    "samples": audio.frames, "subtype": audio.subtype,
                    "duration_seconds": audio.frames / audio.samplerate}
        waveform = np.ascontiguousarray(audio.read(dtype="float32", always_2d=True)[:, config.channel])
    if not np.isfinite(waveform).all():
        raise ValueError(f"{path}: non-finite audio samples")
    metadata.update({
        "waveform_sha256": hashlib.sha256(waveform.astype("<f4", copy=False).tobytes()).hexdigest(),
        "rms": float(np.sqrt(np.mean(waveform.astype(np.float64) ** 2))),
        "peak": float(np.max(np.abs(waveform))),
        "dc_offset": float(np.mean(waveform, dtype=np.float64)),
        "clipped_fraction": float(np.mean(np.abs(waveform) >= 1 - 1 / 32768)),
        "silent": bool(np.max(np.abs(waveform)) == 0),
    })
    return waveform, metadata
