"""Presentation-ready descriptive figures; examples are not model results."""

from collections import Counter
from pathlib import Path
import numpy as np


def create_figures(records, config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import librosa
    from .audio import load_audio

    output = Path(config.output_dir) / "figures"
    output.mkdir(exist_ok=True)
    ids = sorted({r["machine_id"] for r in records})
    counts = Counter((r["machine_id"], r["condition"]) for r in records)
    x = np.arange(len(ids))
    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    for offset, condition, color in ((-0.2, "normal", "#187f8c"), (0.2, "abnormal", "#d86445")):
        bars = ax.bar(x + offset, [counts[mid, condition] for mid in ids], width=0.4,
                      label=condition.title(), color=color)
        ax.bar_label(bars, padding=3)
    subtitle = f"{config.snr_db} dB SNR" if config.snr_db is not None else "official DCASE development data"
    ax.set(xticks=x, xticklabels=ids, ylabel="10-second recordings",
           title=f"{Path(config.data_root).name.title()} subset, {subtitle}")
    ax.set_ylim(top=max(counts.values()) * 1.15)
    ax.legend(frameon=False)
    fig.savefig(output / "dataset_counts.png", dpi=180)
    fig.savefig(output / "dataset_counts.svg")
    plt.close(fig)

    examples = [next(r for r in records if r["machine_id"] == mid and r["condition"] == condition)
                for mid in ids for condition in ("normal", "abnormal")]
    features = [np.load(Path(config.output_dir) / r["feature_path"], allow_pickle=False) for r in examples]
    vmin, vmax = np.percentile(np.concatenate([f.ravel() for f in features]), [2, 98])
    frequencies = librosa.mel_frequencies(n_mels=config.n_mels + 2, fmin=config.fmin, fmax=config.fmax)[1:-1]
    fig, axes = plt.subplots(len(ids), 2, figsize=(12, 2.4 * len(ids)), layout="constrained", squeeze=False)
    for ax, row, feature in zip(axes.ravel(), examples, features):
        mesh = ax.imshow(feature, origin="lower", aspect="auto", cmap="magma", vmin=vmin, vmax=vmax,
                         extent=(0, config.duration_seconds, -0.5, config.n_mels - 0.5))
        ticks = np.linspace(0, config.n_mels - 1, 5).astype(int)
        ax.set(yticks=ticks, yticklabels=[f"{frequencies[i]:.0f}" for i in ticks],
               xlabel="Time (s)", ylabel="Mel center (Hz)",
               title=f'{row["machine_id"]} / {row["condition"]} / {row["filename"]}')
    fig.colorbar(mesh, ax=axes.ravel().tolist(), label="Log-mel power (dB, common color scale)", shrink=0.7)
    fig.savefig(output / "logmel_examples.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 5), layout="constrained")
    for ax, row in zip(axes, examples[:2]):
        waveform, _ = load_audio(Path(config.data_root) / row["relative_path"], config)
        ax.plot(np.arange(0, waveform.size, 16) / config.sample_rate, waveform[::16], linewidth=0.35)
        ax.set(xlabel="Time (s)", ylabel="PCM amplitude",
               title=f'{row["machine_id"]} / {row["condition"]} — microphone {config.channel + 1}')
    limit = max(abs(bound) for ax in axes for bound in ax.get_ylim())
    for ax in axes:
        ax.set_ylim(-limit, limit)
    fig.savefig(output / "waveform_examples.png", dpi=160)
    plt.close(fig)

    # Standalone process diagram for a progress-update slide.
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(14, 6), layout="constrained")
    ax.set(xlim=(0, 14), ylim=(0, 6))
    ax.axis("off")
    def box(x, y, width, title, detail, color="#e8f4f4"):
        patch = FancyBboxPatch((x, y), width, 1.1, boxstyle="round,pad=0.06",
                               facecolor=color, edgecolor="#187f8c", linewidth=1.3)
        ax.add_patch(patch)
        ax.text(x + width / 2, y + 0.76, title, ha="center", va="center", fontsize=12, fontweight="bold")
        ax.text(x + width / 2, y + 0.34, detail, ha="center", va="center", fontsize=10)
    def arrow(start, end):
        ax.annotate("", xy=end, xytext=start,
                    arrowprops={"arrowstyle": "->", "color": "#39545b", "lw": 1.8})
    channels = "/".join(str(n) for n in sorted({r["channels"] for r in records}))
    box(0.2, 4.1, 2.8, "Original audio",
        f"{config.duration_seconds:g} s · {config.sample_rate / 1000:g} kHz · {channels} channels")
    box(3.7, 4.1, 2.8, f"Microphone {config.channel + 1}", "Validate · preserve amplitude")
    box(7.2, 4.1, 2.8, "Hann STFT", f"{config.n_fft:,} window · {config.hop_length} hop")
    box(10.7, 4.1, 2.8, "Log-mel features", f"{config.n_mels} bands × {config.expected_frames} time frames")
    for left, right in ((3, 3.7), (6.5, 7.2), (10, 10.7)):
        arrow((left + 0.08, 4.65), (right - 0.08, 4.65))
    train = 1 - config.validation_fraction - config.test_normal_fraction
    split_text = f"Normal: {train:.0%} / {config.validation_fraction:.0%} / {config.test_normal_fraction:.0%} · all faults: test only"
    if "official_partition" in records[0]:
        split_text = f"Official test fixed · {config.validation_fraction:.0%} of training normals for validation"
    box(1.0, 2.0, 5.0, "Recording-level split", split_text)
    box(7.4, 2.0, 5.8, "Training-only normalization", "Per-mel mean / std · same transform for every split")
    arrow((12.1, 4.03), (3.5, 3.18))
    arrow((6.08, 2.55), (7.32, 2.55))
    ax.text(7, 1.2, "Model-ready inputs", ha="center", fontsize=13, fontweight="bold", color="#187f8c")
    vectors = config.expected_frames - config.context_frames + 1
    ax.text(7, 0.65, f"Autoencoder: {vectors} × {config.n_mels * config.context_frames} context vectors per clip"
            f"     |     ID classifier: 1 × {config.n_mels} × {config.expected_frames}", ha="center", fontsize=12)
    ax.text(7, 0.15, "Split full recordings before creating overlapping context windows.", ha="center", fontsize=10)
    ax.set_title("Preprocessing pipeline", fontsize=20, fontweight="bold", pad=14)
    fig.savefig(output / "preprocessing_pipeline.png", dpi=180)
    fig.savefig(output / "preprocessing_pipeline.svg")
    plt.close(fig)
