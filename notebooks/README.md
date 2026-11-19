# Study notebooks

| Notebook | What to look for |
|---|---|
| [Audio preprocessing](03_audio_to_logmel.ipynb) | A 440 Hz tone shows where a sound appears in the 128-band log-mel input. |
| [CNN architectures](02_model_architectures.ipynb) | Layer definitions, parameter counts and an example forward pass. |
| [Fault detection](01_reproduce_results.ipynb) | Per-clip predictions become alarm thresholds and detection rates. |
| [Ensemble findings](05_ensemble_findings.ipynb) | How many faults each combination caught and which machines were harder to diagnose. |
| [Ensemble comparison](04_ensemble_combinations.ipynb) | Recalculate all 57 combinations and check the selected model on a separate group of recordings. |

For the underlying implementation, see the [split rules](code/preprocessing/dataset.ipynb), [compact CNN](code/architectures/acoustic_cnn.ipynb), [training loop](code/training/finetune.ipynb) and [anomaly metrics](code/evaluation/evaluation.ipynb). The [validation suite](code/tests/validation_suite.ipynb) groups the automated checks by experiment stage.

The [reproduction guide](../REPRODUCE.md) covers installation, data download and GPU training.
