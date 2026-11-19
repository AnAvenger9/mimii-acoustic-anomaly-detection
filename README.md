# Acoustic fault detection on MIMII

A faulty fan can still sound recognizably like the same fan. This project tests how often a change in that acoustic signature is enough to detect a fault. Five CNNs and an XGBoost model learned to identify individual machines from healthy recordings. At test time, a drop in confidence in the known machine ID contributed to the fault score.

| Model | ROC-AUC | Faults detected | Healthy clips flagged |
|---|---:|---:|---:|
| ResNet-18 | 90.47% | 69.98% | 6.27% |
| Five CNNs | 93.62% | 76.69% | 8.17% |
| ResNet-18, EfficientNet-B0, wide CNN and XGBoost | 95.66% | 80.70% | 7.42% |

The four-model ensemble caught **7,989 of 9,900 faulty clips** and flagged **654 of 8,814 healthy clips**. It ranked first in an exploratory comparison of 57 ensembles on the same test set. Its closest three-model alternative caught just four fewer faults. [Results by machine and ensemble](notebooks/05_ensemble_findings.ipynb)

## Read the study

[Read the six-page report](docs/report/README.md) · [Download the PDF](https://raw.githubusercontent.com/AnAvenger9/mimii-acoustic-anomaly-detection/main/docs/mimii_report.pdf)

1. [From audio to a spectrogram](notebooks/03_audio_to_logmel.ipynb). See how a ten-second recording becomes the input to a CNN.
2. [Model architectures](notebooks/02_model_architectures.ipynb). Inspect the three pretrained backbones and two custom networks.
3. [Reproduce the fault-detection results](notebooks/01_reproduce_results.ipynb). Recalculate AUC, partial AUC, recall and false alarms from the saved predictions.

The original MIMII data covers fans, pumps, sliders and valves. We kept the three noise mixtures of each recording together when splitting the data and set alarm thresholds using healthy training clips. [MIMII dataset](https://zenodo.org/records/3384388)

The notebooks display their saved outputs on GitHub. To run them locally or retrain the models on a GPU, see the [reproduction instructions](REPRODUCE.md).
