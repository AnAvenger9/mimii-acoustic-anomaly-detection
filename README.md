# Acoustic fault detection on MIMII

Can a microphone flag a fault in a known machine using only examples of its healthy operation for training? On the original MIMII corpus, our four-model ensemble caught **7,989 of 9,900 faulty clips** and raised **654 false alarms on 8,814 healthy clips**. All evaluation clips are ten seconds long.

[Read the six-page IEEE report](docs/report/README.md) · [Download the report](https://raw.githubusercontent.com/AnAvenger9/mimii-acoustic-anomaly-detection/main/docs/mimii_report.pdf) ·

## Download trained models

The [trained-model release](https://github.com/AnAvenger9/mimii-acoustic-anomaly-detection/releases/tag/models-v1) contains the selected weights for all four machine types. To score a recording, download the final ensemble and open the inference notebook.

| Download | Contents |
|---|---|
| [Final ensemble](https://github.com/AnAvenger9/mimii-acoustic-anomaly-detection/releases/download/models-v1/final-ensemble.zip) | ResNet-18, EfficientNet-B0, wide acoustic CNN and XGBoost at seed 42, with per-type normalization and per-ID fusion cutoffs. |
| [Other reported fits](https://github.com/AnAvenger9/mimii-acoustic-anomaly-detection/releases/download/models-v1/other-reported-fits.zip) | MobileNet and compact CNN seeds 42–44, wide CNN seeds 43–44 and the compact mean-only ablation at seed 42. |

[Score a WAV](notebooks/05_score_recording.ipynb) · [Setup and weight formats](REPRODUCE.md#run-the-trained-ensemble)

## Why these experiments?

1. **Keep noise versions of the same sound together.** The 54,057 clips come from 18,019 recordings of fans, pumps, sliders and valves, each mixed at −6, 0 and +6 dB. Within each ID, split healthy source recordings into training, validation and test. Reserve every fault for test. Use the first microphone channel, a 1,024-sample Hann window and a 512-sample hop to produce 128 × 313 log-mel features. Fit each machine type’s per-band normalization on healthy training frames. [Preprocessing notebook](notebooks/03_audio_to_logmel.ipynb)
2. **Learn a machine’s identity from healthy audio.** Each type has four known units. Train a four-way ID classifier per type and retain the checkpoint with the lowest healthy-validation cross-entropy. At inference the monitored ID is known, so a fall in its probability raises the anomaly score, −log P(known ID | clip). ResNet-18 identified 4,393 of 4,395 healthy validation clips correctly, yet its alarm caught only 69.98% of faulty clips. This is why the main comparison uses fault rankings and alarm counts. [ID and fault results](notebooks/01_reproduce_results.ipynb)
3. **Test what the CNN needs to represent.** Start with the lightweight ImageNet-pretrained MobileNetV3-Small. Compare residual learning (ResNet-18) with inverted bottlenecks (EfficientNet-B0) on the same mono spectrograms. Healthy-validation loss selected a backbone learning rate of 0.0003 for both. Train a six-block acoustic CNN from scratch, ablate its temporal standard-deviation pooling, then widen it at fixed depth. The roughly sevenfold parameter increase moved seed-42 pAUC from 82.14% to 82.40%. [Architectures and parameter counts](notebooks/02_model_architectures.ipynb)
4. **Combine detectors with different errors.** ResNet ranked fan faults better than EfficientNet, while EfficientNet ranked valve faults better. First combine all five CNNs. Then test a different representation. XGBoost learns the same healthy-ID task from the temporal mean and standard deviation of each mel band. For each model and ID, standardize the log anomaly score against healthy training scores. Average the selected models equally and alarm above their combined healthy-training 95th percentile. [Ensemble experiment](notebooks/04_ensemble_combinations.ipynb)

## What changed at the alarm threshold?

All rows use training seed 42 on the same 18,714 test clips. AUC and standardized pAUC average 48 machine type/ID/noise groups, with pAUC restricted to false-positive rates up to 10%. Accuracy, recall and false-alarm rate count individual clips.

| Detector | AUC | pAUC | Fault accuracy | Fault recall | False alarms on healthy clips |
|---|---:|---:|---:|---:|---:|
| ResNet-18 | 90.47% | 84.21% | 81.16% | 69.98% | 6.27% |
| XGBoost | 91.12% | 82.45% | 81.75% | 72.37% | 7.73% |
| Five CNNs | 93.62% | 88.79% | 83.82% | 76.69% | 8.17% |
| ResNet-18 + EfficientNet-B0 + XGBoost | 95.58% | 90.08% | 86.23% | 80.66% | 7.51% |
| **Those three + wide acoustic CNN** | **95.66%** | **90.55%** | **86.29%** | **80.70%** | **7.42%** |

The four-member ensemble caught 397 more faulty clips and raised 66 fewer false alarms than the five-CNN ensemble. Adding the wide CNN to the three-member model accounted for just four extra detections and eight fewer false alarms. The 57 ensembles were compared on a previously inspected test set. A retrospective source-grouped check gave the selected ensemble 90.42% pAUC on the second half. Performance on new machines and recording sessions remains open. The DCASE 2020 autoencoder in the report is a published reference using the organizers’ split. [All combinations and grouped check](docs/results/ensemble_subsets_seed42/RESULTS.md)

## Reproduce the report tables

The CPU path uses the bundled scores and audited healthy-ID counts. Start with Python 3.10+ and run from the repository root. These two notebooks need no audio download, weights or GPU.

```bash
git clone https://github.com/AnAvenger9/mimii-acoustic-anomaly-detection.git
cd mimii-acoustic-anomaly-detection
python -m venv .venv
source .venv/bin/activate
python -m pip install "numpy>=1.26,<3" "scikit-learn>=1.4,<2" nbconvert ipykernel
mkdir -p runs/reproduced_notebooks
python -m nbconvert --to notebook --execute --output-dir=runs/reproduced_notebooks notebooks/01_reproduce_results.ipynb notebooks/04_ensemble_combinations.ipynb
```

To inspect each output interactively, install JupyterLab in the same environment and choose **Run All Cells**. The committed notebooks also show the executed tables directly on GitHub.

| Report | Where to check it |
|---|---|
| I · Split inventory | Notebook 01 counts training/test labels and reads the validation denominators. |
| II · CNN parameters and layers | Notebook 02 constructs all five networks and runs a forward pass. This step additionally needs PyTorch and torchvision. |
| III · Healthy machine-ID accuracy | Notebook 01 pools the exact correct/total counts from the checkpoint replay and LDA diagnostic. The CNN rows use validation clips and LDA uses healthy test clips. |
| IV–V · Rankings and alarm decisions | Notebook 01 computes individual models and the original five-CNN ensemble from scores. Notebook 04 computes the other ensembles. Seeds 42–44 appear separately as a stability check for MobileNet and the compact and wide CNNs. |
| VI · Results by machine | Notebook 04 breaks down the selected ensemble by machine type and writes all 63 configurations, per-machine results and the grouped check to `runs/ensemble_subsets_seed42/`. |

For a full run from the original WAVs, including archive verification, feature extraction, ImageNet initialization and GPU training, follow the [training and reproduction guide](REPRODUCE.md). [Dataset source](https://zenodo.org/records/3384388)
