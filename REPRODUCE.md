# Reproduce the experiments

Run from the repository root on Linux. For the CNN training scripts, use a CUDA-enabled PyTorch installation with BF16 support and its matching torchvision build. The recorded environment used Python 3.13, torch 2.14.1+cu130, torchvision 0.29.1+cu130, NumPy 2.3.5, librosa 1.0.0, scikit-learn 1.9.1 and XGBoost 3.1.2. Start with a fresh environment and select the appropriate [PyTorch installation](https://pytorch.org/get-started/locally/) for your GPU driver.

```bash
python -m pip install jupyterlab ipykernel
jupyter nbconvert --to notebook --execute --stdout notebooks/00_runtime.ipynb > /dev/null
python -m pip install -e '.[dev,transfer,trees]' jupyterlab ipykernel
python -m pytest
```

The setup notebook builds local importable modules, tests and training entry points from tagged notebook cells. Run it again after editing the implementation notebooks.

## 1. Download MIMII and build the features

Allow about 160 GiB of free disk space for audio, temporary archives, features and training outputs. The download script verifies the original Zenodo archives against their MD5 values. You can also place the official ZIPs in data/ before running it. Preserve the split config and filenames. Terms of use and attribution are on the [dataset record](https://zenodo.org/records/3384388).

```bash
mkdir -p data/metadata checkpoints runs
cp configs/mimii_release.json data/metadata/zenodo-3384388.json
python scripts/download_mimii.py
python -m mimii.prepare_full --config configs/full_dataset_logmel.json
python scripts/verify_full_preprocessing.py
```

The pipeline selects microphone 1 from each 16 kHz, 10-second recording and converts it to a 128 × 313 log-mel array (1,024-sample Hann window, 512-sample hop). Normalization uses healthy training data per machine type. Check that the final inventory contains 54,057 clips and 18,019 source groups.

## 2. Train the five CNNs

Fetch the three official torchvision initializations. The two custom CNNs start from random weights.

```bash
for weight in mobilenet_v3_small-047dcff4.pth resnet18-f37072fd.pth efficientnet_b0_rwightman-7f5810bc.pth; do
  curl -fL --retry 3 "https://download.pytorch.org/models/$weight" -o "checkpoints/$weight" || exit 1
done
python scripts/train_mobilenet.py
python scripts/verify_mobilenet_results.py
python scripts/train_cnn_optimization.py
python scripts/extend_cnn_study.py --target-epochs 12 --minutes 240
python scripts/train_acoustic_cnn.py
python scripts/train_large_acoustic_cnn.py
```

Run sequentially. ResNet-18 and EfficientNet-B0 use validation-based learning-rate selection, followed by continuation from the selected six-epoch fits. Check that the parent study's status is complete and its final plan reached epoch 6 before extending it. The custom models and MobileNet use the minimum healthy-validation loss for checkpoint selection. Wall-clock guards in the study configs can be increased **before** launching on slower hardware. Published run-specific hashes record the original artifacts. A fresh extraction can have different filesystem timestamps, and exact numeric results depend on library versions and hardware.

## 3. Fuse the scores and train the tree baseline

```bash
python -m mimii.ensemble_xgb
```

The seed-42 ensemble reads saved scores from the five trained networks. The published transfer models both selected learning-rate trial 1. XGBoost learns the same healthy-ID task using 256 spectral statistics. Final metrics and clip predictions appear under `runs/ensemble_xgb_seed42/`.

## 4. Review the saved evidence

The CPU results notebook reads `results/fits.json` and `results/clip_scores.npz`. The compressed arrays contain relative clip paths, machine types, IDs, SNRs, binary labels and one float64 anomaly score for every training and test clip per fit. Their ordering is shared across models. This supports independent threshold calibration and metric calculations without shipping the audio or GPU checkpoints. The separate `docs/results/` directory contains cell-level summaries and audit results.

## 5. Reproduce the follow-up ensemble study

Run [notebook 04](notebooks/04_ensemble_combinations.ipynb) on a CPU with NumPy and scikit-learn. It recomputes every seed-42 subset, verifies the original ensemble scores, and writes three files under `runs/ensemble_subsets_seed42/`. Compare these with [the published outputs](docs/results/ensemble_subsets_seed42/RESULTS.md). [Notebook 05](notebooks/05_ensemble_findings.ipynb) displays the published per-machine and grouped-check results.
