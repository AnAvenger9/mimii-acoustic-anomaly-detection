# Industrial machine fault detection from sound

Can a classifier trained to recognize a healthy machine detect when it sounds faulty? This study uses the original MIMII recordings of fans, pumps, sliders and valves. The original five-CNN ensemble achieved **93.62% ROC-AUC**. In a follow-up comparison of 57 ensembles, ResNet-18, EfficientNet-B0, the wide custom CNN and XGBoost together achieved **95.66% ROC-AUC**. They detected **7,989 of 9,900 faulty clips** and raised **654 false alarms on 8,814 healthy clips**.

**Start with the [results notebook](notebooks/01_reproduce_results.ipynb).** Its saved output is visible on GitHub. Run its cells to recompute the table on a CPU from the included training and test scores. The [model walkthrough](notebooks/02_model_architectures.ipynb) builds all five CNNs and measures their parameter counts. The [preprocessing walkthrough](notebooks/03_audio_to_logmel.ipynb) demonstrates the exact feature transformation on a generated audio example. The [notebook index](notebooks/README.md) organizes all implementation and test notebooks by topic. The [ensemble comparison](notebooks/04_ensemble_combinations.ipynb) recomputes the follow-up study from the saved scores. The [ensemble findings](notebooks/05_ensemble_findings.ipynb) show the results by machine and the source-grouped selection check.

| Model | Seeds | AUC (%) | Partial AUC (%) | Fault recall (%) | False alarms (%) |
|---|---:|---:|---:|---:|---:|
| MobileNetV3-Small | 3 | 82.68 | 74.89 | 62.66 | 10.96 |
| ResNet-18 | 1 | 90.47 | 84.21 | 69.98 | 6.27 |
| EfficientNet-B0 | 1 | 89.68 | 84.33 | 69.38 | 6.63 |
| Compact custom CNN | 3 | 88.40 | 81.43 | 64.51 | 6.75 |
| Wide custom CNN | 3 | 87.94 | 82.06 | 63.67 | 6.31 |
| Five-CNN ensemble | 1 | 93.62 | 88.79 | 76.69 | 8.17 |
| XGBoost | 1 | 91.12 | 82.45 | 72.37 | 7.73 |
| ResNet-18 + EfficientNet-B0 + XGBoost | 1 | 95.58 | 90.08 | 80.66 | 7.51 |
| ResNet-18 + EfficientNet-B0 + wide CNN + XGBoost | 1 | 95.66 | 90.55 | 80.70 | 7.42 |

The last two rows come from an exploratory search on the same test recordings. When selected by partial AUC on one source-grouped half, the four-member ensemble achieved 90.42% partial AUC on the other half. [All combinations and per-machine results](docs/results/ensemble_subsets_seed42/RESULTS.md)

## Reproduce the tables on a CPU

Run these commands from the repository root in a Python 3.10+ environment.

```bash
python -m pip install numpy scikit-learn jupyterlab ipykernel
jupyter lab notebooks
```

Open notebook 01 or 04 and choose **Run All Cells**. Notebook 01 reproduces the original results. Notebook 04 evaluates every combination. To retrain the models from audio, follow the [GPU reproduction guide](REPRODUCE.md).

## Experimental design

Each of the 18,019 source recordings has three noise mixtures (−6, 0 and +6 dB). We kept all versions of each source in the same partition. The split contains 30,948 healthy training clips, 4,395 healthy validation clips and 18,714 test clips. Each CNN learns four healthy machine IDs within one machine type. A low probability for the known ID raises the anomaly score. The ensemble averages five scores after standardizing their logarithms on healthy training data. XGBoost uses the temporal mean and standard deviation of each mel band.

We compute AUC and standardized partial AUC (maximum false-positive rate 0.1) in 48 machine type, ID and noise groups, then average. Recall and false-alarm rates count clips using each ID's 95th percentile of healthy training scores. For repeated fits the table averages across seeds. Experiments share a previously inspected test split. They measure binary fault detection on known machines.

## Explore the evidence

- [Rebuild the pipeline from the original audio, with GPU training](REPRODUCE.md)
- [Per-machine ensemble and XGBoost outcomes](docs/results/ensemble_xgb_seed42/RESULTS.md)
- [Compact CNN and pooling ablation](docs/results/acoustic_residual_stats_cnn/RESULTS.md)
- [Wide CNN capacity study](docs/results/large_acoustic_residual_stats_cnn/RESULTS.md)
- [DCASE 2020 published autoencoder reference](https://dcase.community/challenge2020/task-unsupervised-detection-of-anomalous-sounds). Its official development partition differs from the split used here.
- [Original MIMII dataset and citation](https://zenodo.org/records/3384388)
