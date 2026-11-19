# CNN ensemble and XGBoost fault detection

Both methods used the complete original MIMII study split and seed 42. Each machine type contributed four known IDs at three SNRs. The held-out evaluation covered 9,900 faulty clips and 8,814 healthy clips. Three SNR versions share each source recording.

The ensemble combines the five existing seed-42 CNNs (MobileNetV3-Small, ResNet-18, EfficientNet-B0 and both custom mean/std CNNs). For each member and machine ID, we log-transformed the anomaly scores and standardized them using its healthy training mean and standard deviation. The ensemble score is their unweighted mean. This project-specific fusion tests whether the CNNs provide complementary information.

For each machine type, XGBoost learned to predict the four machine IDs from 256 features: temporal means and standard deviations of 128 normalized log-mel bands. Training used only healthy clips, 100 boosting rounds, depth 4, and learning rate 0.1. The anomaly score was the negative log probability of the known ID.

For both methods we fixed the configuration before this run and used the per-ID 95th percentile of healthy training scores as the decision threshold across all SNRs. AUC and standardized partial AUC (maximum FPR 0.1) weight the 48 type/ID/SNR cells equally. Recall and false alarm rate count test clips. These single-seed follow-ups use a previously inspected test set.

| Method | Machine | AUC (%) | pAUC (%) | Fault recall (%) | Healthy false alarms (%) |
|---|---|---:|---:|---:|---:|
| cnn_ensemble | fan | 87.57 | 80.00 | 60.52 | 8.03 |
| cnn_ensemble | pump | 89.08 | 82.88 | 67.18 | 8.37 |
| cnn_ensemble | slider | 98.30 | 94.16 | 96.22 | 7.84 |
| cnn_ensemble | valve | 99.53 | 98.11 | 99.23 | 8.41 |
| cnn_ensemble | Overall | 93.62 | 88.79 | 76.69 | 8.17 |
| xgboost | fan | 86.04 | 79.89 | 63.01 | 6.35 |
| xgboost | pump | 91.57 | 82.82 | 68.79 | 9.48 |
| xgboost | slider | 93.06 | 85.20 | 86.52 | 6.06 |
| xgboost | valve | 93.81 | 81.89 | 78.36 | 8.91 |
| xgboost | Overall | 91.12 | 82.45 | 72.37 | 7.73 |

| Method | Faults caught | Faults missed | False alarms | Healthy accepted |
|---|---:|---:|---:|---:|
| cnn_ensemble | 7592 | 2308 | 720 | 8094 |
| xgboost | 7165 | 2735 | 681 | 8133 |

The independent check recomputed the thresholds, decisions, confusion counts, AUC and pAUC from the saved per-clip files and split manifests.

For the matched single-model reference, ResNet-18 at seed 42 achieved 90.47% AUC, 84.21% pAUC, 69.98% recall and 6.27% false alarms.

Reproduce from the project root using the existing feature cache and saved seed-42 CNN scores. Install the project training dependencies and xgboost-cpu==3.1.2, then run:

```bash
PYTHONPATH=src OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 python -m mimii.ensemble_xgb
```

Per-clip scores, fitted XGBoost models, CNN calibration values, score-file hashes, and per-cell metrics are under runs/ensemble_xgb_seed42/.
