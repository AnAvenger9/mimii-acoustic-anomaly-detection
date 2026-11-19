# Larger acoustic CNN results

Parameters: 1,286,964; scratch initialization; seeds [42, 43, 44].

| Machine | AUC (%) | pAUC (%) | Fault recall (%) | False alarms (%) |
|---|---:|---:|---:|---:|
| fan | 78.62 ± 0.50 | 73.30 ± 1.42 | 46.15 ± 1.53 | 6.22 ± 0.90 |
| pump | 86.85 ± 0.14 | 79.16 ± 0.86 | 53.97 ± 3.49 | 6.33 ± 0.10 |
| slider | 95.64 ± 0.63 | 88.92 ± 0.76 | 90.42 ± 3.26 | 6.58 ± 0.87 |
| valve | 90.65 ± 5.13 | 86.85 ± 3.03 | 77.13 ± 5.91 | 6.14 ± 0.39 |
| Overall | 87.94 ± 1.36 | 82.06 ± 0.65 | 63.67 ± 0.41 | 6.31 ± 0.33 |

Values are mean ± population SD across seeds. AUC/pAUC use equal-weight type/ID/SNR cells; recall/false alarms count individual clips. Standardized pAUC uses max FPR=0.1.

This detects abnormal sound, not a specific fault mechanism. Machine-ID accuracy is a separate auxiliary task.
Checkpoints minimize healthy-validation cross-entropy; decisions use per-ID training-normal thresholds. No abnormal clips are used for training or threshold selection.

## Matched small-to-large comparison

| Seed | AUC change (percentage points) | pAUC change (percentage points) |
|---|---:|---:|
| 42 | -0.19 | +0.27 |
| 43 | -1.15 | +0.42 |
| 44 | -0.05 | +1.20 |

Exploratory width/head-capacity comparison with identical data, depth, pooling and training settings. Test data were inspected in earlier experiments; no blind benchmark claim.
Three seeds describe initialization variability, not independent recordings or deployment uncertainty.
