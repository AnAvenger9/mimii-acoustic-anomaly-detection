# Larger CNN: audited binary fault detection

All 12 selected checkpoints were evaluated on 18,714 held-out recordings per seed: 9,900 abnormal and 8,814 normal. The three SNR versions of a source are correlated.
The independent audit passed for 56,142 test predictions and 92,844 training-normal scores across three seeds. Thresholds were recomputed from training normals; no test-label threshold tuning was performed.

| Metric | Mean ± population SD (%) |
|---|---:|
| Fault recall | 63.67 ± 0.41 |
| Missed faults | 36.33 ± 0.41 |
| False-positive rate | 6.31 ± 0.33 |
| Precision | 91.90 ± 0.36 |
| Binary accuracy | 77.81 ± 0.17 |
| F1 | 75.22 ± 0.24 |

| Seed | Faults detected | Faults missed | False alarms | Healthy accepted |
|---|---:|---:|---:|---:|
| 42 | 6348 | 3552 | 557 | 8257 |
| 43 | 6312 | 3588 | 591 | 8223 |
| 44 | 6250 | 3650 | 520 | 8294 |

## By machine

| Group | Recall (%) | False-positive rate (%) |
|---|---:|---:|
| fan | 46.15 ± 1.53 | 6.22 ± 0.90 |
| pump | 53.97 ± 3.49 | 6.33 ± 0.10 |
| slider | 90.42 ± 3.26 | 6.58 ± 0.87 |
| valve | 77.13 ± 5.91 | 6.14 ± 0.39 |

## By snr_db

| Group | Recall (%) | False-positive rate (%) |
|---|---:|---:|
| -6 | 67.95 ± 0.16 | 14.42 ± 0.78 |
| 0 | 60.76 ± 0.35 | 1.99 ± 0.24 |
| 6 | 62.30 ± 1.09 | 2.52 ± 0.03 |

Primary AUC and standardized pAUC use equal-weight machine/ID/SNR cells, unlike the clip-weighted rates above. See RESULTS.md for those results and matched-seed capacity comparisons.
Precision, F1 and accuracy depend on the test prevalence and are not estimates for factories where faults are rare. This is binary anomaly detection, not diagnosis of a specific fault mechanism. Seed SD is not a confidence interval.

Reproduce after running scripts/verify_mobilenet_results.py: `OPENBLAS_NUM_THREADS=1 /opt/pytorch/bin/python scripts/report_acoustic_fault_detection.py`
