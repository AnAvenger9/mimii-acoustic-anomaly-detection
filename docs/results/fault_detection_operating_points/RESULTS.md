# Fault detection at the deployed decision rule

Binary normal/anomalous predictions on 18,714 held-out recordings: 9,900 anomalous and 8,814 normal. These include three correlated SNR versions per source recording; they are not counts of independent physical faults.
Thresholds are the per-machine-type/ID 95th percentiles of training-normal anomaly scores, fixed across test SNRs. No fault labels fit thresholds.
Rates count every test clip equally within each run, then average across available initialization seeds. This differs from equal-ID/SNR macro averaging used for primary AUC/pAUC tables. Standard deviations are population SD across seeds, not confidence intervals.

| Model | Seeds | Fault recall (%) | Missed-fault rate (%) | False-positive rate (%) | Precision (%) | Binary accuracy (%) | F1 (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| MobileNetV3-Small | 3 | 62.66 ± 0.60 | 37.34 | 10.96 ± 0.33 | 86.53 ± 0.34 | 75.08 ± 0.33 | 72.68 ± 0.44 |
| resnet18 | 1 | 69.98 | 30.02 | 6.27 | 92.61 | 81.16 | 79.72 |
| efficientnet_b0 | 1 | 69.38 | 30.62 | 6.63 | 92.16 | 80.68 | 79.17 |
| Custom mean/std | 3 | 64.51 ± 1.43 | 35.49 | 6.75 ± 0.20 | 91.47 ± 0.19 | 78.04 ± 0.70 | 75.65 ± 0.99 |
| Custom mean-only | 1 | 65.01 | 34.99 | 6.57 | 91.75 | 78.40 | 76.10 |

## Exact counts by run

| Model | Seed | Faults caught | Faults missed | False alarms | Healthy correctly accepted |
|---|---:|---:|---:|---:|---:|
| MobileNetV3-Small | 42 | 6232 | 3668 | 1006 | 7808 |
| MobileNetV3-Small | 43 | 6257 | 3643 | 940 | 7874 |
| MobileNetV3-Small | 44 | 6121 | 3779 | 952 | 7862 |
| resnet18 | 42 | 6928 | 2972 | 553 | 8261 |
| efficientnet_b0 | 42 | 6869 | 3031 | 584 | 8230 |
| Custom mean/std | 42 | 6582 | 3318 | 614 | 8200 |
| Custom mean/std | 43 | 6325 | 3575 | 572 | 8242 |
| Custom mean/std | 44 | 6252 | 3648 | 600 | 8214 |
| Custom mean-only | 42 | 6436 | 3464 | 579 | 8235 |

## Custom mean/std CNN by machine type

Rates below average three seeds; every clip within a machine type has equal weight.

| Machine | Fault recall (%) | False-positive rate (%) |
|---|---:|---:|
| fan | 46.89 | 6.14 |
| pump | 57.46 | 7.42 |
| slider | 88.60 | 6.86 |
| valve | 80.70 | 6.66 |

Recall is the fraction of faulty recordings flagged; false-positive rate is the fraction of healthy recordings flagged. Precision is the fraction of alarms that correspond to anomalous test recordings. Precision, F1 and binary accuracy depend on this test-set mixture and should not be extrapolated to a factory with rare faults.
AUC and pAUC are also fault-detection metrics, but measure ranking across thresholds rather than the percentage of faults caught at the fixed operating threshold.
The project does not predict specific fault causes or severity. Its output is normal/anomalous; fault-type diagnosis was not trained or evaluated.
