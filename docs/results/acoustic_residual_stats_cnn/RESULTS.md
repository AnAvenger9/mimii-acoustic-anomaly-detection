# Custom acoustic CNN results

A 177,724-parameter CNN trained from scratch on healthy machine IDs.
The main model uses temporal mean/std pooling and three seeds. The mean-only ablation uses seed 42; its std input is zeroed while the classifier shape is retained.

| Variant | Machine | AUC (%) | pAUC (%) |
|---|---|---:|---:|
| stats | fan | 79.50 ± 1.38 | 72.69 ± 1.87 |
| stats | pump | 85.80 ± 1.45 | 79.04 ± 0.58 |
| stats | slider | 94.00 ± 1.01 | 86.21 ± 1.85 |
| stats | valve | 94.29 ± 2.10 | 87.77 ± 1.99 |
| stats | Overall | 88.40 ± 0.89 | 81.43 ± 0.58 |
| mean_only | fan | 80.16 (seed 42) | 72.75 (seed 42) |
| mean_only | pump | 86.29 (seed 42) | 78.30 (seed 42) |
| mean_only | slider | 94.55 (seed 42) | 87.54 (seed 42) |
| mean_only | valve | 95.91 (seed 42) | 88.75 (seed 42) |
| mean_only | Overall | 89.23 (seed 42) | 81.84 (seed 42) |

## Paired pooling ablation

Seed 42: mean/std AUC/pAUC = 89.29/82.14%; mean-only = 89.23/81.84%.
Compare the matched seed above for the ablation, not the three-seed mean against a single run.

## Interpretation

Architecture combines established residual, depthwise/inverted-bottleneck and channel-attention components. No architectural novelty claim is made.
Mean/std pooling is a hypothesis about temporal variability, not a guarantee of improved fault sensitivity. The anomaly score remains known-ID negative log probability.
All recordings and frozen splits are retained. Checkpoints use healthy-validation cross-entropy; thresholds use training-normal quantiles. Test outcomes do not select configurations.
The architecture was proposed after earlier test results and separability diagnostics were inspected. This is exploratory work on a reused test set, not a fresh blind benchmark.
Comparisons with ImageNet-initialized models also differ in pretraining, parameter count and training schedules.
