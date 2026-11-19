# Comparing CNN and XGBoost ensembles

ResNet-18, EfficientNet-B0, the wide CNN and XGBoost together achieved 95.66% AUC, 90.55% standardized pAUC, 80.70% fault recall and a 7.42% false-alarm rate. At the healthy-training 95th-percentile threshold, this combination caught 397 more faulty clips and raised 66 fewer false alarms than the original five-CNN ensemble. Dropping the wide CNN cost four detections and added eight false alarms.

Six seed-42 predictors supplied 57 ensembles and six individual references. We standardized each log anomaly score with its machine type and ID's healthy training statistics, then averaged the selected model scores with equal weights. Each combination received its own threshold derived from fused healthy training scores. AUC and pAUC weight all 48 machine/ID/SNR groups equally. Recall and false-alarm rate weight recordings equally.

The four-member combination ranked first by pAUC on a group-preserving selection half (9,276 clips). It achieved 90.42% pAUC on the remaining 9,438 clips, compared with 88.48% for the original ensemble. Both halves come from the previously inspected project test split. Results describe a single seed.

- [Recompute the comparison from clip scores](../../../notebooks/04_ensemble_combinations.ipynb)
- [All 63 configurations](all_combinations.csv)
- [Results by machine](by_machine.csv)
- [Grouped selection and check](split_check.json)
