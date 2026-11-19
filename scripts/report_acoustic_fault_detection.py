"""Recompute operating-point results from audited acoustic CNN test predictions."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/mimii-matplotlib')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
from mimii.training_data import FeatureStore


def counts_and_rates(rows):
    counts = {key: 0 for key in ('tp', 'fn', 'fp', 'tn')}
    for row in rows:
        label, predicted = int(row['label']), int(row['prediction'])
        counts[('tp' if predicted else 'fn') if label else ('fp' if predicted else 'tn')] += 1
    tp, fn, fp, tn = (counts[k] for k in ('tp', 'fn', 'fp', 'tn'))
    return {**counts, 'recall': tp/(tp+fn), 'missed_fault_rate': fn/(tp+fn),
            'fpr': fp/(fp+tn), 'precision': tp/(tp+fp) if tp+fp else 0.,
            'accuracy': (tp+tn)/(tp+fn+fp+tn), 'f1': 2*tp/(2*tp+fp+fn)}


def stats(rows):
    return {key: {'mean': float(np.mean([r[key] for r in rows])),
                  'sd': float(np.std([r[key] for r in rows]))}
            for key in ('recall', 'missed_fault_rate', 'fpr', 'precision', 'accuracy', 'f1')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', default='runs/large_acoustic_residual_stats_cnn/stats')
    parser.add_argument('--output-dir', default='docs/results/large_acoustic_residual_stats_cnn')
    args = parser.parse_args()
    root, dest = Path(args.run_dir), Path(args.output_dir)
    audit = json.loads((root/'validation.json').read_text())
    if audit['state'] != 'passed':
        raise ValueError('Run the independent saved-score audit before reporting')
    protocol = json.loads((root/'protocol.json').read_text())
    summary = json.loads((root/'summary.json').read_text())
    cfg = protocol['config']
    rows, hashes = [], {}
    for machine in cfg['machine_types']:
        store = FeatureStore(Path(cfg['feature_root'])/'by_machine'/machine)
        lookup = {r['relative_path']: r for r in store.split('test')}
        for seed in cfg['seeds']:
            path = root/machine/f'seed_{seed}'/'scores.csv'
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            with path.open() as handle:
                scores = list(csv.DictReader(handle))
            assert len(scores) == len(lookup) and {r['relative_path'] for r in scores} == set(lookup)
            for row in scores:
                assert int(row['label']) == lookup[row['relative_path']]['label']
                assert int(row['prediction']) == int(float(row['score']) > float(row['threshold']))
                rows.append({**row, 'machine': machine, 'seed': seed, 'snr_db': lookup[row['relative_path']]['snr_db']})
    per_seed = [{'seed': seed, **counts_and_rates([r for r in rows if r['seed']==seed])} for seed in cfg['seeds']]
    for row in per_seed:
        assert row['tp']+row['fn']==9900 and row['fp']+row['tn']==8814
        expected = next(r['operating_point'] for r in summary['per_seed'] if r['seed']==row['seed'])
        assert all(row[k]==expected[k] for k in ('tp','fn','fp','tn'))
    result = {'per_seed': per_seed, 'mean_sd': stats(per_seed), 'groups': {}, 'source_sha256': hashes,
              'audit_sha256': hashlib.sha256((root/'validation.json').read_bytes()).hexdigest(),
              'aggregation': 'Clip-weighted rates within each seed, mean and population SD across seeds',
              'threshold': 'Per-type/ID training-normal 95th percentile, fixed across test SNRs'}
    for key, values in [('machine',cfg['machine_types']), ('snr_db',[-6,0,6])]:
        result['groups'][key] = {str(value): stats([counts_and_rates([r for r in rows if r[key]==value and r['seed']==seed])
                                                  for seed in cfg['seeds']]) for value in values}
    dest.mkdir(parents=True, exist_ok=True)
    (dest/'fault_detection.json').write_text(json.dumps(result, indent=2)+'\n')
    fmt = lambda value: f"{100*value['mean']:.2f} ± {100*value['sd']:.2f}"
    lines = ['# Larger CNN: audited binary fault detection', '',
             'All 12 selected checkpoints were evaluated on 18,714 held-out recordings per seed: 9,900 abnormal and 8,814 normal. The three SNR versions of a source are correlated.',
             'The independent audit passed for 56,142 test predictions and 92,844 training-normal scores across three seeds. Thresholds were recomputed from training normals; no test-label threshold tuning was performed.', '',
             '| Metric | Mean ± population SD (%) |', '|---|---:|']
    for key, label in [('recall','Fault recall'),('missed_fault_rate','Missed faults'),('fpr','False-positive rate'),('precision','Precision'),('accuracy','Binary accuracy'),('f1','F1')]:
        lines.append(f"| {label} | {fmt(result['mean_sd'][key])} |")
    lines += ['', '| Seed | Faults detected | Faults missed | False alarms | Healthy accepted |', '|---|---:|---:|---:|---:|']
    for row in per_seed:
        lines.append(f"| {row['seed']} | {row['tp']} | {row['fn']} | {row['fp']} | {row['tn']} |")
    for group, values in result['groups'].items():
        lines += ['', f'## By {group}', '', '| Group | Recall (%) | False-positive rate (%) |', '|---|---:|---:|']
        for name, value in values.items():
            lines.append(f"| {name} | {fmt(value['recall'])} | {fmt(value['fpr'])} |")
    lines += ['', 'Primary AUC and standardized pAUC use equal-weight machine/ID/SNR cells, unlike the clip-weighted rates above. See RESULTS.md for those results and matched-seed capacity comparisons.',
              'Precision, F1 and accuracy depend on the test prevalence and are not estimates for factories where faults are rare. This is binary anomaly detection, not diagnosis of a specific fault mechanism. Seed SD is not a confidence interval.', '',
              'Reproduce after running scripts/verify_mobilenet_results.py: `OPENBLAS_NUM_THREADS=1 /opt/pytorch/bin/python scripts/report_acoustic_fault_detection.py`']
    (dest/'FAULT_DETECTION.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'per_seed':per_seed,'mean_sd':result['mean_sd']},indent=2))


if __name__ == '__main__':
    main()
