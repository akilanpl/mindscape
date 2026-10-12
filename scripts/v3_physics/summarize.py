"""Post-run descriptive error, calibration and resource summaries; no neural calls."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from mindscape.v3.physics_world.data import load
from mindscape.v3.physics_world.storage import atomic_json


def summarize(dataset, root, output):
    manifest = load(dataset)
    approved = {r['identity'] for r in manifest['records'] if r['scene']['split'] == 'train'}
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    groups, strata = defaultdict(list), defaultdict(list)
    worst = []
    batches = {}
    neural_steps = 0
    for path in root.rglob('*.json'):
        if len(path.stem) != 64:
            continue
        d = json.loads(path.read_text())
        t, identity = d['trace'], d['identity']
        if 'metrics' not in t:
            continue
        assert not t['private_future_to_predictor']
        if t['device'] == 'mps':
            key_batch = (str(path.parent), t['horizon'], t['shared_batch_prediction_seconds'])
            batches[key_batch] = t['shared_batch_prediction_seconds']
            neural_steps += t['horizon']
        key = (identity['family'], identity.get('budget', 0), identity['variant'], t['split'])
        groups[key].append(t['metrics'])
        if identity.get('budget') == 250 and t['split'] == 'ood':
            strata[(identity['family'], identity['variant'], t['ood_condition'])].append(t['metrics']['normalized_rollout_mse'])
            if identity['variant'] == 'mindscape':
                worst.append({'scene': t['scene_id'], 'family': identity['family'],
                              'seed': identity['seed'], 'condition': t['ood_condition'],
                              'error': t['metrics']['normalized_rollout_mse'], 'horizon': t['horizon']})
    rows = []
    for key, values in sorted(groups.items()):
        row = dict(zip(('family', 'budget', 'variant', 'split'), key))
        row['rollouts'] = len(values)
        for metric in ('normalized_rollout_mse', 'one_step_mse', 'final_position_rmse',
                       'contact_accuracy', 'contact_brier', 'uncertainty_coverage95',
                       'floor_penetration_mean', 'momentum_total_mse'):
            v = [r[metric] for r in values if r.get(metric) is not None]
            row[metric] = float(np.mean(v)) if v else None
        rows.append(row)
    with (output / 'learning-curves.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    training = []
    policies = []
    for path in root.rglob('training-invocation-*.json'):
        d = json.loads(path.read_text())
        if d['identity']['variant'] == 'model_free':
            assert d['device'] == 'mps' and d['step'] == 400
            policies.append(d)
            continue
        assert d['device'] == 'mps' and d['placement_verified']
        assert set(d['training_trajectory_ids']) <= approved
        assert d['complete'] and d['step'] == 400
        training.append(d)
    result = {'cells': rows, 'worst_primary_final_ood': sorted(worst, key=lambda r: -r['error'])[:20],
              'ood_strata': [dict(zip(('family', 'variant', 'condition'), k),
                                  mean_error=float(np.mean(v)), rollouts=len(v))
                             for k, v in sorted(strata.items())],
              'training': {'dynamics_fits': len(training),
                           'gradient_updates': sum(d['new_updates'] for d in training),
                           'transition_presentations': sum(d['new_transition_presentations'] for d in training),
                           'summed_fit_seconds': sum(d['seconds'] for d in training),
                           'peak_driver_allocated_bytes': max(d['driver_allocated_bytes'] for d in training),
                           'parameter_counts': sorted({d['parameter_count'] for d in training}),
                           'training_only_ids_verified': True},
              'forecast_compute': {'batches': len(batches), 'summed_batch_seconds': sum(batches.values()),
                                   'predicted_scene_steps': neural_steps,
                                   'scene_steps_per_second': neural_steps / sum(batches.values()),
                                   'limit': 'Full run has at most100 scenes per horizon group and batch128; timings shared by scenes are counted once'},
              'policies': {'fits': len(policies), 'gradient_updates': sum(d['new_updates'] for d in policies),
                           'summed_fit_seconds': sum(d['seconds'] for d in policies),
                           'teacher_steps_per_fit': [d['teacher_environment_steps'] for d in policies],
                           'information_limit': 'Extra action-choice supervision; not equal-information'},
              'neural_calls': 0, 'statistics': 'descriptive pooled means; inferential scene pairing in acceptance.json'}
    atomic_json(output / 'error-analysis.json', result)
    print(json.dumps(result['training']))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--run', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    summarize(a.dataset, a.run, a.output)
