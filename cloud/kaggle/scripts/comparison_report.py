"""Verify completed cloud evidence and produce descriptive comparisons; never fit."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from cloud.kaggle.reproducibility import verify_export, SEEDS, CONDITIONS

EXPECTED = {(s, c) for s in SEEDS for c in CONDITIONS}
METRICS = (('math', 'joint_accuracy'), ('coding', 'accuracy'), ('physics', 'mse'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def metrics(records):
    groups = {d: [r for r in records if r['domain'] == d] for d, _ in METRICS}
    if any(not rows for rows in groups.values()):
        raise ValueError('All three evaluation domains required')
    result = {}
    for d, key in METRICS:
        rows = groups[d]
        if d == 'physics':
            values = [sum((a - b) ** 2 for a, b in zip(r['prediction']['physics_delta'], r['reference']['physics_delta'])) / 2 for r in rows]
        else:
            keys = ('math_digit', 'math_carry') if d == 'math' else ('coding_index',)
            values = [float(all(r['prediction'][k] == r['reference'][k] for k in keys)) for r in rows]
        result[d] = {'count': len(rows), key: statistics.mean(values)}
    positions = defaultdict(list)
    for row in groups['coding']:
        positions[str(row['reference']['coding_index'])].append(row['prediction']['coding_index'] == row['reference']['coding_index'])
    result['coding']['by_target_position'] = {k: {'count': len(v), 'accuracy': statistics.mean(v)} for k, v in sorted(positions.items())}
    result['coding']['balanced_position_accuracy'] = statistics.mean(v['accuracy'] for v in result['coding']['by_target_position'].values())
    return result


def verify_metrics(declared, actual):
    for domain, key in METRICS:
        if declared[domain]['count'] != actual[domain]['count'] or not math.isclose(declared[domain][key], actual[domain][key], rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError('Declared metrics do not match preserved predictions')


def descriptive(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['cohort'], row['condition'])].append(row)
    result = []
    for (cohort, condition), selected in sorted(groups.items()):
        if sorted(r['seed'] for r in selected) != sorted(SEEDS):
            raise ValueError('Exactly three unique research seeds required per cohort/condition')
        result.append({'cohort': cohort, 'condition': condition, 'seeds': list(SEEDS), 'metrics': {
            domain: {'mean': statistics.mean(values), 'min': min(values), 'max': max(values), 'values': values}
            for domain, key in METRICS
            for values in [[next(r for r in selected if r['seed'] == seed)['metrics'][domain][key] for seed in SEEDS]]}})
    return result


def probe_summary(records):
    groups = defaultdict(list)
    for row in records:
        groups[(row['problem']['domain'], row['probe'])].append(row)
    result = []
    for (domain, probe), selected in sorted(groups.items()):
        summary = {'domain': domain, 'probe': probe, 'count': len(selected),
                   'prediction_changed_fraction': statistics.mean(r['prediction'] != r['original_prediction'] for r in selected)}
        if domain == 'physics':
            summary['mse_to_analytic_reference'] = statistics.mean(sum((a - b) ** 2 for a, b in zip(r['prediction']['physics_delta'], r['analytic_reference']['physics_delta'])) / 2 for r in selected)
        else:
            keys = ('math_digit', 'math_carry') if domain == 'math' else ('coding_index',)
            summary['exact_accuracy_to_analytic_reference'] = statistics.mean(all(r['prediction'][k] == r['analytic_reference'][k] for k in keys) for r in selected)
        if probe.startswith('candidate_rotation_'):
            summary['same_selected_name_fraction'] = statistics.mean(r['same_selected_name'] for r in selected)
        result.append(summary)
    return result


def _identity_records(records):
    result = {r['normalized_identity']: r for r in records}
    if len(result) != len(records):
        raise ValueError('Duplicate prediction identities')
    if any(r.get('repaired') is not False for r in records):
        raise ValueError('Repaired predictions forbidden')
    return result


def build(fits, evaluation, historical, output):
    fits, evaluation, historical, output = map(Path, (fits, evaluation, historical, output))
    if output.exists():
        raise FileExistsError('Refuse to overwrite comparison evidence')
    receipts = {}
    index = {}
    for path in sorted(fits.rglob('result.json')):
        receipt = verify_export(path.parent)
        key = receipt['seed'], receipt['condition']
        if key in receipts:
            raise ValueError('Duplicate completed fit; operator review required')
        receipts[key] = receipt
        index[str(path.resolve())] = sha(path)
    if set(receipts) != EXPECTED:
        raise ValueError('All nine exported fits required before comparison')
    receipt_path = evaluation / 'evaluation_receipt.json'
    receipt = read(receipt_path)
    if receipt.get('status') != 'COMPLETED_CLOUD_EVALUATION' or receipt.get('device') != 'cpu' or receipt.get('evaluation_optimizer_updates') != 0 or receipt.get('qualification') is not False or receipt.get('selection_performed') is not False:
        raise ValueError('Completed independent CPU evaluation receipt required')
    index[str(receipt_path.resolve())] = sha(receipt_path)
    historical_manifest = read(historical / 'data-v1/manifest.json')
    scoreboard_path = historical / 'scoreboard-final/scoreboard.json'
    scoreboard = read(scoreboard_path)
    index[str(scoreboard_path.resolve())] = sha(scoreboard_path)
    rows, probes, controls, seen = [], [], [], set()
    distribution_hashes = {}
    for job in receipt['results']:
        seed = job['seed']
        if seed in distribution_hashes or seed not in SEEDS:
            raise ValueError('Duplicate or unknown evaluation seed')
        expected = historical_manifest['files'][f'evaluator-only/final-{seed}.json']
        actual_path = historical / f'data-v1/evaluator-only/final-{seed}.json'
        if sha(actual_path) != expected or job.get('evaluation_sha256') != expected:
            raise ValueError('Historical/cloud distribution bytes differ or are unverified')
        distribution_hashes[seed] = expected
        if job.get('training_sha256') != receipts[(seed, CONDITIONS[0])]['training_sha256']:
            raise ValueError('Evaluator training identity differs')
        if job['overlap_audit']['normalized_overlap'] != 0 or job['overlap_audit']['global_training_union']['normalized_overlap'] != 0:
            raise ValueError('Evaluation overlaps training')
        for run in job['runs']:
            key = run['seed'], run['condition']
            if key in seen or key not in EXPECTED or key[0] != seed or run['version'] != 'R57c':
                raise ValueError('Unknown or duplicate evaluated fit')
            seen.add(key)
            path = evaluation / run['artifact']
            if not path.resolve().is_relative_to(evaluation.resolve()) or sha(path) != run['sha256']:
                raise ValueError('Evaluation raw artifact hash mismatch')
            value = read(path)
            if value['checkpoint_sha256'] != receipts[key]['checkpoint_sha256']:
                raise ValueError('Evaluator used different checkpoint')
            current = metrics(value['records']); verify_metrics(run['metrics'], current); verify_metrics(value['metrics'], current)
            indexed = _identity_records(value['records'])
            rows.append({'cohort': 'R57c CUDA cloud', 'seed': seed, 'condition': key[1], 'metrics': current, 'artifact': str(path.resolve())})
            probes.append({'seed': seed, 'condition': key[1], 'summaries': probe_summary(value['perturbations']), 'raw_artifact': str(path.resolve())})
            index[str(path.resolve())] = sha(path)
            for version in ('R56', 'R57', 'R57c'):
                old_path = historical / f'scoreboard-final/predictions/{seed}-{key[1]}-{version}-final.json'
                old = read(old_path); old_indexed = _identity_records(old['records'])
                if indexed.keys() != old_indexed.keys() or any(indexed[k]['reference'] != old_indexed[k]['reference'] for k in indexed):
                    raise ValueError('Historical predictions refer to different cases or labels')
                old_metrics = metrics(old['records']); verify_metrics(old['metrics'], old_metrics)
                summary = [r for r in scoreboard['summary'] if (r['seed'], r['condition'], r['version'], r['phase']) == (seed, key[1], version, 'final')]
                if len(summary) != 1:
                    raise ValueError('Missing historical final summary')
                verify_metrics(summary[0]['metrics'], old_metrics)
                rows.append({'cohort': version + ' historical', 'seed': seed, 'condition': key[1], 'metrics': old_metrics, 'artifact': str(old_path.resolve())})
                index[str(old_path.resolve())] = sha(old_path)
        for name in ('analytic_rule', 'training_majority_constant'):
            value = job['baselines'][name]
            computed = metrics(value['records']); verify_metrics(value['metrics'], computed)
            controls.append({'seed': seed, 'baseline': name, 'metrics': computed, 'classification': 'Non-neural control; not native Mindscape performance'})
    if seen != EXPECTED or set(distribution_hashes) != set(SEEDS):
        raise ValueError('All nine independent evaluations required')
    changes = []
    for row in rows:
        if row['cohort'] != 'R57c CUDA cloud':
            continue
        prior = next(r for r in rows if r['cohort'] == 'R57c historical' and (r['seed'], r['condition']) == (row['seed'], row['condition']))
        changes.append({'seed': row['seed'], 'condition': row['condition'], 'cloud_minus_historical': {
            d: row['metrics'][d][k] - prior['metrics'][d][k] for d, k in METRICS},
            'regression_domains': [d for d, k in METRICS if (row['metrics'][d][k] > prior['metrics'][d][k] if d == 'physics' else row['metrics'][d][k] < prior['metrics'][d][k])]})
    initial_comparison = []
    for old in scoreboard['fit_receipts']:
        key = old.get('seed'), old.get('condition')
        if key in receipts and 'initial_training_loss' in old and 'initial_training_loss' in receipts[key]:
            initial_comparison.append({'seed': key[0], 'condition': key[1], 'cloud_minus_historical_initial_loss': {d: receipts[key]['initial_training_loss'][d] - old['initial_training_loss'][d] for d, _ in METRICS}})
    result = {'status': 'VERIFIED_DESCRIPTIVE_COMPARISON', 'qualification': False,
              'optimizer_updates_verified': sum(r['optimizer_updates'] for r in receipts.values()),
              'evaluation_optimizer_updates': 0, 'distribution_sha256': distribution_hashes,
              'per_seed': rows, 'seed_descriptions': descriptive(rows), 'perturbations': probes,
              'cuda_vs_historical_r57c_changes': changes, 'initial_loss_comparison': initial_comparison,
              'non_neural_controls': controls, 'cloud_training_torch_versions': sorted({r['torch'] for r in receipts.values()}),
              'historical_training_torch_versions': sorted({r['torch'] for r in scoreboard['fit_receipts']}),
              'limitations': ['Descriptive three-seed mean/range only; no statistical superiority or acceptance claim.',
                'Same historical evaluation distribution reused; not a fresh independent generalization distribution.',
                'Training backend, operating system and PyTorch version changed together; observed differences cannot be attributed solely to CUDA.',
                'Initial weights may differ across PyTorch versions even with assigned seeds; small initial-loss differences are not proof of a backend-only effect.',
                'Ordinal decimal output-geometry prior bypasses the math core; these tiny tasks do not establish CBSE Class 12 competence.']}
    output.mkdir(parents=True, exist_ok=False)
    for name, value in [('comparison.json', result), ('artifact-index.json', index)]:
        with (output / name).open('x') as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
    (output / 'CLOUD_COMPARISON.md').write_text(render(result))
    return result


def render(result):
    lines = ['# Verified cloud comparison', '', 'All nine CUDA fits and independent CPU evaluations passed artifact verification. 18,000 optimizer updates; zero evaluator updates. Native neural results below use exactly the same frozen evaluation bytes as the historical results.', '',
        '| Cohort | Condition | Math exact mean [range] | Coding exact mean [range] | Physics MSE mean [range] |',
        '|---|---|---:|---:|---:|']
    for row in result['seed_descriptions']:
        cells = [row['cohort'], row['condition']]
        for domain, _ in METRICS:
            x = row['metrics'][domain]
            cells.append(f"{x['mean']:.6g} [{x['min']:.6g}, {x['max']:.6g}]")
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines += ['', '## Per-seed native neural results', '', '| Cohort | Seed | Condition | Math exact | Coding exact | Physics MSE | Balanced coding positions |', '|---|---:|---|---:|---:|---:|---:|']
    for row in result['per_seed']:
        m = row['metrics']
        lines.append(f"| {row['cohort']} | {row['seed']} | {row['condition']} | {m['math']['joint_accuracy']:.6g} | {m['coding']['accuracy']:.6g} | {m['physics']['mse']:.6g} | {m['coding']['balanced_position_accuracy']:.6g} |")
    lines += ['', '## Descriptive regressions against historical R57c', '', '| Seed | Condition | Math delta | Coding delta | Physics MSE delta | Worse domains |', '|---:|---|---:|---:|---:|---|']
    for row in result['cuda_vs_historical_r57c_changes']:
        d = row['cloud_minus_historical']
        lines.append(f"| {row['seed']} | {row['condition']} | {d['math']:+.6g} | {d['coding']:+.6g} | {d['physics']:+.6g} | {', '.join(row['regression_domains']) or 'none'} |")
    lines += ['', 'Deltas are cloud minus historical: lower classification accuracy or higher physics MSE is worse. These flags are descriptive and do not establish statistical significance.', '', '## Perturbations', '', 'Raw logits, decoded predictions, original observations and derived probes remain in the hashed per-fit evaluation artifacts. The comparison JSON preserves per-position counts/accuracy and each probe summary. Prediction changes alone do not imply correct sensitivity.', '', '| Seed | Condition | Probe | Count | Prediction changed | Exact accuracy or physics MSE vs analytic reference |', '|---:|---|---|---:|---:|---:|']
    for fit in result['perturbations']:
        for probe in fit['summaries']:
            score = probe.get('exact_accuracy_to_analytic_reference', probe.get('mse_to_analytic_reference'))
            lines.append(f"| {fit['seed']} | {fit['condition']} | {probe['probe']} | {probe['count']} | {probe['prediction_changed_fraction']:.6g} | {score:.6g} |")
    lines += ['', '## Deterministic and non-neural controls', '', 'These results are separate from native neural performance. The analytic control directly implements the reference rules.', '', '| Seed | Control | Math exact | Coding exact | Physics MSE |', '|---:|---|---:|---:|---:|']
    for row in result['non_neural_controls']:
        m = row['metrics']
        lines.append(f"| {row['seed']} | {row['baseline']} | {m['math']['joint_accuracy']:.6g} | {m['coding']['accuracy']:.6g} | {m['physics']['mse']:.6g} |")
    lines += ['', '## Interpretation limits', ''] + ['- ' + x for x in result['limitations']]
    lines += ['', 'Cloud training PyTorch: ' + ', '.join(result['cloud_training_torch_versions']) + '. Historical training PyTorch: ' + ', '.join(result['historical_training_torch_versions']) + '.', '', 'Artifact SHA-256 identities are recorded in artifact-index.json; machine-readable metrics and raw artifact references are in comparison.json.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('fits', 'evaluation', 'historical', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    build(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
