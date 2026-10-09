"""Analyze only complete, checksum-bound paired executions of the frozen study."""
import argparse
import hashlib
import html
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_p(b, c):
    n = b+c
    return min(1., 2*sum(math.comb(n, k) for k in range(min(b, c)+1))/2**n) if n else 1.


def holm(values):
    result = [0.]*len(values)
    previous = 0.
    for rank, index in enumerate(sorted(range(len(values)), key=values.__getitem__)):
        previous = max(previous, min(1., (len(values)-rank)*values[index]))
        result[index] = previous
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    args = parser.parse_args()
    folder = args.study
    protocol = json.loads((ROOT/'configs/research_v2/mechanism_protocol_v1.json').read_text())
    dataset = json.loads((ROOT/'datasets/research_v2/numeric_mechanisms_v1.json').read_text())
    complete = json.loads((folder/'complete.json').read_text())
    rows = [json.loads(line) for line in (folder/'rows.jsonl').read_text().splitlines()]
    expected = {(s, c, r['example']['example_id']) for s in protocol['model_seeds']
                for c in protocol['conditions'] for r in dataset['cases']}
    lookup = {tuple(r['key']): r for r in rows}
    if len(rows) != len(lookup) or set(lookup) != expected or complete['raw_sha256'] != digest(folder/'rows.jsonl'):
        raise RuntimeError('Incomplete, duplicate or changed evidence')
    for row in rows:
        if row['success'] not in (True, False, 0, 1) or row['neural_calls'] < 1:
            raise RuntimeError('Invalid/unexecuted outcome')
        if digest(folder/row['trace_path']) != row['trace_sha256']:
            raise RuntimeError('Trace checksum mismatch')
    cells, tests, contrasts = [], [], []
    rng = np.random.default_rng(202610100)
    for split in protocol['splits']:
        ids = sorted({r['task_id'] for r in rows if r['split'] == split})
        if len(ids) != 40:
            raise RuntimeError('Split incomplete')
        for seed in protocol['model_seeds']:
            for condition in protocol['conditions']:
                outcomes = [lookup[seed, condition, task]['success'] for task in ids]
                p = sum(outcomes)/len(ids)
                z, n = 1.959963984540054, len(ids)
                center = (p+z*z/(2*n))/(1+z*z/n)
                half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
                cells.append({'split': split, 'seed': seed, 'condition': condition, 'successes': sum(outcomes), 'n': n,
                                  'rate': p, 'wilson95': [center-half, center+half]})
        for control in ('011', '101', '110'):
            diffs = []
            for seed in protocol['model_seeds']:
                pairs = [(bool(lookup[seed, '111', t]['success']), bool(lookup[seed, control, t]['success'])) for t in ids]
                b = sum(a and not c for a, c in pairs)
                c = sum(c and not a for a, c in pairs)
                tests.append({'split': split, 'seed': seed, 'comparison': '111-'+control,
                                  'discordant': [b, c], 'difference': (b-c)/40, 'exact_p': exact_p(b, c)})
                diffs.append([int(a)-int(c) for a, c in pairs])
            task_means = np.asarray(diffs).mean(axis=0)
            bootstrap = task_means[rng.integers(0, 40, size=(10000, 40))].mean(axis=1)
            contrasts.append({'split': split, 'comparison': '111-'+control, 'seed_mean_difference': float(task_means.mean()),
                                  'task_cluster_bootstrap95': np.quantile(bootstrap, [.025, .975]).tolist(),
                                  'uncertainty_scope': 'Task sampling conditional on these three fixed trained seeds'})
    for test, adjusted in zip(tests, holm([t['exact_p'] for t in tests]), strict=True):
        test['holm_p_family18'] = adjusted
    result = {'status': 'PASS', 'completed': len(rows), 'unique_tasks': 80, 'cells': cells, 'primary_tests': tests,
                  'primary_contrasts': contrasts, 'executed_unsuccessful': sum(not r['success'] for r in rows),
                  'scope': protocol['scope'], 'limitations': protocol['limitations'],
                  'latency_note': 'Instrumented trace times include observer overhead; not dedicated latency benchmarks',
                  'historical_models': 'Unrun: exact GPT/Gemini releases, access and budget unresolved',
                  'raw_sha256': digest(folder/'rows.jsonl')}
    (folder/'statistics.json').write_text(json.dumps(result, indent=2)+'\n')
    lines = ['# Frozen numeric mechanism study', '', result['scope'], '',
             f"Executed: {len(rows)}; unsuccessful: {result['executed_unsuccessful']}. All outcomes retained.", '',
             '| Split | Seed | Condition | Success / 40 | Wilson 95% interval |', '|---|---|---|---|---|']
    for c in cells:
        lines.append(f"| {c['split']} | {c['seed']} | {c['condition']} | {c['successes']} | {c['wilson95']} |")
    lines += ['', 'Primary paired tests use exact McNemar with Holm correction across 18 tests. Bootstrap intervals cluster by task, conditional on three fixed model seeds.', '',
              '## Limitations', *['- '+x for x in result['limitations']], '', result['latency_note'], result['historical_models']]
    report = '\n'.join(lines)+'\n'
    (folder/'REPORT.md').write_text(report)
    (folder/'dashboard.html').write_text('<!doctype html><meta charset="utf-8"><title>Mindscape numeric study</title><h1>Verified numeric mechanism evidence</h1><pre>'+html.escape(report)+'</pre>')
    manifest = {str(p.relative_to(folder)): digest(p) for p in sorted(folder.rglob('*')) if p.is_file() and p.name not in ('runner.lock', 'MANIFEST.json')}
    (folder/'MANIFEST.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('PASS complete paired analysis, 18 corrected tests, dashboard and artifact hashes')


if __name__ == '__main__':
    main()
