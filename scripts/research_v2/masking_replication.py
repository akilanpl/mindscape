"""Execute only the prospectively frozen 288 targeted masking control cases."""
import argparse
import copy
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from analyze_mechanisms import exact_p, holm
from coding_telemetry import digest, historical_index
from tracing import trace_case

from mindscape.data.schemas import BenchmarkExample
from mindscape.models.numpy_backend import NumpyMLP

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config_path = ROOT/'configs/research_v2/gap_closure/protocol.json'
    data_path = ROOT/'datasets/research_v2/masking_replication_v1.json'
    config, data = json.loads(config_path.read_text()), json.loads(data_path.read_text())
    if digest(config_path) != data['protocol_sha256']:
        raise RuntimeError('Frozen data/config mismatch')
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    cps = {s: args.evidence_root/f'results/learned_development_v1/mindscape_n50_seed{s}/checkpoint' for s in config['seeds']}
    pins = {str(p): digest(p) for cp in cps.values() for p in cp.iterdir() if p.is_file()}
    identity = {'protocol': digest(config_path), 'dataset': digest(data_path), 'checkpoints': pins,
                'source': {str(p.relative_to(ROOT)): digest(p) for folder in ('src', 'scripts/research_v2')
                           for p in sorted((ROOT/folder).rglob('*.py'))}, 'numpy': np.__version__}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    expected = {(seed, cond, r['example']['example_id']) for seed in cps for cond in config['conditions'] for r in data['cases']}
    with (output/'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = output/'IDENTITY.json'
        if manifest.exists() and json.loads(manifest.read_text())['fingerprint'] != fingerprint:
            raise RuntimeError('Conflicting output identity')
        manifest.write_text(json.dumps({'fingerprint': fingerprint, **identity}, indent=2)+'\n')
        path = output/'rows.jsonl'
        rows = [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
        lookup = {tuple(r['key']): r for r in rows}
        if len(lookup) != len(rows) or not set(lookup) <= expected:
            raise RuntimeError('Duplicate or unexpected keys')
        for row in rows:
            if digest(output/row['trace']) != row['sha256']:
                raise RuntimeError('Saved trace changed')
        (output/'traces').mkdir(exist_ok=True)
        for seed, cp in cps.items():
            trained = NumpyMLP.load(cp)
            training_ids = json.loads((cp/'model.json').read_text())['training_metadata']['training_ids']
            if set(training_ids) & {r['example']['example_id'] for r in data['cases']}:
                raise RuntimeError('Training identity leakage')
            for condition in config['conditions']:
                model = copy.deepcopy(trained)
                if condition.startswith('zero'):
                    for weight in model.weights.values():
                        weight.fill(0)
                for record in data['cases']:
                    example = BenchmarkExample(**record['example'])
                    key = (seed, condition, example.example_id)
                    if key in lookup:
                        continue
                    trace = trace_case(model, example, mask=condition.endswith('_mask'))
                    trace['intervention'] = {'weights_zeroed': condition.startswith('zero'),
                                             'selection': 'exact legal mask' if condition.endswith('_mask') else 'all action types admitted; same environment scorer'}
                    filename = f'traces/{seed}-{condition}-{example.example_id}.json.gz'
                    with gzip.open(output/filename, 'wt') as stream:
                        json.dump(trace, stream, allow_nan=False)
                    selections = [e['observed'] for e in trace['events'] if e['component']=='MindscapeModel.select_action']
                    if not selections:
                        raise RuntimeError('Unexecuted case cannot be scored')
                    row = {'key': list(key), 'family': record['family'], 'success': trace['result']['score'],
                           'failure': trace['result']['prediction']['error_type'] if trace['result']['prediction'] else trace['result']['failure'],
                           'legal_cardinalities': [len(e['valid_actions']) for e in selections],
                           'trace': filename, 'sha256': digest(output/filename)}
                    with path.open('a') as stream:
                        stream.write(json.dumps(row)+'\n');stream.flush();os.fsync(stream.fileno())
                    lookup[key] = row;rows.append(row)
        if set(lookup) != expected or any(digest(Path(p)) != h for p, h in pins.items()):
            raise RuntimeError('Incomplete evidence or mutated checkpoint')
        rng = np.random.default_rng(config['task_seed'])
        cells, tests, effects = [], [], []
        for family in config['families']:
            ids = [r['example']['example_id'] for r in data['cases'] if r['family']==family]
            for cond in config['conditions']:
                cells.append({'family': family, 'condition': cond,
                              'successes': sum(lookup[s, cond, t]['success'] for s in cps for t in ids), 'executions': 24})
            for policy in ('trained', 'zero'):
                diffs = []
                for seed in cps:
                    pairs = [(int(lookup[seed, policy+'_mask', t]['success']), int(lookup[seed, policy+'_permissive', t]['success'])) for t in ids]
                    b = sum(a and not c for a, c in pairs);c = sum(c and not a for a, c in pairs)
                    tests.append({'family': family, 'seed': seed, 'policy': policy, 'discordant': [b,c], 'exact_p': exact_p(b,c)})
                    diffs.append([a-c for a,c in pairs])
                means = np.asarray(diffs).mean(axis=0)
                bs = means[rng.integers(0,len(ids),size=(10000,len(ids)))].mean(axis=1)
                effects.append({'family': family, 'policy': policy, 'difference':float(means.mean()),
                    'task_cluster_bootstrap95':np.quantile(bs,[.025,.975]).tolist(),
                    'interpretation':'Conditional on three historical models; zero-weight controls are identical, not independent seeds'})
        for test, p in zip(tests, holm([t['exact_p'] for t in tests]), strict=True):test['holm_p_family18']=p
        result = {'status':'PASS','completed':len(rows),'unique_tasks':24,'cells':cells,'effects':effects,'tests':tests,
                  'unsuccessful':sum(not r['success'] for r in rows),'raw_sha256':digest(path),
                  'scope':config['scope'],'limitations':config['limits'],'checkpoints_unchanged':True}
        (output/'statistics.json').write_text(json.dumps(result,indent=2)+'\n')
        if not (output/'coding_historical').exists():historical_index(args.evidence_root,output/'coding_historical')
        print('PASS288 controlled executions, paired analysis and2320 historical coding telemetry pointers; resume skips saved keys')


if __name__ == '__main__':
    main()
