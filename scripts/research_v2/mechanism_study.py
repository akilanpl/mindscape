"""Frozen three-seed factorial numeric study; every completed case stays auditable."""
import argparse
import fcntl
import gzip
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
from trace_diagnostic import render
from tracing import trace_case

from mindscape.data.schemas import BenchmarkExample
from mindscape.models.numpy_backend import NumpyMLP

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = ROOT/'configs/research_v2/mechanism_protocol_v1.json'
    data_path = ROOT/'datasets/research_v2/numeric_mechanisms_v1.json'
    protocol, dataset = json.loads(protocol_path.read_text()), json.loads(data_path.read_text())
    if dataset['protocol_sha256'] != digest(protocol_path):
        raise RuntimeError('Frozen protocol/data binding changed')
    cases = [BenchmarkExample(**r['example']) for r in dataset['cases']]
    if len({c.example_id for c in cases}) != 80:
        raise RuntimeError('Frozen case identities invalid')
    checkpoints = {s: args.evidence_root/protocol['checkpoint_template'].format(seed=s)
                   for s in protocol['model_seeds']}
    pins = {str(p): digest(p) for cp in checkpoints.values() for p in cp.iterdir() if p.is_file()}
    identity = {'protocol_sha256': digest(protocol_path), 'dataset_sha256': digest(data_path),
                'checkpoints': pins, 'numpy': np.__version__, 'python': platform.python_version(),
                'machine': platform.machine(),
                'source_sha256': {str(p.relative_to(ROOT)): digest(p) for folder in ('src', 'scripts/research_v2')
                                  for p in sorted((ROOT/folder).rglob('*.py'))}}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    expected = {(s, cond, c.example_id) for s in checkpoints for cond in protocol['conditions'] for c in cases}
    with (output/'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = output/'IDENTITY.json'
        if manifest_path.exists():
            if json.loads(manifest_path.read_text())['fingerprint'] != fingerprint:
                raise RuntimeError('Conflicting study namespace; never merge different code/inputs')
        else:
            manifest_path.write_text(json.dumps({'fingerprint': fingerprint, **identity}, indent=2)+'\n')
        rows_path = output/'rows.jsonl'
        saved = [json.loads(x) for x in rows_path.read_text().splitlines()] if rows_path.exists() else []
        keys = [tuple(r['key']) for r in saved]
        if len(keys) != len(set(keys)) or not set(keys) <= expected:
            raise RuntimeError('Duplicate/unexpected completed keys')
        done = set(keys)
        for r in saved:
            if digest(output/r['trace_path']) != r['trace_sha256']:
                raise RuntimeError('Completed trace checksum changed')
        (output/'traces').mkdir(exist_ok=True)
        (output/'reports').mkdir(exist_ok=True)
        for seed, cp in checkpoints.items():
            metadata = json.loads((cp/'model.json').read_text())['training_metadata']
            assert metadata['config']['seed'] == seed and metadata['config']['budget'] == protocol['training_budget']
            if set(metadata['training_ids']) & {c.example_id for c in cases}:
                raise RuntimeError('Training/evaluation identity leakage')
            model = NumpyMLP.load(cp)
            assert model.groups == [4] and model.seed == seed
            for condition in protocol['conditions']:
                flags = dict(zip(('mask', 'state_features', 'previous_action_features'), (c == '1' for c in condition), strict=True))
                for case in cases:
                    key = (seed, condition, case.example_id)
                    if key in done:
                        continue
                    filename = f'{seed}-{condition}-{case.example_id}'
                    path = output/'traces'/(filename+'.json.gz')
                    if path.exists():
                        with gzip.open(path, 'rt') as stream:
                            trace = json.load(stream)
                        if trace['study_fingerprint'] != fingerprint or tuple(trace['study_key']) != key:
                            raise RuntimeError('Orphan trace belongs to a different study')
                    else:
                        trace = trace_case(model, case, **flags)
                        if not any(e['computation'] == 'learned neural forward' for e in trace['events']):
                            raise RuntimeError('Case did not execute neural computation; do not score as zero')
                        trace.update(study_fingerprint=fingerprint, study_key=list(key))
                        temporary = path.with_suffix('.tmp')
                        with gzip.open(temporary, 'wt') as stream:
                            json.dump(trace, stream, allow_nan=False, separators=(',', ':'))
                        with temporary.open('rb') as stream:
                            os.fsync(stream.fileno())
                        temporary.replace(path)
                    (output/'reports'/(filename+'.md')).write_text(render(trace))
                    prediction = trace['result'].get('prediction')
                    decisions = prediction['diagnostics']['decisions'] if prediction else []
                    row = {'key': list(key), 'seed': seed, 'condition': condition,
                           'task_id': case.example_id, 'split': case.metadata['split'],
                           'success': trace['result']['score'],
                           'trajectory_valid': trace['result']['independent_verification']['trajectory_valid'],
                           'execution_status': trace['result']['execution_status'],
                           'error': prediction['error_type'] if prediction else trace['result'].get('failure'),
                           'neural_calls': sum(e['computation'] == 'learned neural forward' for e in trace['events']),
                           'mask_corrections': sum(d['masked_correction'] for d in decisions),
                           'wall_seconds_including_telemetry': trace['wall_seconds'],
                           'trace_path': str(path.relative_to(output)), 'trace_sha256': digest(path)}
                    with rows_path.open('a') as stream:
                        stream.write(json.dumps(row)+'\n')
                        stream.flush()
                        os.fsync(stream.fileno())
                    done.add(key)
                    if len(done) % 100 == 0:
                        print('completed', len(done), '/', len(expected), flush=True)
        if done != expected:
            raise RuntimeError('Study incomplete; no missing outcome fabrication')
        if any(digest(Path(p)) != h for p, h in pins.items()):
            raise RuntimeError('Trained checkpoints mutated')
        (output/'complete.json').write_text(json.dumps({'status': 'PASS', 'fingerprint': fingerprint,
            'completed': len(done), 'planned': len(expected), 'unique_tasks': len(cases),
            'seed_count': len(checkpoints), 'conditions': 8, 'checkpoints_unchanged': True,
            'scope': protocol['scope'], 'raw_sha256': digest(rows_path)}, indent=2)+'\n')
        print('PASS', len(done), 'real scored factorial cases; no coding-foundation or superiority claim')


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT/'scripts/research_v2'))
    main()
