"""Freeze novel task identities before inference interventions; no model evaluation."""
import hashlib
import json
import random
from pathlib import Path

from mindscape.data.backends import get_backend

ROOT = Path(__file__).resolve().parents[2]


def main():
    protocol = ROOT/'configs/research_v2/mechanism_protocol_v1.json'
    config = json.loads(protocol.read_text())
    output = ROOT/'datasets/research_v2/numeric_mechanisms_v1.json'
    if output.exists():
        raise RuntimeError('Frozen data already exists; never overwrite')
    backend = get_backend('integer_multiplication')
    known = set()
    inputs = {}
    for path in sorted((ROOT/'datasets/generated').rglob('*.jsonl')):
        inputs[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        for line in path.read_text().splitlines():
            record = json.loads(line)
            if record.get('environment') == backend.name:
                known.add(record['example_id'])
    rows = []
    for split, spec in config['splits'].items():
        rng = random.Random(str(config['task_seed'])+':'+split)
        count = 0
        for _ in range(100000):
            observation = backend.sample(spec['structure'], rng, {'signed': spec['signed'], 'include_zero': False})
            example = backend.example(observation, config['task_seed'], split)
            if example.example_id in known:
                continue
            backend.validate(example)
            assert example.target_answer == observation['operands'][0]*observation['operands'][1]
            known.add(example.example_id)
            rows.append({'split': split, 'example': example.to_dict()})
            count += 1
            if count == spec['count']:
                break
        if count != spec['count']:
            raise RuntimeError('Unique task space exhausted; do not silently alter the protocol')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'protocol_sha256': hashlib.sha256(protocol.read_bytes()).hexdigest(),
                                 'historical_dataset_sha256': inputs,
                                 'novel_task_count': len(rows), 'cases': rows}, indent=2)+'\n')
    receipt = {'stage': 4, 'status': 'prospective protocol and task freeze PASS',
               'dataset': str(output.relative_to(ROOT)), 'dataset_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
               'protocol_sha256': hashlib.sha256(protocol.read_bytes()).hexdigest(),
               'cases': len(rows), 'seeds': config['model_seeds'], 'conditions': 8,
               'planned_executions': len(rows)*len(config['model_seeds'])*8,
               'leakage_check': 'No identity overlap with any preserved generated numeric train/validation/test split',
               'no_model_evaluation_during_freeze': True, 'new_architecture_or_training_changes': False,
               'interpretation_scope': config['scope'], 'inference_status': 'not yet run'}
    (ROOT/'experiments/research_v2/stage4_protocol_frozen.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print('PASS frozen', len(rows), 'novel cases;1920 executions planned; no inference run')


if __name__ == '__main__':
    main()
