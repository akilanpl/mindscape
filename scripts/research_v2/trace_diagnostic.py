"""Run four real archived-checkpoint diagnostics and render observed traces."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

from tracing import trace_case

from mindscape.data.backends import get_backend
from mindscape.models.numpy_backend import NumpyMLP

REPO = Path(__file__).resolve().parents[2]


def render(trace):
    result = trace['result']
    text = f"# Trace {trace['task_id']}\n\n{trace['interpretation']}\n\n"
    text += '## Exact input and configuration\n\n```json\n'+json.dumps({
        'input': trace['exact_model_input'], 'configuration': trace['configuration'],
        'model': trace['model']}, indent=2)+'\n```\n\n'
    text += '## Actual computation path\n\n'
    for event in trace['events']:
        text += f"- {event['index']}: {event['component']} — {event['computation']}\n"
    text += '\nThe JSON includes every observed numeric feature/logit/probability, shape and memory change. Hidden activation summaries are explicitly reconstructed; probabilities are derived and uncalibrated. Neither establishes why an output occurred. Structured state/action transitions and final verification are in `result.episode` and `result.independent_verification`.\n\n'
    text += '## Result\n\n```json\n'+json.dumps({k: result[k] for k in
        ('score', 'execution_status', 'independent_verification')}, indent=2)+'\n```\n\n'
    text += 'Components absent from this backend: '+', '.join(trace['unimplemented_here'])+'.\n'
    return text


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    config_path = REPO/'configs/research_v2/trace_diagnostic.json'
    config = json.loads(config_path.read_text())
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError('Diagnostic output already exists; inspect before repeating')
    output.mkdir(parents=True)
    checkpoint = args.evidence_root/config['checkpoint']
    model = NumpyMLP.load(checkpoint)
    records = []
    backend = get_backend('integer_multiplication')
    for operands in config['inputs']:
        example = backend.example({'operands': operands, 'kind': 'observation',
                                   'source': 'instrumentation_diagnostic'}, 20261009, 'diagnostic')
        for mask in config['mask_settings']:
            trace = trace_case(model, example, mask=mask)
            name = example.example_id+('-masked' if mask else '-unmasked')
            path = output/(name+'.json')
            path.write_text(json.dumps(trace, indent=2, allow_nan=False)+'\n')
            (output/(name+'.md')).write_text(render(trace))
            records.append({'path': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                            'score': trace['result']['score'], 'mask': mask})
    (output/'MANIFEST.json').write_text(json.dumps({'scope': config['scope'], 'records': records,
         'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
         'checkpoint_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in checkpoint.iterdir() if p.is_file()},
         'real_saved_checkpoint': True, 'benchmark_claim': False}, indent=2)+'\n')
    print('PASS', len(records), 'actual learned-core diagnostic traces; not foundation benchmarks')


if __name__ == '__main__':
    sys.path.insert(0, str(REPO/'scripts/research_v2'))
    main()
