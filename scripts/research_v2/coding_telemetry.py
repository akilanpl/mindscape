"""Live coding observers and checksum-bound historical index; no invented internals."""
import contextlib
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


@contextlib.contextmanager
def observe(coder, events):
    """Observe existing calls without changing generation settings or adding forwards."""
    from mindscape.coding.environment import CodeRepairEnvironment
    from mindscape.coding.memory import CodingMemory
    from mindscape.coding.planning import BoundedPlanner
    original_generate, original_step = coder.generate, CodeRepairEnvironment.step
    originals = {}
    started = time.perf_counter()
    def emit(kind, value):
        events.append({'kind': kind, 'elapsed_seconds': time.perf_counter()-started, 'observed': value})
    def generate(system, prompt, max_tokens=256, seed=0):
        emit('model_request', {'system': system, 'prompt': prompt, 'max_tokens': max_tokens, 'seed': seed,
                               'revision': coder.revision, 'provenance': 'Actual generate arguments'})
        try:
            result = original_generate(system, prompt, max_tokens, seed)
        except Exception as exc:
            emit('model_failure', {'type': type(exc).__name__, 'message': str(exc)})
            raise
        emit('model_response', {'response': result, 'timing': coder.timings[-1]})
        return result
    def step(env, action):
        result = original_step(env, action)
        emit('deterministic_environment_transition', asdict(result))
        return result
    for cls, names in ((CodingMemory, ('reset_working', 'remember', 'retrieve', 'dream')),
                       (BoundedPlanner, ('plan',))):
        for name in names:
            original = getattr(cls, name)
            originals[cls, name] = original
            def observed_component(instance, *args, _original=original, _name=name, _class=cls, **kwargs):
                before = {k: len(getattr(instance, k)) for k in ('working', 'episodic', 'concepts') if hasattr(instance, k)}
                result = _original(instance, *args, **kwargs)
                after_state = {k: len(getattr(instance, k)) for k in before}
                emit('deterministic_component', {'component': _class.__name__+'.'+_name,
                    'before_counts': before, 'after_counts': after_state,
                    'returned_count': len(result) if isinstance(result, (list, tuple)) else None,
                    'provenance': 'Actual invocation; retrieved/dreamed payloads occur in exact model requests; hypotheses are not outcomes'})
                return result
            setattr(cls, name, observed_component)
    hooks = []
    if getattr(coder, 'model', None) is not None:
        def before(module, args, kwargs):
            ids = kwargs.get('input_ids', args[0] if args else None)
            if ids is not None:
                emit('actual_token_input', {'shape': list(ids.shape), 'token_ids': ids.detach().cpu().tolist(),
                                           'device': str(ids.device), 'provenance': 'Direct model forward input'})
        def after(module, args, output):
            logits = output.logits
            last = logits[0, -1].detach().float()
            values, indices = last.topk(5)
            probabilities = last.softmax(-1)[indices]
            emit('actual_forward_output', {'logits_shape': list(logits.shape), 'dtype': str(logits.dtype),
                 'top5_ids': indices.cpu().tolist(), 'top5_logits': values.cpu().tolist(),
                 'derived_top5_probabilities': probabilities.cpu().tolist(),
                 'interpretation': 'Observed logits and derived probabilities, not causal explanation'})
        target = coder.model.get_base_model() if hasattr(coder.model, 'get_base_model') else coder.model
        hooks = [target.register_forward_pre_hook(before, with_kwargs=True),
                 target.register_forward_hook(after)]
    coder.generate, CodeRepairEnvironment.step = generate, step
    try:
        yield
    finally:
        coder.generate, CodeRepairEnvironment.step = original_generate, original_step
        for (cls, name), original in originals.items():
            setattr(cls, name, original)
        for hook in hooks:
            hook.remove()


def historical_index(evidence, output):
    """Pointers into immutable episodes avoid duplicating repositories and transitions."""
    output.mkdir(parents=True, exist_ok=False)
    sources, entries = {}, []
    for name in ('completion_lockbox_eval_v1', 'completion_gradient_v1'):
        path = evidence/'results/coding'/name/'rows.jsonl'
        sources[name] = {'path': str(path.relative_to(evidence)), 'sha256': digest(path)}
        for number, line in enumerate(path.read_text().splitlines(), 1):
            row = json.loads(line)
            transitions = row['trajectory']['transitions']
            entries.append({'source': name, 'line': number, 'raw_row_sha256': hashlib.sha256(line.encode()).hexdigest(),
                'task_id': row['task_id'], 'condition': row['condition'], 'key': row.get('key'),
                'success': row['success'], 'model_calls': row['model_calls'],
                'transition_count': len(transitions), 'component_actions': [t['action']['name'] for t in transitions],
                'observed_generation': [a.get('latency') for a in row['attempts']],
                'failures': [a.get('parse_error') for a in row['attempts']],
                'model_input_provenance': 'Historical exact prompt/token IDs/logits not retained; source task and transitions remain available',
                'adapter': row.get('adapter'), 'protocol': row.get('protocol'),
                'observed_output_pointer': 'attempts[].response', 'state_pointer': 'trajectory.transitions[].state_before/state_after'})
    if len(entries) != 2320:
        raise RuntimeError('Historical evidence count mismatch')
    (output/'index.json').write_text(json.dumps({'sources': sources, 'entries': entries,
        'scope': 'Historical observed evidence index, not new inference or reconstructed logits'}, indent=2)+'\n')


def render_probe(trace_path, report_path):
    """Readable observed path; full values stay in the single bound JSON artifact."""
    from collections import Counter
    trace = json.loads(trace_path.read_text())
    counts = Counter(e['observed'].get('component', e['kind']) for e in trace['events'])
    lines = ['# Observed coding-foundation diagnostic', '', trace['scope'], '',
             'Task: '+trace['task_id'], 'Model: '+trace['model_revision'],
             f"Backend: {trace['backend']} {trace['precision']}; generation cache: {trace['cache_enabled']}",
             f"Actual model calls: {trace['result']['model_calls']}; independently scored success: {trace['result']['success']}",
             f"Trace SHA-256: {digest(trace_path)}", '', '| Directly observed event/component | Count |', '|---|---:|']
    lines += [f'| {name} | {count} |' for name,count in sorted(counts.items())]
    lines += ['', 'Exact generate arguments and direct forward token IDs are in the JSON events. Logits are direct forward observations; their softmax summaries are derived, uncalibrated probabilities. State changes are actual environment transitions. Memory and planner events record actual invocations; hypothetical proposals are not executed outcomes or causal explanations.', '',
              'Failures: '+json.dumps([a['parse_error'] for a in trace['result']['attempts']]), '',
              'Historical prompts/logits are not retroactively recreated. These CPU float32 diagnostics are distinct from the original MPS/CPU benchmark measurements and are not new accuracy cases. Selected hidden activations and all-adapter live coverage remain unmeasured.']
    report_path.write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Render a checksum-bound actual coding trace')
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    options = parser.parse_args()
    render_probe(options.trace, options.report)
