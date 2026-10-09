"""One cached-weight coding telemetry diagnostic; never counted as a new benchmark case."""
import argparse
import fcntl
import json
import resource
import time
from pathlib import Path

from coding_telemetry import digest, observe

from mindscape.coding.backend import CodingBackend
from mindscape.coding.model import LocalCoder
from mindscape.coding.policy import CodingPolicy
from mindscape.coding.schema import CodeTask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--condition', choices=('model_only', 'mindscape_c'), default='model_only')
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('Inspect existing probe; never silently duplicate inference')
    args.output.mkdir(parents=True)
    with (args.output/'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        model = Path('work/coding/hf/models--Qwen--Qwen2.5-Coder-1.5B-Instruct/snapshots/2e1fd397ee46e1388853d2af2c993145b0f1098a')
        adapter = Path('results/coding/gradient_v1/seed_11/checkpoint_100') if args.condition=='model_only' else Path('results/coding/completion_gradient_v1/mindscape_c/seed_11/checkpoint_100')
        pins = {str(p): digest(p) for root in (model, adapter) for p in root.iterdir() if p.is_file()}
        task = CodeTask(**json.loads(Path('results/coding/completion_lockbox_v1/tasks.json').read_text())[0])
        print('Loading existing pinned weights on CPU; no downloads', flush=True)
        started = time.time()
        coder = LocalCoder(model, adapter=adapter, cache_enabled=False, device='cpu', precision='float32')
        events = []
        if args.condition == 'mindscape_c':
            from mindscape.coding.memory import CodingMemory
            from mindscape.coding.policy import ToolCodingPolicy
            teacher = Path('results/coding/final_retrieval_v1/teacher_memory.json')
            pins[str(teacher)] = digest(teacher)
            memory = CodingMemory(episodic=json.loads(teacher.read_text())[:100])
            policy = ToolCodingPolicy(coder, CodingBackend()._sandbox(), memory, max_steps=8, max_edits=3)
        else:
            policy = CodingPolicy(coder, CodingBackend()._sandbox(), condition='model_only')
        with observe(coder, events):
            result = policy.solve(task, seed=11)
        if any(digest(Path(p)) != h for p, h in pins.items()):
            raise RuntimeError('Cached weights changed')
        import sys
        (args.output/'trace.json').write_text(json.dumps({'scope': 'One repeated historical task as a telemetry diagnostic, not new accuracy evidence',
            'task_id': task.task_id, 'condition': args.condition, 'model_revision': coder.revision, 'parameter_count': coder.parameter_count,
            'checkpoint_sha256': pins, 'backend': 'cpu', 'precision': 'float32', 'cache_enabled': False,
            'events': events, 'result': result, 'elapsed_seconds_including_load': time.time()-started,
            'peak_process_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
            'limitations': ['Historical prompts/logits remain unavailable', 'One diagnostic does not measure benchmark accuracy',
                           'Observer overhead is included in timing', 'No causal activation explanation']}, indent=2)+'\n')
        if not any(e['kind'] == 'actual_forward_output' for e in events):
            raise RuntimeError('No actual neural computation observed; partial trace preserved')
        print('PASS observed coding diagnostic', len(events), 'events; success', result['success'], flush=True)


if __name__ == '__main__':
    main()
