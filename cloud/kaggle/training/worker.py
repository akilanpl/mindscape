"""One-shot Kaggle CUDA fit of the frozen R5.7c exploratory protocol.

No evaluator imports, no resume, no local fitting, no reduced-budget option.
The package manifest binds source bytes and the original ordered training bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

SEEDS = (17, 29, 43)
CONDITIONS = ('graph-concept-on', 'graph-concept-off', 'matched-gru')
PROTOCOL = {'optimizer': 'Adam', 'learning_rate': .001, 'batch_size': 1,
            'optimizer_updates': 2000, 'seeds': list(SEEDS), 'conditions': list(CONDITIONS)}
TRAINING_HASHES = {
    '17': '663392c1c3c2e004d05685b852a847a0baedf21c9bf27e1e32350d8168f91f92',
    '29': 'e38e69d2024ac7c9dd00188c20394fcf9c92b335d41a1b31d073905f88e87d7f',
    '43': 'd530e849e4fc74b3ec992e3cbce12fe4229fffef44ac7c37524bdea70682a5de',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)


def validate_package(package, seed, condition):
    """Pure file validation before torch/model/optimizer imports."""
    package = Path(package).resolve(strict=True)
    manifest = json.loads((package / 'manifest.json').read_text())
    if seed not in SEEDS or condition not in CONDITIONS:
        raise ValueError('Fit outside frozen nine-fit comparison')
    if manifest.get('protocol') != PROTOCOL:
        raise ValueError('Frozen protocol differs')
    if not str(manifest.get('source_commit', '')).startswith('d54ff8f'):
        raise ValueError('Require verified R5.7c d54ff8f source identity')
    if manifest.get('training_files_sha256') != TRAINING_HASHES:
        raise ValueError('Require exact historical training-file identities')
    hashes = manifest.get('source_files_sha256', {})
    required = {'src/mindscape/r57c/ordinal_model.py', 'src/mindscape/r57/model.py',
                'src/mindscape/r56/neural.py', 'src/mindscape/r56/native_domains.py',
                'src/mindscape/r56/benchmarks.py', 'src/mindscape/r56/learning.py',
                'src/mindscape/r56/compact_gpu.py'}
    if not required.issubset(hashes):
        raise ValueError('Manifest omits actual model/checkpoint/loss sources')
    source = package / 'source'
    actual_sources = {str(p.relative_to(source)) for p in (source / 'src').rglob('*.py')}
    if not actual_sources or not actual_sources.issubset(hashes):
        raise ValueError('Manifest omits staged source files')
    for relative, digest in hashes.items():
        path = source / relative
        if path.is_symlink() or not path.resolve().is_relative_to(source) or sha(path) != digest:
            raise ValueError('Source identity differs: ' + relative)
    data = package / 'train' / f'{seed}.json'
    if data.is_symlink() or sha(data) != TRAINING_HASHES[str(seed)]:
        raise ValueError('Historical training bytes/order changed')
    rows = json.loads(data.read_text())
    if len(rows) != 128 or any(r['split'] != 'train' or r['seed'] != seed for r in rows):
        raise ValueError('Expected exactly 128 frozen train records')
    return manifest, rows, source


def require_cuda():
    # An environment label alone is not hardware proof; CUDA checks below are mandatory.
    if not (Path('/kaggle/working').is_dir() and Path('/kaggle/input').is_dir()):
        raise RuntimeError('KAGGLE_CUDA_REQUIRED; no local fitting permitted')
    if platform.system() != 'Linux':
        raise RuntimeError('KAGGLE_LINUX_REQUIRED')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError('CUDA_REQUIRED; CPU substitution forbidden')
    torch.cuda.set_device(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats(0)
    return torch


def make_portable_model(condition, seed, device='cpu'):
    """Inference helper; MPS is historical compatibility, training always uses CUDA."""
    if device not in ('cpu', 'cuda', 'cuda:0', 'mps'):
        raise ValueError('Unsupported backend')
    from mindscape.r57c.ordinal_model import make_model
    return make_model(condition, seed).to(device)


def _equal_outputs(model, restored, prepared, torch):
    with torch.no_grad():
        for item in prepared:
            a, b = model(item), restored(item)
            if a.keys() != b.keys() or any(not torch.equal(a[k], b[k])
                                           for k in a if torch.is_tensor(a[k])):
                raise RuntimeError('Checkpoint reload changed inference')


def run_fit(package, output, seed, condition):
    """Create a fresh output directory. Any existing attempt requires operator review."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    began, updates = time.perf_counter(), 0
    try:
        manifest, rows, source = validate_package(package, seed, condition)
        torch = require_cuda()
        sys.path.insert(0, str(source / 'src'))
        module = importlib.import_module('mindscape.r57c.ordinal_model')
        if Path(module.__file__).resolve() != source / 'src/mindscape/r57c/ordinal_model.py':
            raise RuntimeError('Imported model differs from staged package')
        from mindscape.r56.learning import save_checkpoint, load_checkpoint
        # Reject stale imports from an unrelated checkout in reused notebook kernels.
        for name, imported in tuple(sys.modules.items()):
            filename = getattr(imported, '__file__', None)
            if name.startswith('mindscape.') and filename and str(filename).endswith('.py'):
                loaded = Path(filename).resolve()
                if not loaded.is_relative_to(source / 'src'):
                    raise RuntimeError('Mindscape import outside staged package: ' + name)
        model = make_portable_model(condition, seed, 'cuda:0')
        if any(p.device.type != 'cuda' for p in model.parameters()):
            raise RuntimeError('Model parameters not entirely CUDA')
        prepared = [model.prepare(r['problem']) for r in rows]

        def summary():
            values = {d: [] for d in ('math', 'coding', 'physics')}
            with torch.no_grad():
                for p, r in zip(prepared, rows):
                    loss = module.domain_loss(model(p), r, torch, 'cuda:0')
                    values[r['problem']['domain']].append(float(loss.cpu()))
            return {d: sum(v) / len(v) for d, v in values.items()}

        meta = {'seed': seed, 'condition': condition, 'optimizer_steps': 0,
                'scope': 'EXPLORATORY_NOT_SCIENTIFICALLY_QUALIFIED',
                'source_commit': manifest['source_commit'],
                'training_sha256': TRAINING_HASHES[str(seed)],
                'manifest_sha256': sha(Path(package) / 'manifest.json')}
        initial_loss = summary()
        initial_sha = save_checkpoint(model, output / 'initial.json', meta)
        restored = make_portable_model(condition, seed, 'cuda:0')
        load_checkpoint(restored, output / 'initial.json', initial_sha)
        _equal_outputs(model, restored, prepared[:3], torch)
        del restored
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        actual_steps, touches = [0], {}
        hook = optimizer.register_step_post_hook(
            lambda *args: actual_steps.__setitem__(0, actual_steps[0] + 1))
        torch.cuda.synchronize()
        start = time.perf_counter()
        with (output / 'loss.jsonl').open('x', buffering=1) as log:
            for step in range(2000):
                index = step % 128
                optimizer.zero_grad(set_to_none=True)
                loss = module.domain_loss(model(prepared[index]), rows[index], torch, 'cuda:0')
                if loss.device.type != 'cuda' or not bool(torch.isfinite(loss)):
                    raise RuntimeError('Invalid CUDA loss')
                loss.backward()
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                if not grads or any(g.device.type != 'cuda' or not bool(g.isfinite().all()) for g in grads):
                    raise RuntimeError('Invalid CUDA gradients')
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None:
                        touches[name] = touches.get(name, 0) + 1
                optimizer.step()
                updates = actual_steps[0]
                if any(not bool(p.isfinite().all()) for p in model.parameters()):
                    raise RuntimeError('Nonfinite weights')
                log.write(json.dumps({'update': updates, 'record_index': index,
                    'domain': rows[index]['problem']['domain'],
                    'preupdate_loss': float(loss.detach().cpu())}) + '\n')
                if updates % 250 == 0:
                    print(json.dumps({'seed': seed, 'condition': condition, 'updates': updates}), flush=True)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        hook.remove()
        steps = {n: int(optimizer.state[p]['step'].item()) for n, p in model.named_parameters()
                 if 'step' in optimizer.state[p]}
        if updates != 2000 or steps != touches:
            raise RuntimeError('Exact optimizer count/exposure verification failed')
        result = {**meta, 'optimizer_steps': updates, 'optimizer_updates': updates,
                  'step_post_hook_count': actual_steps[0], 'adam_steps_by_parameter': steps,
                  'gradient_exposures_by_parameter': touches, 'initial_sha256': initial_sha,
                  'initial_training_loss': initial_loss, 'final_training_loss': summary(),
                  'seconds': seconds, 'device': 'cuda:0', 'cuda_device_count': torch.cuda.device_count(),
                  'gpu_name': torch.cuda.get_device_name(0),
                  'gpu_total_memory_bytes': torch.cuda.get_device_properties(0).total_memory,
                  'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated(0),
                  'gpu_peak_reserved_bytes': torch.cuda.max_memory_reserved(0),
                  'torch': str(torch.__version__), 'cuda_version': torch.version.cuda,
                  'cudnn_version': torch.backends.cudnn.version(), 'python': sys.version,
                  'platform': platform.platform(), 'deterministic': True, 'tf32': False,
                  'cpu_fallback': False, 'parameter_count': sum(p.numel() for p in model.parameters()),
                  'unused_parameter_names': [n for n, p in model.named_parameters() if n not in steps],
                  'loss_sha256': sha(output / 'loss.jsonl'), 'initial_reload_bitwise_equal': True,
                  'checkpoint_policy': 'final-update-2000-only', 'official_updates': 0,
                  'numerical_note': 'Same source/FP32 protocol; CUDA versus historical MPS equivalence is not assumed.'}
        result['checkpoint_sha256'] = save_checkpoint(model, output / 'final.json', result)
        restored = make_portable_model(condition, seed, 'cuda:0')
        load_checkpoint(restored, output / 'final.json', result['checkpoint_sha256'])
        _equal_outputs(model, restored, prepared[:3], torch)
        result.update(status='COMPLETED_EXPLORATORY', final_reload_bitwise_equal=True)
        write_new(output / 'result.json', result)
        return result
    except BaseException as exc:
        write_new(output / 'failure.json', {'status': 'FAILED_NO_AUTOMATIC_RESUME',
                  'seed': seed, 'condition': condition, 'actual_optimizer_updates': updates,
                  'error_type': type(exc).__name__, 'error': str(exc),
                  'elapsed_seconds': time.perf_counter() - began, 'traceback': traceback.format_exc()})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, choices=SEEDS, required=True)
    parser.add_argument('--condition', choices=CONDITIONS, required=True)
    args = parser.parse_args()
    run_fit(args.package, args.output, args.seed, args.condition)


if __name__ == '__main__':
    main()
