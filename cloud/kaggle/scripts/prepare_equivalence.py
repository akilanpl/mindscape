"""Prepare a private, offline CUDA/CPU inference validation kernel. Never submit."""
from __future__ import annotations
import argparse
import base64
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import sys
import zipfile

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from cloud.kaggle.reproducibility import verify_export, FROZEN_COMMIT


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def prepare(training_payload, checkpoint_index, owner, slug, output, checkpoint_dataset_receipt=None):
    payload, output = map(Path, (training_payload, output))
    spec = importlib.util.spec_from_file_location('equivalence_index_validator', REPO / 'cloud/kaggle/scripts/prepare_evaluation.py')
    validator = importlib.util.module_from_spec(spec); spec.loader.exec_module(validator)
    raw_index = load(checkpoint_index)
    raw_entries = raw_index['checkpoints'] if isinstance(raw_index, dict) else raw_index
    checkpoints = validator.validate_checkpoint_index(raw_index)
    transport = validator.validate_checkpoint_dataset(checkpoint_dataset_receipt, checkpoints) if checkpoint_dataset_receipt else None
    if transport:
        checkpoints = transport['checkpoints']
    for entry in raw_entries:
        receipt = verify_export(Path(entry['receipt_path']).parent)
        if receipt['checkpoint_sha256'] != entry['sha256']:
            raise ValueError('Verified export differs from checkpoint index')
    manifest = load(payload / 'manifest.json')
    if manifest['source_commit'] != FROZEN_COMMIT:
        raise ValueError('Require exact frozen R5.7c source commit')
    contents = {}
    for relative, digest in manifest['source_files_sha256'].items():
        source = payload / 'source' / relative
        if source.is_symlink() or not source.resolve().is_relative_to((payload / 'source').resolve()) or sha(source) != digest:
            raise ValueError('Frozen source bytes differ')
        contents[relative] = source.read_bytes()
    for seed, digest in manifest['training_files_sha256'].items():
        source = payload / f'train/{seed}.json'
        if sha(source) != digest:
            raise ValueError('Frozen training bytes differ')
        rows = load(source)
        if len(rows) != 128 or any(r['split'] != 'train' or r['seed'] != int(seed) for r in rows):
            raise ValueError('Expected original frozen training records')
        contents[f'train/{seed}.json'] = source.read_bytes()
    identity = f'{owner}/{slug}'
    if not re.fullmatch(r'[A-Za-z0-9_-]+/[a-z0-9-]+', identity):
        raise ValueError('Invalid private kernel identity')
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, value in sorted(contents.items()):
            archive.writestr(name, value)
    blueprint = {'schema': 'mindscape-numerical-equivalence-v1', 'source_commit': FROZEN_COMMIT,
                 'source_manifest_sha256': sha(payload / 'manifest.json'), 'checkpoints': checkpoints,
                 'files_sha256': {name: hashlib.sha256(value).hexdigest() for name, value in contents.items()},
                 'archive_sha256': hashlib.sha256(buffer.getvalue()).hexdigest(),
                 'atol': 1e-5, 'rtol': 1e-4, 'training_observations_only': True,
                 'optimizer_updates': 0, 'independent_evaluation': False,
                 'checkpoint_transport': transport}
    metadata = {'id': identity, 'title': slug, 'code_file': 'kernel.py', 'language': 'python',
                'kernel_type': 'script', 'is_private': True, 'enable_gpu': True, 'enable_tpu': False,
                'enable_internet': False, 'dataset_sources': [transport['reference']] if transport else [], 'competition_sources': [],
                'model_sources': [], 'kernel_sources': [] if transport else sorted(e['kernel_id'] for e in checkpoints)}
    output.mkdir(parents=True, exist_ok=False)
    source = RUNTIME.replace('__ARCHIVE_B64__', repr(base64.b64encode(buffer.getvalue()).decode())).replace('__BLUEPRINT__', repr(blueprint))
    (output / 'kernel.py').write_text(source)
    for name, value in [('kernel-metadata.json', metadata), ('equivalence-blueprint.json', blueprint)]:
        (output / name).write_text(json.dumps(value, sort_keys=True, indent=2))
    receipt = {'status': 'PREPARED_INFERENCE_ONLY_NOT_EXECUTED', 'optimizer_updates': 0,
               'kernel_sha256': sha(output / 'kernel.py'), 'metadata_sha256': sha(output / 'kernel-metadata.json'),
               'blueprint_sha256': sha(output / 'equivalence-blueprint.json')}
    (output / 'preparation-receipt.json').write_text(json.dumps(receipt, indent=2, sort_keys=True))
    return receipt


RUNTIME = r'''"""Genuine CPU/CUDA inference validation of frozen finalized weights; zero updates."""
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import sys
import traceback
import zipfile

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
BLUEPRINT = __BLUEPRINT__
ARCHIVE_B64 = __ARCHIVE_B64__

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))

def restore(model, path, entry, torch):
    if path.stat().st_size > 64000000 or sha(path) != entry['sha256']:
        raise ValueError('Pinned checkpoint differs')
    payload = read(path)
    meta = payload.get('metadata', {})
    if payload.get('format') != 'r56-json-tensors-v1' or payload.get('scientifically_qualified') is not False or (meta.get('seed'), meta.get('condition'), meta.get('optimizer_steps')) != (entry['seed'], entry['condition'], 2000):
        raise ValueError('Finalized safe checkpoint identity differs')
    state = model.state_dict()
    if set(payload['tensors']) != set(state):
        raise ValueError('Checkpoint keys differ')
    restored = {}
    for name, expected in state.items():
        value = payload['tensors'][name]
        if value['shape'] != list(expected.shape) or value['dtype'] != str(expected.dtype):
            raise ValueError('Tensor schema differs')
        tensor = torch.tensor(value['values'], dtype=expected.dtype, device='cpu')
        if tensor.shape != expected.shape or not bool(tensor.isfinite().all()):
            raise ValueError('Tensor shape/finite check failed')
        restored[name] = tensor
    model.load_state_dict(restored, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model

def decode(output, domain):
    if domain == 'math':
        return {'math_digit': int(output['math_digit_logits'].argmax()), 'math_carry': int(output['math_carry_logits'].argmax())}
    if domain == 'coding':
        return {'coding_index': int(output['coding_logits'].argmax())}
    value = output['physics_delta'].detach().cpu()
    return {'physics_delta': value.tolist() if bool(value.isfinite().all()) else None}

def main():
    destination = Path('/kaggle/working/equivalence-output')
    destination.mkdir(exist_ok=False)
    receipt = {'schema': BLUEPRINT['schema'], 'status': 'STARTED', 'optimizer_updates': 0,
               'independent_evaluation': False, 'training_observations_only': True,
               'atol': BLUEPRINT['atol'], 'rtol': BLUEPRINT['rtol'], 'source_commit': BLUEPRINT['source_commit'],
               'python': sys.version, 'platform': platform.platform(), 'fits': []}
    try:
        if not Path('/kaggle/input').is_dir() or platform.system() != 'Linux':
            raise RuntimeError('Actual Kaggle Linux CUDA execution required')
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            raise RuntimeError('Actual CUDA required; no CPU substitution')
        torch.cuda.set_device(0)
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        receipt.update(torch=str(torch.__version__), cuda_version=torch.version.cuda,
                       cudnn_version=torch.backends.cudnn.version(), gpu_name=torch.cuda.get_device_name(0),
                       cuda_device_count=torch.cuda.device_count(),
                       gpu_total_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
                       device='cuda:0', cpu_fallback=False, deterministic=True, tf32=False)
        raw = base64.b64decode(ARCHIVE_B64)
        if hashlib.sha256(raw).hexdigest() != BLUEPRINT['archive_sha256']:
            raise ValueError('Embedded frozen source archive changed')
        runtime = Path('/kaggle/working/equivalence-runtime'); runtime.mkdir(exist_ok=False)
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            for name in archive.namelist():
                if not (runtime / name).resolve().is_relative_to(runtime):
                    raise ValueError('Unsafe archive member')
            archive.extractall(runtime)
        for name, expected in BLUEPRINT['files_sha256'].items():
            if sha(runtime / name) != expected:
                raise ValueError('Frozen source/training file hash differs')
        sys.path.insert(0, str(runtime / 'src'))
        from mindscape.r57c.ordinal_model import make_model
        for name, module in tuple(sys.modules.items()):
            if name in ('mindscape.r56.learning', 'mindscape.r56.compact_gpu', 'mindscape.r57c.runner'):
                raise RuntimeError('Training helper imported in inference validator')
            if name.startswith('mindscape.') and getattr(module, '__file__', None):
                path = Path(module.__file__).resolve()
                relative = str(path.relative_to(runtime))
                if relative not in BLUEPRINT['files_sha256'] or sha(path) != BLUEPRINT['files_sha256'][relative]:
                    raise ValueError('Imported source outside exact frozen pins')
        candidates = list(Path('/kaggle/input').rglob('*final.json'))
        transport = BLUEPRINT.get('checkpoint_transport')
        if transport:
            receipt['checkpoint_transport'] = {k: transport[k] for k in ('dataset_id', 'version', 'reference', 'receipt_sha256', 'provider_metadata_sha256')}
            for filename, expected in transport['files_sha256'].items():
                copies = [p for p in Path('/kaggle/input').rglob(filename) if sha(p) == expected]
                if len(copies) != 1 or os.access(copies[0], os.W_OK):
                    raise ValueError('Require exact read-only private dataset checkpoint and receipt files')
        for entry in BLUEPRINT['checkpoints']:
            matches = [p for p in candidates if sha(p) == entry['sha256']]
            if len(matches) != 1:
                raise ValueError('Require exactly one immutable mounted checkpoint by hash')
            path = matches[0]
            sidecar = path.parent / entry.get('receipt_filename', 'result.json')
            if not sidecar.is_file() or sha(sidecar) != entry['receipt_sha256']:
                raise ValueError('Pinned finalized checkpoint receipt differs')
            if os.access(path, os.W_OK):
                raise PermissionError('Checkpoint input must be mounted read-only')
            cpu = restore(make_model(entry['condition'], entry['seed']), path, entry, torch).cpu()
            cuda = restore(make_model(entry['condition'], entry['seed']), path, entry, torch).to('cuda:0')
            training = read(runtime / f"train/{entry['seed']}.json")
            fit = {'seed': entry['seed'], 'condition': entry['condition'], 'checkpoint_sha256': entry['sha256'],
                   'kernel_id': entry['kernel_id'], 'checkpoint_read_only': True, 'domains': []}
            receipt['fits'].append(fit)
            with torch.inference_mode():
                for domain in ('math', 'coding', 'physics'):
                    row = next(row for row in training if row['problem']['domain'] == domain)
                    problem = row['problem']
                    left = cpu.predict(problem); right = cuda.predict(problem)
                    if left.keys() != right.keys():
                        raise ValueError('Output head keys differ')
                    heads = []
                    for key, a in left.items():
                        if not torch.is_tensor(a):
                            continue
                        b = right[key]
                        if a.device.type != 'cpu' or b.device.type != 'cuda':
                            raise RuntimeError('Head device substitution detected')
                        b_cpu = b.detach().cpu()
                        finite = bool(a.isfinite().all()) and bool(b_cpu.isfinite().all())
                        close = finite and torch.allclose(a, b_cpu, atol=BLUEPRINT['atol'], rtol=BLUEPRINT['rtol'])
                        item = {'head': key, 'shape': list(a.shape), 'finite': finite, 'allclose': bool(close),
                                'max_absolute_error': float((a - b_cpu).abs().max()) if finite and a.numel() else None,
                                'cpu_output': a.tolist() if bool(a.isfinite().all()) else None,
                                'cuda_output': b_cpu.tolist() if bool(b_cpu.isfinite().all()) else None}
                        if key.endswith('_logits'):
                            item['same_argmax'] = int(a.argmax()) == int(b_cpu.argmax())
                        heads.append(item)
                    decoded_cpu, decoded_cuda = decode(left, domain), decode(right, domain)
                    fit['domains'].append({'domain': domain, 'training_record_identity': row.get('identity'),
                         'heads': heads, 'cpu_prediction': decoded_cpu, 'cuda_prediction': decoded_cuda,
                         'decoded_predictions_exactly_equal': decoded_cpu == decoded_cuda,
                         'discrete_prediction_changed': decoded_cpu != decoded_cuda if domain != 'physics' else None})
            del cpu, cuda
            if sha(path) != entry['sha256']:
                raise ValueError('Immutable checkpoint changed during validation')
        if len(receipt['fits']) != 9 or sum(len(f['domains']) for f in receipt['fits']) != 27:
            raise ValueError('Require all nine fits and 27 domain checks')
        heads = [h for fit in receipt['fits'] for domain in fit['domains'] for h in domain['heads']]
        receipt['all_heads_within_tolerance'] = all(h['allclose'] for h in heads)
        receipt['discrete_prediction_changes'] = sum(d['discrete_prediction_changed'] is True for f in receipt['fits'] for d in f['domains'])
        receipt['max_absolute_error'] = max((h['max_absolute_error'] for h in heads if h['max_absolute_error'] is not None), default=None)
        receipt['status'] = 'PASSED_NUMERICAL_EQUIVALENCE' if receipt['all_heads_within_tolerance'] and receipt['discrete_prediction_changes'] == 0 else 'NUMERICAL_DIFFERENCES_OBSERVED'
    except BaseException as exc:
        receipt.update(status='FAILED_NUMERICAL_VALIDATION', error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
    finally:
        with (destination / 'numerical_equivalence_receipt.json').open('x') as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False)
        print(json.dumps({k: receipt[k] for k in ('status', 'optimizer_updates')}), flush=True)
    if receipt['status'] != 'PASSED_NUMERICAL_EQUIVALENCE':
        raise RuntimeError(receipt['status'] + '; preserved numerical receipt; no source or tolerance changed')

if __name__ == '__main__':
    main()
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('training-payload', 'checkpoint-index', 'owner', 'slug', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--checkpoint-dataset-receipt', type=Path)
    prepare(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
