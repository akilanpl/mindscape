"""Pure packaging/static checks; no local neural fitting or inference."""
import ast
import importlib.util
import json
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / 'cloud/kaggle/scripts/prepare_equivalence.py'
spec = importlib.util.spec_from_file_location('prepare_equivalence', PATH)
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


def test_runtime_has_no_training_calls_or_imports():
    tree = ast.parse(module.RUNTIME.replace('__ARCHIVE_B64__', repr('')).replace('__BLUEPRINT__', repr({})))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in {'backward', 'step', 'Adam', 'SGD', 'zero_grad'}
        if isinstance(node, ast.ImportFrom):
            assert node.module not in {'mindscape.r56.learning', 'mindscape.r56.compact_gpu', 'mindscape.r57c.runner'}
    assert "torch.inference_mode()" in module.RUNTIME
    assert "NUMERICAL_DIFFERENCES_OBSERVED" in module.RUNTIME
    assert "cpu_fallback=False" in module.RUNTIME


def test_private_offline_package_requires_all_checkpoints(tmp_path):
    index = tmp_path / 'index.json'; index.write_text('[]')
    with pytest.raises(ValueError, match='Exactly nine'):
        module.prepare(tmp_path, index, 'owner', 'slug', tmp_path / 'output')
    assert not (tmp_path / 'output').exists()


@pytest.mark.parametrize('use_dataset', [False, True])
def test_private_offline_runtime_contains_exact_frozen_bytes(tmp_path, monkeypatch, use_dataset):
    payload = tmp_path / 'payload'; source = payload / 'source/src/mindscape/model.py'
    source.parent.mkdir(parents=True); source.write_text('# frozen source\n')
    train_hashes = {}
    for seed in (17, 29, 43):
        path = payload / f'train/{seed}.json'; path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps([{'seed': seed, 'split': 'train'}] * 128))
        train_hashes[str(seed)] = module.sha(path)
    (payload / 'manifest.json').write_text(json.dumps({'source_commit': module.FROZEN_COMMIT,
        'source_files_sha256': {'src/mindscape/model.py': module.sha(source)}, 'training_files_sha256': train_hashes}))
    entries = []
    for seed in (17, 29, 43):
        for condition in ('graph-concept-on', 'graph-concept-off', 'matched-gru'):
            run = tmp_path / f'{seed}-{condition}'; run.mkdir()
            checkpoint = run / 'final.json'; checkpoint.write_text(json.dumps({'metadata': {'optimizer_steps': 2000, 'seed': seed, 'condition': condition}}))
            receipt = {'status': 'COMPLETED_EXPLORATORY', 'optimizer_updates': 2000,
                       'optimizer_steps': 2000, 'step_post_hook_count': 2000, 'seed': seed,
                       'condition': condition, 'checkpoint_sha256': module.sha(checkpoint)}
            receipt_path = run / 'result.json'; receipt_path.write_text(json.dumps(receipt))
            entries.append({'seed': seed, 'condition': condition, 'kernel_id': f'owner/fit-{seed}-{condition}',
                            'version': 1, 'path': str(checkpoint), 'receipt_path': str(receipt_path), 'sha256': module.sha(checkpoint)})
    index = tmp_path / 'index.json'; index.write_text(json.dumps(entries))
    monkeypatch.setattr(module, 'verify_export', lambda path: module.load(path / 'result.json'))
    dataset_receipt = None
    if use_dataset:
        dataset_receipt = tmp_path / 'dataset-receipt.json'
        files = {}
        for entry in entries:
            stem = f"{entry['seed']}-{entry['condition']}"
            files[stem + '-final.json'] = entry['sha256']
            files[stem + '-result.json'] = module.sha(entry['receipt_path'])
        dataset_receipt.write_text(json.dumps({'dataset_id': 'owner/frozen-checkpoints', 'version': 1,
            'is_private': True, 'download_verified': True, 'provider_metadata_sha256': 'a' * 64,
            'files_sha256': files, 'checkpoints': entries}))
    output = tmp_path / 'prepared'; module.prepare(payload, index, 'owner', 'numerical-check', output, dataset_receipt)
    meta = module.load(output / 'kernel-metadata.json')
    blueprint = module.load(output / 'equivalence-blueprint.json')
    assert meta['is_private'] is True and meta['enable_gpu'] is True and meta['enable_internet'] is False
    if use_dataset:
        assert meta['kernel_sources'] == []
        assert meta['dataset_sources'] == ['owner/frozen-checkpoints/1']
        assert len(blueprint['checkpoint_transport']['files_sha256']) == 18
        assert all(e['input_filename'].endswith('-final.json') for e in blueprint['checkpoints'])
        assert all(e['receipt_filename'].endswith('-result.json') for e in blueprint['checkpoints'])
    else:
        assert len(meta['kernel_sources']) == 9
        assert meta['dataset_sources'] == []
    assert blueprint['atol'] == 1e-5 and blueprint['rtol'] == 1e-4
    assert blueprint['optimizer_updates'] == 0
    assert set(blueprint['files_sha256']) == {'src/mindscape/model.py', 'train/17.json', 'train/29.json', 'train/43.json'}
    ast.parse((output / 'kernel.py').read_text())
    with pytest.raises(FileExistsError):
        module.prepare(payload, index, 'owner', 'numerical-check', output)


def test_runtime_verifies_flat_dataset_sidecars_and_readonly_mounts():
    assert "rglob('*final.json')" in module.RUNTIME
    assert "entry.get('receipt_filename', 'result.json')" in module.RUNTIME
    assert "sha(sidecar) != entry['receipt_sha256']" in module.RUNTIME
    assert "transport['files_sha256'].items()" in module.RUNTIME
    assert "os.access(copies[0], os.W_OK)" in module.RUNTIME
