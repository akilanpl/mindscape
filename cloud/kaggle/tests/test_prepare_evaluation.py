"""Pure staging/controller validation tests; no model inference or optimization."""
import importlib.util
import json
from pathlib import Path
import pytest

REPO = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location('prepare_evaluation', REPO / 'cloud/kaggle/scripts/prepare_evaluation.py')
builder = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(builder)


def fixture(tmp_path):
    payload = tmp_path / 'payload'; data = tmp_path / 'data'; fits = tmp_path / 'fits'
    (payload / 'source/src/mindscape').mkdir(parents=True)
    (payload / 'source/src/mindscape/__init__.py').write_text('')
    builder.write(payload / 'manifest.json', {'source_commit': 'd54ff8f',
        'source_files_sha256': {'src/mindscape/__init__.py': builder.sha(payload / 'source/src/mindscape/__init__.py')}})
    files = {}; entries = []
    for seed in builder.SEEDS:
        for rel in (f'train/{seed}.json', f'evaluator-only/final-{seed}.json'):
            path = data / rel; path.parent.mkdir(parents=True, exist_ok=True); builder.write(path, [])
            files[rel] = builder.sha(path)
        for condition in builder.CONDITIONS:
            folder = fits / f'{seed}-{condition}'; folder.mkdir(parents=True)
            builder.write(folder / 'final.json', {'metadata': {'optimizer_steps': 2000}, 'seed': seed, 'condition': condition})
            checkpoint_sha = builder.sha(folder / 'final.json')
            builder.write(folder / 'result.json', {'status': 'COMPLETED_EXPLORATORY', 'optimizer_updates': 2000,
                'optimizer_steps': 2000, 'step_post_hook_count': 2000, 'checkpoint_sha256': checkpoint_sha,
                'seed': seed, 'condition': condition})
            entries.append({'seed': seed, 'condition': condition, 'path': str(folder / 'final.json'),
                'receipt_path': str(folder / 'result.json'), 'sha256': checkpoint_sha,
                'kernel_id': f'owner/train-{seed}-{condition}', 'version': '1'})
    builder.write(data / 'manifest.json', {'files': files})
    index = tmp_path / 'index.json'; builder.write(index, entries)
    return payload, data, index


def test_stage_requires_all_nine_finished_fits(tmp_path):
    payload, data, index = fixture(tmp_path)
    values = builder.load(index); builder.write(index, values[:8])
    with pytest.raises(ValueError, match='Exactly nine'):
        builder.stage(REPO, payload, data, index, 'owner', 'evaluator-v1', tmp_path / 'stage')


def test_stage_private_cpu_no_scoring_and_pinned_core(tmp_path):
    payload, data, index = fixture(tmp_path)
    out = tmp_path / 'stage'
    result = builder.stage(REPO, payload, data, index, 'owner', 'evaluator-v1', out)
    metadata = builder.load(out / 'kernel-metadata.json')
    assert metadata['is_private'] is True and metadata['enable_internet'] is False
    assert metadata['enable_gpu'] is False and metadata['enable_tpu'] is False
    assert len(metadata['kernel_sources']) == 9
    assert len(result['data_files']) == 6 and len(result['evaluations']) == 3
    assert result['historical_evaluation_reuse'] is True
    source = (out / 'kernel.py').read_text()
    assert 'STAGED_NO_SCORING' in source and 'result = run(' not in source
    compile(source, '<staging launcher>', 'exec')
    final_source = builder.launcher(out, {'test_fixture_only': True})
    compile(final_source, '<evaluation launcher>', 'exec')
    assert 'torch.optim' not in final_source
    assert "input_root = pathlib.Path('/kaggle/input')" in final_source


def test_checkpoint_receipt_tampering_rejected(tmp_path):
    _, _, index = fixture(tmp_path)
    values = builder.load(index); receipt = Path(values[0]['receipt_path'])
    data = builder.load(receipt); data['optimizer_updates'] = 1999; builder.write(receipt, data)
    with pytest.raises(ValueError, match='optimizer count mismatch'):
        builder.validate_checkpoint_index(values)


def test_finalization_does_not_invent_provider_evidence(tmp_path):
    payload, data, index = fixture(tmp_path); out = tmp_path / 'stage'
    builder.stage(REPO, payload, data, index, 'owner', 'evaluator-v1', out)
    evidence = tmp_path / 'evidence.json'; builder.write(evidence, {})
    with pytest.raises(PermissionError, match='Kaggle runtime evidence required'):
        builder.finalize(out, evidence)
    assert not (out / 'finalize-reservation.json').exists()


def test_staged_core_changes_refused(tmp_path):
    payload, data, index = fixture(tmp_path); out = tmp_path / 'stage'
    builder.stage(REPO, payload, data, index, 'owner', 'evaluator-v1', out)
    (out / 'evaluator-core.zip').write_bytes(b'changed')
    evidence = tmp_path / 'evidence.json'; builder.write(evidence, {})
    with pytest.raises(ValueError, match='core changed'):
        builder.finalize(out, evidence)


@pytest.mark.parametrize('dataset_transport', [False, True])
def test_finalize_keeps_core_identity_and_binds_all_trainers(tmp_path, dataset_transport):
    payload, data, index = fixture(tmp_path); out = tmp_path / 'stage'
    receipt = dataset_receipt(tmp_path, index)[0] if dataset_transport else None
    blueprint = builder.stage(REPO, payload, data, index, 'owner', 'evaluator-v1', out, receipt)
    common = {'version': '1', 'source_sha256': 'a' * 64, 'remote_metadata_sha256': 'b' * 64,
              'credential_access_code_absent': True, 'control_plane_source_verified': True,
              'payload_evaluation_records': 0, 'evaluation_inputs_attached': False}
    trainers = [{**common, 'kernel_id': e['kernel_id'], 'metadata': {'id': e['kernel_id'],
                 'is_private': True, 'enable_internet': False}} for e in blueprint['checkpoints']]
    worker_path = 'cloud/kaggle/evaluation/worker.py'
    evaluator = {**common, 'kernel_id': blueprint['kernel_id'], 'version': '2',
                 'source_sha256': blueprint['core_source_files'][worker_path],
                 'metadata': builder.load(out / 'kernel-metadata.json'),
                 'source_identity_scope': 'evaluator-core-file', 'source_path': worker_path,
                 'core_source_files': blueprint['core_source_files']}
    evidence = {'provider': 'kaggle', 'protection_scope': 'frozen-runtime-isolation',
                'controller_trusted': True, 'checkpoint_export_hash_verified': True,
                'checkpoint_input_version': 'nine-unique-one-shot-version-1-inputs',
                'trainer': trainers[0], 'trainers': trainers, 'evaluator': evaluator,
                'test_fixture_only': True}
    if dataset_transport:
        evidence['checkpoint_transport'] = blueprint['checkpoint_transport']
        evidence['checkpoint_input_version'] = [blueprint['checkpoint_transport']['reference']]
        evidence['evaluator']['metadata']['dataset_sources'] = ['owner/final-checkpoints-v1']
    # Actual Kaggle pull changes kernel.py to the notebook slug's Python filename.
    evidence['evaluator']['metadata']['code_file'] = 'mindscape-evaluator-v1.py'
    evidence['evaluator']['metadata']['kernel_sources'].reverse()
    evidence_path = tmp_path / 'evidence.json'; builder.write(evidence_path, evidence)
    before = builder.sha(out / 'evaluator-core.zip')
    result = builder.finalize(out, evidence_path)
    assert result['status'] == 'FINALIZED_READY_FOR_PRIVATE_CPU_SUBMISSION'
    assert builder.sha(out / 'evaluator-core.zip') == before
    compile((out / 'kernel.py').read_text(), '<final launcher>', 'exec')
    with pytest.raises(FileExistsError): builder.finalize(out, evidence_path)
    evidence['evaluator']['metadata']['kernel_sources'].append('owner/unexpected-source')
    builder.write(evidence_path, evidence)
    with pytest.raises(ValueError, match='source references differ'):
        builder.finalize(out, evidence_path)
    evidence['evaluator']['metadata']['kernel_sources'].pop()
    evidence['evaluator']['metadata']['code_file'] = '../unexpected.py'
    builder.write(evidence_path, evidence)
    with pytest.raises(ValueError, match='code_file must name'):
        builder.finalize(out, evidence_path)


def dataset_receipt(tmp_path, index):
    entries = builder.load(index)
    checked = builder.validate_checkpoint_index(entries)
    files = {}
    for entry in checked:
        stem = f"{entry['seed']}-{entry['condition']}"
        files[stem + '-final.json'] = entry['sha256']
        files[stem + '-result.json'] = entry['receipt_sha256']
    receipt = {'dataset_id': 'owner/final-checkpoints-v1', 'version': 1,
        'is_private': True, 'provider_metadata_sha256': 'e' * 64,
        'download_verified': True, 'files_sha256': files, 'checkpoints': entries}
    path = tmp_path / 'dataset-receipt.json'; builder.write(path, receipt)
    return path, checked


def test_private_dataset_transport_pins_all_checkpoint_receipts(tmp_path):
    payload, data, index = fixture(tmp_path)
    receipt, checked = dataset_receipt(tmp_path, index)
    transport = builder.validate_checkpoint_dataset(receipt, checked)
    assert transport['reference'] == 'owner/final-checkpoints-v1/1'
    assert len(transport['files_sha256']) == 18
    out = tmp_path / 'stage'
    blueprint = builder.stage(REPO, payload, data, index, 'owner', 'evaluator-dataset-v1', out, receipt)
    metadata = builder.load(out / 'kernel-metadata.json')
    assert metadata['kernel_sources'] == []
    assert metadata['dataset_sources'] == ['owner/final-checkpoints-v1/1']
    assert blueprint['checkpoint_transport']['receipt_sha256'] == builder.sha(receipt)
    assert all(entry['input_filename'].endswith('-final.json') for entry in blueprint['checkpoints'])
    final_source = builder.launcher(out, {'test_fixture_only': True})
    compile(final_source, '<dataset evaluator launcher>', 'exec')
    assert "entry.get('receipt_filename', 'result.json')" in final_source


@pytest.mark.parametrize('field,value', [('is_private', False), ('download_verified', False), ('version', 2), ('provider_metadata_sha256', '')])
def test_unverified_or_unpinned_dataset_receipts_rejected(tmp_path, field, value):
    _, _, index = fixture(tmp_path); path, checked = dataset_receipt(tmp_path, index)
    receipt = builder.load(path); receipt[field] = value; builder.write(path, receipt)
    with pytest.raises(ValueError): builder.validate_checkpoint_dataset(path, checked)


def test_dataset_extra_evaluation_file_or_altered_checkpoint_rejected(tmp_path):
    _, _, index = fixture(tmp_path); path, checked = dataset_receipt(tmp_path, index)
    receipt = builder.load(path); receipt['files_sha256']['evaluation-final.json'] = 'a' * 64; builder.write(path, receipt)
    with pytest.raises(ValueError, match='exactly the 18'): builder.validate_checkpoint_dataset(path, checked)
    receipt['files_sha256'].pop('evaluation-final.json')
    first = next(iter(receipt['files_sha256'])); receipt['files_sha256'][first] = 'f' * 64
    builder.write(path, receipt)
    with pytest.raises(ValueError, match='exactly the 18'): builder.validate_checkpoint_dataset(path, checked)
