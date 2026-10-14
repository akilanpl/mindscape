import json
import pytest
from mindscape.r56.curriculum import digest, generate, prepare_data
from mindscape.r56.learning import freeze, validate_gate, train
from mindscape.r56.benchmarks import exact_rule, Memorization, paired_bootstrap, holm


def test_deterministic_curriculum_private_labels_and_split_identities():
    data = prepare_data(17)
    assert data == prepare_data(17)
    identities = [{r['problem_identity'] for r in data[s]} for s in ('train', 'dev', 'test')]
    assert not identities[0] & identities[1]
    assert not identities[0] & identities[2]
    assert not identities[1] & identities[2]
    assert all('labels' not in row['problem'] and 'target_name' not in row['problem']
               for rows in data.values() for row in rows)
    for row in data['train']:
        expected = exact_rule(row['problem'])
        assert all(row['labels'][key] == value for key, value in expected.items())


def test_nontraining_freeze_is_unissued_and_content_locked(tmp_path):
    root = tmp_path / 'repo'
    source = root / 'src/mindscape/r56'
    source.mkdir(parents=True)
    (source / 'model.py').write_text('source = 1')
    manifest = freeze(root, tmp_path / 'freeze')
    datasets = json.loads((tmp_path / 'freeze/datasets.json').read_text())
    assert manifest['data_hash'] == digest(datasets)
    unissued = json.loads((tmp_path / 'freeze/authorization.UNISSUED.json').read_text())
    with pytest.raises(PermissionError):
        validate_gate(root, manifest, datasets, unissued, unissued, {})
    # Even asserted issued flags are not trusted without external signatures.
    forged = dict(unissued, issued=True, issuer='local', signature_hex='00')
    with pytest.raises(PermissionError):
        validate_gate(root, manifest, datasets, forged, forged, {'local': '00'})
    (source / 'model.py').write_text('changed = 2')
    with pytest.raises(PermissionError, match='Source/protocol'):
        validate_gate(root, manifest, datasets, forged, forged, {'local': '00'})


def test_training_gate_rejects_before_model_import(tmp_path, monkeypatch):
    root = tmp_path / 'repo'
    (root / 'src/mindscape/r56').mkdir(parents=True)
    freeze(root, tmp_path / 'freeze')
    import builtins
    original = builtins.__import__
    def protected(name, *args, **kwargs):
        if name in ('torch', 'mindscape.r56.neural'):
            raise AssertionError('Neural import before external gate')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', protected)
    with pytest.raises(PermissionError):
        train(root, tmp_path / 'freeze', {}, {}, {}, tmp_path / 'weights.json')
    assert not (tmp_path / 'weights.json').exists()


def test_baselines_and_paired_statistics():
    rows = generate('train', 32)
    memory = Memorization(rows)
    assert memory.predict(rows[0]['problem']) == rows[0]['labels']
    assert memory.predict({'unseen': True}) is None
    result = paired_bootstrap([1, 1, 1], [0, 0, 0], replicates=100)
    assert result['ci95'] == [1.0, 1.0]
    assert holm([.01, .04, .2]) == [.03, .08, .2]
    with pytest.raises(ValueError):
        paired_bootstrap([], [])
