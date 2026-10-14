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


def test_actual_untrained_graph_gradients_and_safe_checkpoint_replay(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    from mindscape.r56.neural import R56Model, ModelConfig
    from mindscape.r56.native_domains import pack_problem
    from mindscape.r56.learning import save_checkpoint, load_checkpoint
    from mindscape.r56.benchmarks import model_accounting
    # Architectural connectivity check only: no optimizer creation or update.
    monkeypatch.setattr(torch.optim, 'Adam', lambda *a, **k: (_ for _ in ()).throw(AssertionError('optimizer forbidden')))
    torch.manual_seed(17)
    model = R56Model(ModelConfig())
    before = {k: v.clone() for k, v in model.state_dict().items()}
    rows = generate('train', 6)
    total_loss = 0
    for row in rows:
        output = model(pack_problem(row['problem']), stateless=True)
        domain, labels = row['problem']['domain'], row['labels']
        if domain == 'math':
            total_loss = total_loss + torch.nn.functional.cross_entropy(output['math_digit_logits'].reshape(1, -1), torch.tensor([labels['math_digit']]))
            total_loss = total_loss + torch.nn.functional.cross_entropy(output['math_carry_logits'].reshape(1, -1), torch.tensor([labels['math_carry']]))
        elif domain == 'coding':
            total_loss = total_loss + torch.nn.functional.cross_entropy(output['coding_logits'].reshape(1, -1), torch.tensor([labels['coding_index']]))
        else:
            total_loss = total_loss + torch.nn.functional.mse_loss(output['physics_delta'], torch.tensor(labels['physics_delta']))
    total_loss.backward()
    assert torch.isfinite(total_loss)
    assert model.features.weight.grad is not None and model.features.weight.grad.abs().sum() > 0
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    accounting = model_accounting(model)
    assert accounting['parameter_count'] > 0 and accounting['gradient_bytes'] > 0
    model.eval()
    graph = pack_problem(rows[0]['problem'])
    original = model(graph, stateless=True)['math_digit_logits'].detach()
    path = tmp_path / 'weights.json'
    identity = save_checkpoint(model, path, {'trained': False, 'optimizer_updates': 0})
    replay = R56Model(ModelConfig())
    metadata = load_checkpoint(replay, path, identity)
    assert metadata['optimizer_updates'] == 0
    assert torch.equal(original, replay(graph, stateless=True)['math_digit_logits'].detach())
    with pytest.raises(ValueError, match='hash'):
        load_checkpoint(replay, path, '0' * 64)


def test_pooled_control_outputs_and_parameter_mismatch():
    torch = pytest.importorskip('torch')
    from mindscape.r56.native_domains import pack_problem
    from mindscape.r56.benchmarks import pooled_baseline, model_accounting
    graph = pack_problem(generate('train', 1)[0]['problem'])
    counts = []
    for kind in ('mlp', 'gru'):
        torch.manual_seed(17)
        model = pooled_baseline(kind)
        output = model(graph, stateless=True)
        assert output['math_digit_logits'].shape == (10,)
        assert output['coding_logits'].shape == (16,)
        assert output['physics_delta'].shape == (2,)
        counts.append(model_accounting(model)['parameter_count'])
    assert counts[0] != counts[1]  # Measured mismatch must not be concealed.


def test_holm_independent_reference_and_bootstrap_pairing():
    assert holm([.2, .001, .04]) == [.2, .003, .08]
    assert holm([.03, .03, .03]) == [.09, .09, .09]
    assert paired_bootstrap([0, 1, 0, 1], [0, 1, 0, 1], replicates=100)['ci95'] == [0.0, 0.0]
    assert paired_bootstrap([1, 0, 1], [0, 0, 1], replicates=100) == paired_bootstrap([1, 0, 1], [0, 0, 1], replicates=100)
    with pytest.raises(ValueError):
        holm([float('nan')])


def test_concept_controls_are_distinct_and_consistent():
    torch = pytest.importorskip('torch')
    from mindscape.r56.native_domains import pack_problem
    from mindscape.r56.benchmarks import controlled_graph
    graph = pack_problem(generate('train', 2)[1]['problem'])
    assert graph.concept_ids.numel() > 0
    assert controlled_graph(graph, 'graph-concept-off').concept_ids.numel() == 0
    assert torch.equal(controlled_graph(graph, 'graph-concept-shuffled').concept_ids,
                       (graph.concept_ids * 17 + 3) % 32)
    assert controlled_graph(graph, 'graph-concept-ablated').concept_ids.eq(0).all()
    assert torch.equal(graph.concept_ids, torch.tensor([3]))


def test_source_locks_cover_language_runtime_and_inference(tmp_path):
    from mindscape.r56.learning import source_locks
    paths = ['src/mindscape/r56/learning.py', 'src/mindscape/r56/language_model.py',
             'src/mindscape/r56/native_domains.py', 'src/mindscape/mind1/runtime.py',
             'src/mindscape/mind1/provider.py']
    for name in paths:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('source = 1')
    before = source_locks(tmp_path)
    assert set(before) == set(paths)
    (tmp_path / paths[3]).write_text('runtime = 2')
    assert source_locks(tmp_path) != before


def test_matched_capacity_baselines_observed_features_and_memory():
    torch = pytest.importorskip('torch')
    from dataclasses import replace
    from mindscape.r56.neural import R56Model
    from mindscape.r56.native_domains import pack_problem
    from mindscape.r56.benchmarks import (matched_baseline, matched_baseline_plan,
        model_accounting, observed_tokens)
    target = model_accounting(R56Model())['parameter_count']
    assert target == 44962
    plans = matched_baseline_plan(target)
    graph = pack_problem(generate('train', 3)[2]['problem'])
    original_tokens = observed_tokens(graph)
    assert original_tokens.shape[1] == 77
    assert not torch.equal(original_tokens, observed_tokens(replace(graph, action=graph.action + 1)))
    assert not torch.equal(original_tokens, observed_tokens(replace(graph, domain=(graph.domain + 1) % 3)))
    assert not torch.equal(original_tokens, observed_tokens(replace(graph, concept_ids=torch.tensor([12]))))
    for kind in ('mlp', 'rnn', 'gru'):
        model = matched_baseline(kind, target)
        assert model_accounting(model)['parameter_count'] == plans[kind]['parameter_count']
        assert plans[kind]['relative_gap'] <= .05
        result = model(graph, stateless=True)
        assert result['physics_delta'].shape == (2,)
        if kind != 'mlp':
            recurrent = model(graph, memory=result['memory'].detach(), stateless=False)
            assert not torch.equal(result['physics_delta'], recurrent['physics_delta'])
        assert torch.equal(result['physics_delta'], model(graph, memory=result['memory'], stateless=True)['physics_delta'])


def test_exposure_summary_reports_repeats_and_frozen_counts():
    from mindscape.r56.curriculum import exposure_summary
    rows = generate('train', 512)
    summary = exposure_summary(rows)
    assert summary['total_exposures'] == 512
    assert summary['domains']['math']['distinct_problems'] < summary['domains']['math']['exposures']
    assert summary['domains']['coding']['distinct_problems'] == summary['domains']['coding']['exposures']
    assert sum(v['exposures'] for v in summary['domains'].values()) == 512


def test_train_refuses_existing_checkpoint_or_failure(tmp_path):
    output = tmp_path / 'weights.json'
    output.write_text('preserve')
    with pytest.raises(FileExistsError):
        train(tmp_path, tmp_path, {}, {}, {}, output)
    assert output.read_text() == 'preserve'
    output.unlink()
    failure = tmp_path / 'weights.json.failure.json'
    failure.write_text('preserve failure')
    with pytest.raises(FileExistsError):
        train(tmp_path, tmp_path, {}, {}, {}, output)
    assert failure.read_text() == 'preserve failure'
