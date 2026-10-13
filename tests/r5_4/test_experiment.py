"""Nontraining regression checks; no optimizer steps or neural OOD evaluation."""
import ast
import copy
import json
from pathlib import Path
import pytest
import torch
from mindscape.r5_4.data import (generate, qualify, reference, operand, HELD_PAIRS, ROOT)
from mindscape.r5_4.inference import Operands, rollout
from mindscape.r5_4.model import Transition, COUNTS, adapter, features
from mindscape.r5_4.training import objective
from mindscape.r5_4.evaluate import predict_rows
from mindscape.r5_4.audit import validate_record, wilson, paired, independent_reference, audit


def test_manifest_reproducibility_and_headroom():
    result = qualify()
    assert result['status'] == 'DATA_READY'
    assert result['unique_contexts']['train'] == 184
    assert result['memory_correct'] == dict(length_known=48, long_carry_known=48, mixed_known=48, unseen_short=0)
    assert result['indirect_training_exposures'] == 0
    tasks = generate()
    ids = [t['task_id'] for t in tasks]
    numbers = [tuple(sorted((t['left'], t['right']))) for t in tasks]
    assert len(set(ids)) == len(set(numbers)) == 1280


def test_reference_all_200_contexts_and_boundaries():
    for a in range(10):
        for b in range(10):
            for c in (0, 1):
                # Incoming carry at second column controlled by first-column 0+0 or 9+9.
                task = dict(task_id='synthetic', left=f'{a}9' if c else f'{a}0', right=f'{b}9' if c else f'{b}0')
                task['left'], task['right'] = str(int(task['left'])), str(int(task['right']))
                assert reference(operand(task)) == independent_reference(task)['trace']
    for a, b in [('0', '0'), ('99999999', '1'), ('1', '99999999'), ('10', '0')]:
        t = dict(task_id='synthetic', left=a, right=b)
        assert reference(operand(t)) == independent_reference(t)['trace']


def test_held_pairs_impossible_for_any_training_carry():
    for task in generate()[:1024]:
        for a, b, *_ in reference(operand(task)):
            assert (a, b) not in HELD_PAIRS


@pytest.mark.parametrize('kind', COUNTS)
def test_architecture_and_cpu_backward_without_update(kind, monkeypatch):
    torch.manual_seed(11)
    model = Transition(kind)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    def forbidden(*args, **kwargs):
        raise AssertionError('Optimizer forbidden in regression tests')
    monkeypatch.setattr(torch.optim, 'AdamW', forbidden)
    tasks = generate()[:32]  # training inputs only
    for update in (1, 401):
        model.zero_grad(set_to_none=True)
        loss, exposures = objective(model, tasks, update)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm() > 0 for p in model.parameters())
        assert all((a, b) not in HELD_PAIRS for a, b, c in exposures)
    assert sum(p.numel() for p in model.parameters()) == COUNTS[kind]
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())


def test_features_contain_only_observations():
    x = features(torch.tensor([9]), torch.tensor([8]), torch.tensor([1]))
    assert x.shape == (1, 22) and x.dtype == torch.float32 and x.sum() == 3
    tree = ast.parse(Path('src/mindscape/r5_4/model.py').read_text())
    assert not any(isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mod, ast.FloorDiv)) for node in ast.walk(tree))


@pytest.mark.parametrize('kind', COUNTS)
def test_neural_answer_blind_inference_training_input_only(kind, monkeypatch):
    import mindscape.r5_4.data as data
    monkeypatch.setattr(data, 'reference', lambda *args: (_ for _ in ()).throw(AssertionError('Label access')))
    model = Transition(kind)
    task = generate()[0]
    task['answer'] = object()
    task['labels'] = object()
    rows = list(predict_rows([task], adapter(model), 'synthetic-neural', 'neural', 11))
    assert rows[0]['condition'] == 'neural' and rows[0]['seed'] == 11
    validate_record(rows[0], task)


def synthetic_row():
    task = dict(task_id='synthetic', split='test', family='synthetic', left='99', right='1')
    row = next(predict_rows([task], lambda a, b, c, h: (0, 1, None), 'synthetic-neural', 'neural', 11))
    return task, row


def test_feedback_is_predicted_even_if_wrong():
    calls = []
    def predictor(a, b, c, h):
        calls.append(c)
        return 7, 1, None
    result = rollout(Operands('synthetic', '100', '0'), predictor)
    assert calls == [0, 1, 1] and result['answer'] == '1777'


def test_shadowing_and_base_replay_separate_from_diagnostics():
    task, row = synthetic_row()
    row['context_diagnostics'] = {'unrelated': 123}
    validate_record(row, task)
    assert row['condition'] == 'neural' and row['model_id'] == 'synthetic-neural'


@pytest.mark.parametrize('tamper', ['carry', 'answer', 'length', 'dtype', 'family', 'metadata'])
def test_replay_fail_closed(tamper):
    task, row = synthetic_row()
    if tamper == 'carry':
        row['trace'][1]['carry_in'] = 0
    elif tamper == 'answer':
        row['answer'] = '999'
    elif tamper == 'length':
        row['trace'].pop()
    elif tamper == 'dtype':
        row['trace'][0]['digit'] = 0.0
    elif tamper == 'family':
        row['family'] = 'other'
    else:
        del row['tensor_metadata']['trace_shape']
    with pytest.raises((ValueError, KeyError)):
        validate_record(row, task)


def test_float32_score_precision_not_reused_as_integer_truth():
    lower, upper = wilson(192, 192)
    assert 0.98 < lower < 1 and upper == pytest.approx(1)
    assert paired([True] * 48, [False] * 48)['p_one_sided'] == 2**-48
    assert paired([True, False], [False, True])['p_one_sided'] == 0.75
    assert type(wilson(47, 48)[0]) is float
    with pytest.raises(ValueError):
        wilson(0, 0)


def test_incomplete_campaign_rejected_before_labels(tmp_path, monkeypatch):
    import mindscape.r5_4.audit as module
    monkeypatch.setattr(module, 'verify_lock', lambda: {})
    with pytest.raises(ValueError, match='Incomplete evidence'):
        audit(tmp_path)


def test_no_implicit_training_authorization(tmp_path, monkeypatch):
    from mindscape.r5_4.campaign import run
    with pytest.raises(RuntimeError, match='authorization'):
        run(tmp_path / 'campaign')
    assert not (tmp_path / 'campaign').exists()


def test_engineering_no_fallback_no_optimizer(monkeypatch):
    from mindscape.r5_4.engineering import check
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: False)
    result = check(generate()[:32])
    assert result['status'] == 'ENGINEERING_NO_GO'
    assert result['optimizer_updates'] == 0 and result['cpu_fallback'] is False


def test_archive_verification_no_extraction_and_tamper(tmp_path):
    import hashlib
    import io
    import tarfile
    from mindscape.r5_4.preservation import verify_archive
    raw = b'frozen evidence'
    receipt = json.dumps({'schema': 'R5.4-preservation-v1', 'sha256': {
        'campaign/evidence.json': hashlib.sha256(raw).hexdigest()}}).encode()
    def write(path, name, payload):
        with tarfile.open(path, 'w:gz') as bundle:
            for key, value in [(name, payload), ('PRESERVATION.json', receipt)]:
                item = tarfile.TarInfo(key)
                item.size = len(value)
                bundle.addfile(item, io.BytesIO(value))
    archive = tmp_path / 'good.tar.gz'
    write(archive, 'campaign/evidence.json', raw)
    assert verify_archive(archive)['status'] == 'VERIFIED'
    write(archive, 'campaign/evidence.json', b'tampered')
    with pytest.raises(ValueError, match='hash'):
        verify_archive(archive)
    write(archive, '../escape', raw)
    with pytest.raises(ValueError, match='Unsafe'):
        verify_archive(archive)
    assert not (tmp_path / 'campaign').exists()


def test_checkpoint_attribution_and_tensor_failures_without_training(tmp_path):
    from mindscape.r5_4.audit import checkpoint_audit
    from mindscape.r5_4.data import digest
    entries = []
    for kind in COUNTS:
        for seed in (11, 23, 47):
            model = Transition(kind)  # initialization only; no forward, backward or optimizer
            mid = f'r54-{kind}-neural-seed-{seed}-u800'
            state = {i: {'step': torch.tensor(800.), 'exp_avg': torch.zeros_like(p),
                         'exp_avg_sq': torch.zeros_like(p)} for i, p in enumerate(model.parameters())}
            payload = dict(model_id=mid, kind=kind, seed=seed, updates=800, parameters=COUNTS[kind],
                model=model.state_dict(), optimizer={'state': state, 'param_groups': [
                    {'lr': .003, 'weight_decay': .0001, 'betas': (.9, .999)}]}, exposure=[(0, 0, 0)],
                tensor_metadata={k: dict(dtype='torch.float32', shape=list(v.shape), device='mps:0')
                                 for k, v in model.state_dict().items()})
            path = tmp_path / f'{mid}.pt'
            torch.save(payload, path)
            entries.append(dict(path=path.name, sha256=digest(path), model_id=mid, kind=kind, seed=seed))
            (tmp_path / f'{mid}.progress.jsonl').write_text(''.join(json.dumps(
                {'update': i, 'loss': 1.0, 'dtype': 'torch.float32'}) + '\n' for i in range(1, 801)))
    seal = {'checkpoints': entries}
    assert len(checkpoint_audit(tmp_path, seal)) == 6  # synthetic evidence fixture, no learned claim
    with pytest.raises(ValueError, match='fit count'):
        checkpoint_audit(tmp_path, {'checkpoints': entries[:-1]})
    wrong = copy.deepcopy(seal)
    wrong['checkpoints'][0]['seed'] = 47
    with pytest.raises(ValueError, match='attribution'):
        checkpoint_audit(tmp_path, wrong)
    path = tmp_path / entries[0]['path']
    payload = torch.load(path, weights_only=True)
    first_key = next(iter(payload['model']))
    payload['model'][first_key].flatten()[0] = float('nan')
    torch.save(payload, path)
    entries[0]['sha256'] = digest(path)
    with pytest.raises(ValueError, match='nonfinite'):
        checkpoint_audit(tmp_path, seal)


def test_historical_float32_interval_score_roundtrip_regression():
    import numpy as np
    from mindscape.r5_3_physics.uncertainty import interval_score
    from mindscape.r5_3_recovery.numerics import native_bounds, score_tolerance, acceptable
    lo, hi, truth = np.array([.1], np.float32), np.array([.3], np.float32), np.array([.2], np.float64)
    native = float(interval_score(truth, lo, hi).mean())
    decoded = json.loads(json.dumps({'lo': lo.tolist(), 'hi': hi.tolist()}))
    replay_lo, replay_hi = np.asarray(decoded['lo']), np.asarray(decoded['hi'])
    replay = float(interval_score(truth, replay_lo, replay_hi).mean())
    assert native != replay
    assert acceptable(native, replay, score_tolerance(truth, replay_lo, replay_hi))
    repaired_lo, repaired_hi = native_bounds(replay_lo, replay_hi)
    assert float(interval_score(truth, repaired_lo, repaired_hi).mean()) == native
    assert not acceptable(native, native + 1e-5, score_tolerance(truth, replay_lo, replay_hi))
