"""Pure schedule/integrity/statistics regressions; no model execution or optimization."""
import importlib.util
import json
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('r54_diag', Path(__file__).with_name('run.py'))
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)
SCHEDULE = json.loads(Path(__file__).with_name('schedule.json').read_text())


def test_fixed_schedule_and_order_balance():
    assert SCHEDULE['worker_orders'] == [['markov', 'recurrent'], ['recurrent', 'markov']] * 2
    for worker in range(4):
        rows = list(diag.sequence(SCHEDULE, worker))
        assert len(rows) == 24
        for kind in ('markov', 'recurrent'):
            own = [r for r in rows if r['kind'] == kind]
            assert [sum(r['phase'] == p for r in own) for p in ('cold', 'warmup', 'measured')] == [1, 3, 8]
            assert [sum(r['phase'] == 'measured' and r['slot'] == s for r in own) for s in (0, 1)] == [4, 4]


def test_distributions_keep_every_measurement():
    result = diag.distribution([1., 4., 2., 3.])
    assert result['samples_seconds'] == [1., 4., 2., 3.]
    assert result['mean'] == result['median'] == 2.5
    assert result['p95'] == pytest.approx(3.85)
    with pytest.raises(ValueError):
        diag.distribution([float('nan')])


def fixture_workers():
    workers = []
    for index in range(4):
        rows = [dict(r, total_seconds=.7 if r['phase'] == 'cold' else .2) for r in diag.sequence(SCHEDULE, index)]
        workers.append(dict(worker=index, status='DIAGNOSTIC_COMPLETE', base_order=SCHEDULE['worker_orders'][index],
                            records=rows, rss_limit_exceeded=False))
    return workers


def test_no_reclassification_and_incomplete_fail_closed():
    workers = fixture_workers()
    result = diag.summarize(workers, SCHEDULE)
    assert result['cold_start_support'] is True
    assert result['original_readiness'] == 'ENGINEERING_NO_GO'
    assert diag.summarize(workers[:-1], SCHEDULE)['status'] == 'INCOMPLETE_NATIVE_DIAGNOSTIC'
    workers[0]['records'].pop()
    with pytest.raises(ValueError, match='measurement schedule'):
        diag.summarize(workers, SCHEDULE)


def test_memory_and_slow_warm_trials_cannot_be_ignored():
    workers = fixture_workers()
    workers[0]['rss_limit_exceeded'] = True
    assert not diag.summarize(workers, SCHEDULE)['cold_start_support']
    workers[0]['status'] = 'DIAGNOSTIC_FAILED'
    failed = diag.summarize(workers[:1], SCHEDULE)
    assert failed['status'] == 'ENGINEERING_NO_GO' and failed['process_rss_limit_exceeded']
    workers = fixture_workers()
    for r in workers[0]['records']:
        if r['phase'] == 'measured':
            r['total_seconds'] = 1.0
    result = diag.summarize(workers, SCHEDULE)
    assert not result['cold_start_support']
    assert not result['all_worker_warm_p95_projection_within_limit']


def test_original_freeze_and_evidence_hashes():
    result = diag.inspect_freeze()
    assert result['locked_files'] == 19
    assert result['version'] == 'R5.4-v1'
    assert len(result['official_engineering_sha256']) == 64


def test_no_official_gate_campaign_or_optimizer_call():
    import ast
    tree = ast.parse(Path(__file__).with_name('run.py').read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                assert node.func.id not in ('fit', 'check', 'qualify', 'adapter', 'rollout')
            if isinstance(node.func, ast.Attribute):
                assert node.func.attr not in ('step', 'AdamW', 'SGD', 'Adam')
