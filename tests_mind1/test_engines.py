import math
import pytest

from mindscape.mind1.engines import capabilities, run_tool, tool_specs


def result(name, **args):
    response = run_tool(name, args)
    assert response['ok'], response
    return response['result']


def test_schema_matches_dispatch():
    names = [s['function']['name'] for s in tool_specs()]
    assert len(set(names)) == 5
    assert not run_tool('unknown', {})['ok']
    assert capabilities()['host_source_execution'] is False


@pytest.mark.skipif(not capabilities()['symbolic_math'], reason='optional sympy absent')
def test_symbolic_calculus_and_equation():
    assert result('math_calculate', expression='x**3+2*x', operation='differentiate')['result'] == '3*x**2 + 2'
    assert result('math_calculate', expression='x**2', operation='integrate')['result'] == 'x**3/3'
    assert result('math_calculate', expression='x**2-4', operation='solve')['solutions'] == ['-2', '2']
    assert float(result('math_calculate', expression='sin(pi/2)+x', values={'x': 2})['result']) == 3
    assert result('math_matrix', matrix=[[1, 2], [3, 4]])['result'] == '-2'
    assert result('math_matrix', matrix=[[1, 2], [2, 4]], operation='rank')['result'] == 1


@pytest.mark.parametrize('expression', ["__import__('os').system('echo bad')", 'x.__class__', '[x for x in []]', 'x**x', '2**1000000', '((999999999999**12)**12)**12', '1/0'])
def test_symbolic_rejects_unsafe_or_unbounded(expression):
    assert run_tool('math_calculate', {'expression': expression})['ok'] is False


def test_projectile_matches_analytic_solution():
    r = result('physics_simulate', model='projectile', parameters={'speed': 10, 'angle_degrees': 45, 'gravity': 10, 'duration': 10})
    assert r['range'] == pytest.approx(10)
    assert r['impact_time'] == pytest.approx(math.sqrt(2))
    assert r['samples'][-1]['y'] == pytest.approx(0, abs=1e-10)


def test_oscillator_energy_and_phase():
    r = result('physics_simulate', model='oscillator', parameters={'duration': 2*math.pi}, steps=500)
    assert r['samples'][-1]['position'] == pytest.approx(1, abs=1e-7)
    assert r['samples'][-1]['energy'] == pytest.approx(.5, abs=1e-7)
    assert not run_tool('physics_simulate', {'model': 'oscillator', 'steps': 1})['ok']


def test_rc_time_constant():
    r = result('physics_simulate', model='rc_circuit', parameters={'resistance': 1000, 'capacitance': .001, 'voltage': 5, 'duration': 1}, steps=1)
    assert r['samples'][-1]['voltage'] == pytest.approx(5*(1-math.exp(-1)))


def test_code_analysis_does_not_execute(tmp_path):
    marker = tmp_path / 'should-not-exist'
    r = result('coding_analyze', source=f"open({str(marker)!r}, 'w').write('bad')\neval('1')")
    assert not marker.exists()
    assert r['findings'][0]['line'] == 2
    assert r['executed'] is False
    assert result('coding_analyze', source='def broken(', language='python')['syntax_valid'] is False
    r = result('coding_analyze', source='unsafe { dangerous(); }', language='rust')
    assert r['syntax_valid'] is None and r['findings']


def test_wasi_fails_closed_when_missing(monkeypatch, tmp_path):
    monkeypatch.setenv('MIND1_WASI_RUNTIME', str(tmp_path))
    r = run_tool('coding_execute', {'source': 'print(42)'})
    assert not r['ok'] and r['error']['code'] == 'unavailable'


def test_reject_invalid_inputs():
    for name, args in [('physics_simulate', {'model': 'projectile', 'parameters': {'speed': float('nan')}}), ('coding_analyze', {'source': 'x'*64001}), ('math_matrix', {'matrix': [[1], [1, 2]]})]:
        assert not run_tool(name, args)['ok']
