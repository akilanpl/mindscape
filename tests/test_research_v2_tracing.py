"""Telemetry must preserve real decisions and distinguish derived observations."""
import importlib.util
from pathlib import Path

from mindscape.data.backends import get_backend
from mindscape.models.learned import MindscapeModel
from mindscape.models.numpy_backend import NumpyMLP

spec = importlib.util.spec_from_file_location('numeric_tracing', Path(__file__).resolve().parents[1]/'scripts/research_v2/tracing.py')
tracing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tracing)


def test_observer_preserves_actual_answer_and_trajectory_and_restores_backend():
    backend = NumpyMLP([4], hidden=8, seed=9)
    example = get_backend('integer_multiplication').example(
        {'operands': [17, 8], 'source': 'test_fixture', 'kind': 'observation'}, 1, 'test')
    original_logits = backend.logits
    expected = MindscapeModel(backend).predict(example.view('experiential'))
    trace = tracing.trace_case(backend, example)
    actual = trace['result']['prediction']
    assert actual['answer'] == expected.answer
    assert actual['trajectory'] == tracing.plain(expected.trajectory)
    assert backend.logits == original_logits
    forwards = [e for e in trace['events'] if e['computation'] == 'learned neural forward']
    assert len(forwards) == expected.model_calls
    assert all(e['observed']['input_shape'] == [1, 52] for e in forwards)
    assert all('Recomputed' in e['observed']['hidden_provenance'] for e in forwards)
    assert trace['exact_model_input'] == example.view('experiential')
    assert 'target_answer' not in trace['exact_model_input']


def test_feature_intervention_is_explicit_and_does_not_mutate_model_weights():
    import numpy as np

    backend = NumpyMLP([4], hidden=8, seed=9)
    weights = {k: v.copy() for k, v in backend.weights.items()}
    example = get_backend('integer_multiplication').example(
        {'operands': [17, 8], 'source': 'test_fixture', 'kind': 'observation'}, 1, 'test')
    trace = tracing.trace_case(backend, example, state_features=False, previous_action_features=False)
    forwards = [e['observed'] for e in trace['events'] if e['computation'] == 'learned neural forward']
    assert forwards and all(not any(e['features_used'][0][18:]) for e in forwards)
    assert all(np.array_equal(v, weights[k]) for k, v in backend.weights.items())
