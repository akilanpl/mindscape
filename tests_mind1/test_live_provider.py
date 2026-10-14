"""Opt-in-by-credential real end-to-end Groq qualification; no mocked success.

Run with a configured MIND1_API_KEY and Groq provider environment. Each case
uses the product Orchestrator, including real tool calls when selected.
"""
import json
import os
import re
import time
from dataclasses import dataclass

import pytest


@dataclass(frozen=True)
class Case:
    name: str
    domain: str
    tier: str
    question: str
    expected: int | None = None
    open_problem: bool = False


CASES = (
    Case('math-foundation', 'math', 'foundation', 'Explain why six plus seven equals the result, then compute it.', 13),
    Case('math-undergraduate', 'math', 'undergraduate', 'Compute the determinant of the matrix with rows [1,2] and [3,4]. Explain the calculation.', -2),
    Case('math-graduate', 'math', 'graduate', 'Give an informal proof that a compact metric space is complete. State the assumptions and distinguish informal from formal verification.'),
    Case('math-research', 'math', 'research', 'Prove or disprove the Riemann hypothesis. If it is unresolved, explicitly state that and offer a useful research reading strategy without fabricating citations.', open_problem=True),
    Case('coding-foundation', 'coding', 'foundation', 'For Python, predict the integer value of sum(range(5)). Explain whether you actually executed it.', 10),
    Case('coding-undergraduate', 'coding', 'undergraduate', 'Explain the time complexity and correctness invariant of merge sort, including why its merge operation is linear.'),
    Case('coding-graduate', 'coding', 'graduate', 'Analyze whether a distributed cache can guarantee linearizable reads during a network partition while remaining available. State failure-model assumptions and explain the tradeoff.'),
    Case('coding-research', 'coding', 'research', 'Give a universally correct polynomial-time algorithm for every NP-complete problem, or clearly explain the unresolved P versus NP barrier and useful research approaches. Do not claim a new solution.', open_problem=True),
    Case('physics-foundation', 'physics', 'foundation', 'A constant mass of 2 kg has acceleration 3 metres per second squared in an inertial frame. Find the net force and state assumptions.', 6),
    Case('physics-undergraduate', 'physics', 'undergraduate', 'An ideal undamped mass-spring oscillator has mass 1 kg and spring constant 9 newtons per metre. Compute the angular frequency in radians per second and discuss dimensions.', 3),
    Case('physics-graduate', 'physics', 'graduate', 'Derive the Euler-Lagrange equations from stationary action with fixed endpoint variations. State regularity and boundary assumptions; distinguish a derivation from experimental validation.'),
    Case('physics-research', 'physics', 'research', 'Provide an experimentally established complete quantum theory of gravity, or explicitly explain why no such established theory is currently available and outline testable research approaches without fabricated sources.', open_problem=True),
)


def _structured_answer(text):
    # A parse failure is explicit; substring matching is never a correctness score.
    blocks = re.findall(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.S)
    candidates = blocks + [text.strip()]
    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            value = decoder.decode(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(value, dict):
            return value
    return None


@pytest.mark.parametrize('case', CASES, ids=lambda case: case.name)
def test_live_groq_product(case, record_property):
    key_present = bool(os.environ.get('MIND1_API_KEY'))
    record_property('api_key_present', key_present)
    if not key_present:
        pytest.skip('MIND1_API_KEY absent; live Groq completion not attempted or simulated')
    if os.getenv('MIND1_MODEL') != 'openai/gpt-oss-120b':
        pytest.skip('Selected live model openai/gpt-oss-120b is not configured')
    if os.getenv('MIND1_PROVIDER_URL', '').rstrip('/') != 'https://api.groq.com/openai/v1':
        pytest.skip('Selected live Groq HTTPS endpoint is not configured')
    if os.getenv('MIND1_REASONING_EFFORT') != 'high':
        pytest.skip('Selected high reasoning effort is not configured')
    from mindscape.mind1.orchestrator import Orchestrator
    prompt = case.question
    if case.expected is not None:
        prompt += '\nReturn a single JSON object with answer_value (a number), explanation, assumptions, and execution_status. Do not put any text outside the object.'
    began = time.monotonic()
    response = Orchestrator().chat({'message': prompt, 'domain': case.domain,
                                    'mode': 'research' if case.tier == 'research' else 'deep'})
    provenance = response.get('provenance', {})
    calls = provenance.get('calls', [])
    assessment = {'case': case.name, 'tier': case.tier, 'domain': case.domain,
        'status': response.get('status'), 'api_key_present': key_present,
        'wall_seconds': round(time.monotonic() - began, 3),
        'provider_receipts': calls, 'provider': provenance.get('provider'),
        'r56_used': provenance.get('r56_used'),
        'human_review_required': case.expected is None,
        'research_correctness_score': None,
        'cost_usd': sum(c['estimated_cost_usd'] for c in calls)
            if calls and all(c.get('estimated_cost_usd') is not None for c in calls) else None}
    if case.expected is not None:
        parsed = _structured_answer(response.get('answer', ''))
        actual = parsed.get('answer_value') if parsed else None
        assessment.update(objective_expected=case.expected, objective_actual=actual,
                          objective_match=type(actual) in (int, float) and actual == case.expected)
    if case.open_problem:
        # Evidence for human review only; keywords cannot establish honest correctness.
        answer = response.get('answer', '').lower()
        assessment['uncertainty_language_observed'] = any(
            phrase in answer for phrase in ('unresolved', 'unsolved', 'open problem', 'not established', 'no experimentally', 'not currently', 'not known'))
    record_property('assessment', json.dumps(assessment, ensure_ascii=False))
    print('\nLIVE_ASSESSMENT ' + json.dumps(assessment, ensure_ascii=False))
    assert response.get('status') == 'completed', assessment
    assert calls, 'No real external provider receipt: cannot qualify live provider'
    assert provenance.get('r56_used') is False
    assert isinstance(response.get('answer'), str) and response['answer'].strip()
    if case.expected is not None:
        assert assessment['objective_match'], assessment
    # Graduate/research responses are recorded for expert review, never auto-scored correct.
    print('LIVE_ANSWER ' + json.dumps({'case': case.name, 'answer': response['answer']}, ensure_ascii=False))
