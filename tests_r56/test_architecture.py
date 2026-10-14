import pytest
from mindscape.r56 import (Action, Composer, State, Protocol, default_registry,
                           encode_state, evaluate_predictions, STATUS)


def test_carry_exhaustive_bounded():
    composer = Composer(default_registry())
    for base in range(2, 37):
        for total in range(100):
            before = State.create('math', total=total, digit=0, carry=0)
            after, records = composer.run(before, [Action.create('math.carry', base=base)])
            assert after.mapping()['digit'] + base * after.mapping()['carry'] == total
            assert records[0].verified


def test_composition_impulses_conserve_momentum():
    state = State.create('physics', mass=2.0, velocity=3.0)
    actions = [Action.create('physics.impulse', impulse=x) for x in (4.0, -2.0, 8.0)]
    after, records = Composer(default_registry()).run(state, actions)
    assert after.mapping()['velocity'] == 8.0
    assert all(r.verified for r in records)
    assert records[1].before.identity == records[0].proposed.identity
    assert evaluate_predictions(records)['scientific_status'] == STATUS


def test_false_prediction_preserved_and_rejected():
    before = State.create('math', total=19, digit=0, carry=0)
    proposed = State.create('math', total=19, digit=8, carry=1)
    evidence = Composer(default_registry()).verify(before, Action.create('math.carry', base=10), proposed, 'untrained-neural')
    assert not evidence.verified
    assert evidence.proposed == proposed
    assert evidence.as_dict()['source'] == 'untrained-neural'


def test_ast_rename_and_capture_rejection():
    composer = Composer(default_registry())
    state = State.create('coding', source='x = x + 1')
    after, records = composer.run(state, [Action.create('coding.rename_assignment', old='x', new='y')])
    assert after.mapping()['source'] == 'y = y + 1'
    assert records[0].verified
    with pytest.raises(ValueError):
        composer.run(State.create('coding', source='x = y + 1'), [Action.create('coding.rename_assignment', old='x', new='y')])


@pytest.mark.parametrize('state,action', [
    (State.create('math', total=True, digit=0, carry=0), Action.create('math.carry', base=10)),
    (State.create('math', total=9, digit=0, carry=0), Action.create('math.carry', base=1)),
    (State.create('physics', mass=0.0, velocity=3.0), Action.create('physics.impulse', impulse=1.0)),
    (State.create('physics', mass=2.0, velocity=float('nan')), Action.create('physics.impulse', impulse=1.0)),
])
def test_invalid_preconditions(state, action):
    with pytest.raises(ValueError):
        Composer(default_registry()).run(state, [action])


def test_encoder_budget_and_domain_boundaries():
    assert encode_state(State.create('physics', mass=2.0, velocity=1.0), ('mass', 'velocity')) == (2.0, 1.0)
    with pytest.raises(ValueError):
        encode_state(State.create('coding', source='x = 1'), ('source',))
    with pytest.raises(ValueError):
        Composer(default_registry()).run(State.create('physics', mass=2.0, velocity=1.0), [Action.create('math.carry', base=10)])
    with pytest.raises(ValueError):
        Composer(default_registry()).run(State.create('math', total=1, carry=0, digit=0), [Action.create('math.carry', base=10)], max_steps=0)


def test_manifest_is_content_addressed_not_authorization():
    with pytest.raises(ValueError):
        Protocol().freeze()
    first = Protocol(dataset_ids=('a' * 64,)).freeze()
    assert first == Protocol(dataset_ids=('a' * 64,)).freeze()
    assert first['sha256'] != Protocol(dataset_ids=('b' * 64,)).freeze()['sha256']
    assert first['authorization'] == 'NOT_AUTHORIZED_BY_FREEZE'
    assert evaluate_predictions([])['verified_fraction'] is None


def test_coding_malformed_proposal_is_failed_evidence():
    composer = Composer(default_registry())
    evidence = composer.verify(State.create('coding', source='x = x + 1'),
        Action.create('coding.rename_assignment', old='x', new='y'),
        State.create('coding', source='not valid !!!'), source='untrained-neural')
    assert not evidence.verified
    with pytest.raises(ValueError):
        composer.run(State.create('coding', source='x = 1'),
            [Action.create('coding.rename_assignment', old='x', new='for')])
