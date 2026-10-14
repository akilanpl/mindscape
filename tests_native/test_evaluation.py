from mindscape.r56.evaluation import axis_datasets, evaluate_episode, transfer_preparation
import pytest


def test_composition_preserves_wrong_carry_and_error_accumulation():
    episode = axis_datasets()['episodes'][0]
    seen=[]
    def wrong(problem):
        seen.append(problem.copy())
        return {'math_digit': 0, 'math_carry': 0}
    result = evaluate_episode(episode, wrong)
    assert not result['metric']['exact_composition']
    assert seen[1]['incoming'] == 0
    assert all(not r['repaired'] for r in result['records'])


def test_physics_error_propagates_without_reference_reset():
    episode = next(e for e in axis_datasets()['episodes'] if e['domain']=='physics')
    seen=[]
    def wrong(problem):
        seen.append(problem.copy())
        return {'physics_delta':[0.,0.]}
    result=evaluate_episode(episode,wrong)
    assert seen[1]['velocity'] == episode['initial_velocity']
    assert result['metric']['final_mse'] > 0
    assert result['records'][1]['accumulated_squared_error'] > result['records'][0]['accumulated_squared_error']


def test_transfer_preparation_copies_only_core_and_preserves_target_heads():
    import torch
    from mindscape.r56.neural import R56Model
    source=R56Model().state_dict(); target=R56Model().state_dict()
    result,receipt=transfer_preparation(source,target,['a'*64],['b'*64],['c'*64],'math','physics')
    assert receipt['training_executed'] is False
    for key in receipt['copied_core_keys']:
        assert torch.equal(result[key],source[key])
    for key in receipt['unchanged_head_keys']:
        assert torch.equal(result[key],target[key])
    with pytest.raises(ValueError):
        transfer_preparation(source,target,['a'*64],['a'*64],['c'*64],'math','physics')
