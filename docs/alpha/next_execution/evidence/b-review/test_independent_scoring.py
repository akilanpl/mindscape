import pytest
from mindscape.alpha_transfer.numerical_scoring import task_metrics, score_campaign

@pytest.mark.parametrize('valid',[2,-1,True,0.5,'1'])
def test_nonbinary_truth_rejected(valid):
    with pytest.raises(ValueError):
        task_metrics({'delta':[[1.]],'valid_probability':1.},{'delta':[[1.]],'valid':valid})

@pytest.mark.parametrize('delta',[[],[[]]])
def test_empty_denominator_rejected(delta):
    with pytest.raises(ValueError):
        task_metrics({'delta':delta,'valid_probability':1.},{'delta':delta,'valid':1})

def test_missing_nine_fits_rejected():
    with pytest.raises(ValueError):
        score_campaign({}, {}, [], {})

def test_nonfinite_prediction_rejected():
    with pytest.raises(ValueError):
        task_metrics({'delta':[[float('nan')]],'valid_probability':1.},{'delta':[[1.]],'valid':1})

def test_coordinate_error_is_macro_per_entity():
    result=task_metrics({'delta':[[3.,2.],[0.,0.]],'valid_probability':.8},{'delta':[[1.,2.],[0.,0.]],'valid':1})
    assert result['changed']==2 and result['unchanged']==0 and result['whole']==1 and result['valid']==1
