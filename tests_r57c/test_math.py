import torch
from mindscape.r57.model import make_model as original
from mindscape.r57c.math_model import make_model


def test_other_weights_preserved_and_math_sensitivity():
    for condition in ('graph-concept-on','graph-concept-off','matched-gru'):
        old=original(condition,17);new=make_model(condition,17)
        assert all(torch.equal(v,new.state_dict()[k]) for k,v in old.state_dict().items())
        p={'domain':'math','operation':'add_digit','left':2,'right':4,'incoming':0,'concept_ids':[0]}
        a=new.predict(p);b=new.predict({**p,'left':3})
        assert not torch.equal(a['math_digit_logits'],b['math_digit_logits'])
        loss=a['math_digit_logits'].sum()+a['math_carry_logits'].sum();loss.backward()
        assert new.scalar_arithmetic.weights.grad is not None
