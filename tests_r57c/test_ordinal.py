import torch
from mindscape.r57.model import make_model as old
from mindscape.r57c.ordinal_model import make_model,domain_loss


def test_preserved_other_weights_trainable_scalar_and_label_geometry():
    for condition in ('graph-concept-on','graph-concept-off','matched-gru'):
        model=make_model(condition,17);original=old(condition,17)
        assert all(torch.equal(v,model.state_dict()[k]) for k,v in original.state_dict().items())
        p={'domain':'math','operation':'borrow_digit','left':3,'right':7,'incoming':0,'concept_ids':[1]}
        out=model.predict(p);loss=domain_loss(out,{'problem':p,'labels':{'math_digit':6,'math_carry':1}},torch,'cpu')
        loss.backward();assert model.ordinal_weights.grad is not None
        assert all(parameter.grad is None for parameter in model.core.parameters())
        assert out['math_digit_logits'].shape==(10,) and out['math_carry_logits'].shape==(2,)
