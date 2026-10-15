"""CPU verification only; native preflight remains a separate operator receipt."""
import torch
from mindscape.r56.neural import OneHotEmbedding,R56Model

def test_small_embedding_parameter_schema_and_numerics():
    torch.manual_seed(17)
    for vocab in (8,16):
        a=torch.nn.Embedding(vocab,48);b=OneHotEmbedding(vocab,48)
        b.load_state_dict(a.state_dict())
        for indices in (torch.arange(vocab),torch.arange(256)%vocab):
            a.zero_grad();b.zero_grad();cot=torch.randn(len(indices),48)
            x=a(indices);y=b(indices)
            torch.testing.assert_close(x,y,atol=0,rtol=0)
            x.backward(cot);y.backward(cot)
            torch.testing.assert_close(a.weight.grad,b.weight.grad,atol=2e-5,rtol=2e-4)
        assert a.state_dict().keys()==b.state_dict().keys()
    m=R56Model()
    assert sum(p.numel() for p in m.parameters())==44962
    assert type(m.types) is OneHotEmbedding and type(m.relations) is OneHotEmbedding
    assert type(m.concepts) is torch.nn.Embedding
