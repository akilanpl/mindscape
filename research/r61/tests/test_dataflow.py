import torch
import torch.nn.functional as F
from research.r6.common import encode_batch,encode_answer,EOS,BOS
from research.r6.architectures.model import make_model

def test_codec_preserves_syntax_operands_order_and_pairing():
    rows=[{'domain':'coding','input':s,'features':[],'target':{'answer':a}} for s,a in [('def f(n):\n return n+2\nr=f(4)','6'),('def f(n):\n return n-2\nr=f(4)','2')]]
    b=encode_batch(rows,'cpu')
    for i,r in enumerate(rows):
        assert bytes((b['tokens'][i,:b['lengths'][i]]-1).tolist()).decode('ascii')==r['input']
        n=len(encode_answer(r['target']['answer']))
        assert b['target_output'][i,:n].tolist()==encode_answer(r['target']['answer'])
        assert b['target_input'][i,0]==BOS
        assert b['target_input'][i,1:n].tolist()==b['target_output'][i,:n-1].tolist()
        assert b['target_output'][i,n-1]==EOS
    assert not torch.equal(b['tokens'][0],b['tokens'][1])

def test_loss_ignores_padding_and_matches_indexed_nll():
    logits=torch.arange(2*4*14,dtype=torch.float32).reshape(2,4,14)/100
    target=torch.tensor([[5,2,0,0],[3,5,2,0]])
    ce=F.cross_entropy(logits.reshape(-1,14),target.flatten(),ignore_index=0)
    manual=-F.log_softmax(logits,-1).gather(-1,target.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(ce,manual[target!=0].mean())
    altered=logits.clone();altered[target==0]=100*torch.arange(14).float()
    assert torch.equal(ce,F.cross_entropy(altered.reshape(-1,14),target.flatten(),ignore_index=0))
