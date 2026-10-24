"""One recovery intervention: masked mean of encoder recurrent states.

The R6 baseline remains imported unchanged. No solver, learned parameter, output
vocabulary, decoder, loss or target mapping is added. Baseline configuration
uses its original last-state encoding. Pooled configuration uses the same GRU
states but provides a direct gradient route from every valid byte position.
"""
import torch
from torch import nn
from research.r6.architectures.model import R6Model

def summarize(outputs,state,lengths,mode):
    if mode=='last':return state[-1]
    if mode!='masked-mean':raise ValueError('Unknown sequence pooling')
    padded,_=nn.utils.rnn.pad_packed_sequence(outputs,batch_first=True)
    mask=torch.arange(padded.shape[1],device=padded.device)[None,:]<lengths.to(padded.device)[:,None]
    return (padded*mask.unsqueeze(-1)).sum(1)/lengths.to(padded.device).unsqueeze(-1)

class RecoveryModel(R6Model):
    def encode(self,tokens,lengths,features,domains):
        mode=self.config.get('sequence_pooling','last')
        if mode=='last':return super().encode(tokens,lengths,features,domains)
        if mode!='masked-mean':raise ValueError('Unknown sequence pooling')
        # The baseline implementation downstream of its encoder is reused
        # without copying or altering graph, fusion, decoder or physics code.
        # A scoped forward hook replaces only the GRU's returned summary state.
        def pool(module,args,result):
            outputs,state=result
            return outputs,summarize(outputs,state,lengths,mode).unsqueeze(0)
        handle=self.sequence.register_forward_hook(pool)
        try:return super().encode(tokens,lengths,features,domains)
        finally:handle.remove()

def make_model(config):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(config.get('seed',61117)))
        return RecoveryModel(config)
