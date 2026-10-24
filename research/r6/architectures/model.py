"""Native R6 hybrid: character sequence context and typed raw-property messages.

No arithmetic executor, program interpreter or physical equation is used here.
"""
import re
import torch
from torch import nn
from research.r6.common import FEATURE_SCALES,MAX_OUTPUT,BOS,EOS,MINUS,encode_batch

class R6Model(nn.Module):
    def __init__(self,config):
        super().__init__();self.config=dict(config);w=int(config.get('width',48))
        self.architecture=config.get('architecture','native-relational')
        if self.architecture not in ('native-relational','recurrent'):raise ValueError('Unknown architecture')
        self.input_embedding=nn.Embedding(129,24,padding_idx=0)
        self.sequence=nn.GRU(24,w,batch_first=True)
        self.domain_embedding=nn.Embedding(3,12)
        if self.architecture=='native-relational':
            self.property_types=nn.Embedding(17,w)
            self.property_value=nn.Linear(1,w)
            self.message=nn.Sequential(nn.Linear(2*w,w),nn.Tanh(),nn.Linear(w,w))
            self.node_update=nn.GRUCell(w,w)
            adjacency=torch.eye(17)
            adjacency[:5,:]=1;adjacency[:,:5]=1
            for i in range(3):
                start=5+4*i;adjacency[start:start+4,start:start+4]=1
                if i:adjacency[start:start+4,start-4:start]=1
            adjacency/=adjacency.sum(-1,keepdim=True)
            self.register_buffer('adjacency',adjacency)
        else:self.numeric_recurrent=nn.GRU(1,w,batch_first=True)
        self.register_buffer('feature_scales',torch.tensor(FEATURE_SCALES))
        self.fusion=nn.Sequential(nn.Linear(2*w+12,w),nn.Tanh())
        self.output_embedding=nn.Embedding(14,16,padding_idx=0)
        self.decoder=nn.GRU(16,w,batch_first=True)
        self.text_head=nn.Linear(w,14)
        self.physics_head=nn.Sequential(nn.Linear(w,w),nn.Tanh(),nn.Linear(w,4))
        self.register_buffer('target_mean',torch.tensor(config.get('physics_target_mean',[0.]*4),dtype=torch.float32))
        self.register_buffer('target_scale',torch.tensor(config.get('physics_target_scale',[1.]*4),dtype=torch.float32))
        if not bool((self.target_scale>0).all()):raise ValueError('Positive training-only target scales required')
        self.last_activation_diagnostics={}

    def encode(self,tokens,lengths,features,domains):
        embedded=self.input_embedding(tokens)
        packed=nn.utils.rnn.pack_padded_sequence(embedded,lengths.cpu(),batch_first=True,enforce_sorted=False)
        _,state=self.sequence(packed);sequence=state[-1]
        numeric=features/self.feature_scales
        if self.architecture=='native-relational':
            nodes=self.property_value(numeric.unsqueeze(-1))+self.property_types.weight.unsqueeze(0)
            for _ in range(3):
                neighbours=torch.matmul(self.adjacency,nodes)
                message=self.message(torch.cat((nodes,neighbours),dim=-1))
                nodes=self.node_update(message.reshape(-1,nodes.shape[-1]),nodes.reshape(-1,nodes.shape[-1])).reshape_as(nodes)
            properties=nodes.mean(1)
        else:
            _,state=self.numeric_recurrent(numeric.unsqueeze(-1));properties=state[-1]
        properties=properties*(domains==1).float().unsqueeze(-1)
        context=self.fusion(torch.cat((sequence,properties,self.domain_embedding(domains)),dim=-1))
        self.last_activation_diagnostics={'context_abs_mean':float(context.detach().abs().mean()),
                                          'context_std':float(context.detach().std()),
                                          'property_abs_mean':float(properties.detach().abs().mean())}
        return context

    def forward(self,tokens,lengths,features,domains,decoder_inputs):
        context=self.encode(tokens,lengths,features,domains)
        decoded,_=self.decoder(self.output_embedding(decoder_inputs),context.unsqueeze(0))
        return {'text_logits':self.text_head(decoded),'physics':self.physics_head(context)}

def make_model(config):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(config.get('seed',61017)));return R6Model(config)

def predict(model,rows):
    device=next(model.parameters()).device;model.eval();results=[]
    with torch.inference_mode():
        for start in range(0,len(rows),64):
            chunk=rows[start:start+64];batch=encode_batch(chunk,device,include_targets=False)
            context=model.encode(batch['tokens'],batch['lengths'],batch['features'],batch['domains'])
            physics=(model.physics_head(context)*model.target_scale+model.target_mean).cpu().tolist()
            state=context.unsqueeze(0);token=torch.full((len(chunk),1),BOS,dtype=torch.long,device=device)
            ids=[]
            for _ in range(MAX_OUTPUT):
                out,state=model.decoder(model.output_embedding(token),state)
                token=model.text_head(out).argmax(-1);ids.append(token[:,0].cpu().tolist())
            for i,row in enumerate(chunk):
                if row['domain']=='physics':results.append({'state':physics[i]});continue
                chars=[]
                for step in ids:
                    value=step[i]
                    if value==EOS:break
                    chars.append('-' if value==MINUS else str(value-4) if value>=4 else '?')
                raw=''.join(chars)
                answer=raw if re.fullmatch(r'-?(0|[1-9][0-9]*)',raw) and raw!='-0' else None
                results.append({'answer':answer,'raw_text':raw})
    return results
