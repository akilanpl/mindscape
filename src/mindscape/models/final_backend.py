"""Offline frozen SmolLM2 features and equally sized learned categorical heads."""
import hashlib
import json
from pathlib import Path
import numpy as np

class FrozenFeatures:
    def __init__(self, path, cache):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch=torch;torch.set_num_threads(2)
        self.path=str(Path(path).resolve());self.cache=Path(cache);self.cache.mkdir(parents=True,exist_ok=True)
        self.tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
        if self.tokenizer.pad_token_id is None:self.tokenizer.pad_token=self.tokenizer.eos_token
        self.device='mps' if torch.backends.mps.is_available() else 'cpu'
        self.model=AutoModel.from_pretrained(path,local_files_only=True).to(self.device).eval()
        for p in self.model.parameters():p.requires_grad_(False)
        self.size=self.model.config.hidden_size+52*11
        self.parameter_count=sum(p.numel() for p in self.model.parameters())
        self.calls=0;self.tokens=0;self.requests=0;self.values={}
        self.revision=Path(path).name
    def encode(self,x):
        x=np.asarray(x);keys=[hashlib.sha256(self.revision.encode()+np.asarray(row,dtype='<f8').tobytes()).hexdigest() for row in x]
        missing={key:row for key,row in zip(keys,x) if key not in self.values}
        todo=[]
        for key,row in missing.items():
            p=self.cache/(key+'.npy')
            if p.exists():self.values[key]=np.load(p,allow_pickle=False)
            else:todo.append((key,row))
        for start in range(0,len(todo),32):
            batch=todo[start:start+32]
            texts=['Structured numerical features: '+' '.join(f'f{i}={v:.6g}' for i,v in enumerate(row) if v!=0) for _,row in batch]
            inputs=self.tokenizer(texts,return_tensors='pt',padding=True,truncation=False)
            self.tokens+=int(inputs['attention_mask'].sum());self.calls+=1
            inputs={k:v.to(self.device) for k,v in inputs.items()}
            with self.torch.inference_mode():
                hidden=self.model(**inputs).last_hidden_state
                mask=inputs['attention_mask'].unsqueeze(-1)
                pooled=(hidden*mask).sum(1)/mask.sum(1)
                pooled=pooled.cpu().numpy().astype(np.float32)
            pooled/=np.maximum(np.linalg.norm(pooled,axis=1,keepdims=True),1e-8)
            for (key,row),vector in zip(batch,pooled):
                categorical=np.zeros((52,11),dtype=np.float32)
                bins=np.clip(np.rint(row*9),0,10).astype(int)
                categorical[np.arange(52),bins]=1
                self.values[key]=np.r_[vector,categorical.ravel()]
                np.save(self.cache/(key+'.npy'),self.values[key],allow_pickle=False)
        self.requests+=len(x)
        return np.asarray([self.values[key] for key in keys],dtype=np.float32)

class FinalBackend:
    identifier='smollm2_frozen_categorical_head_v1'
    groups=[10]*8+[2]
    def __init__(self,encoder,seed=0):
        import torch
        self.encoder=encoder;self.seed=seed;torch.manual_seed(seed)
        self.head=torch.nn.Sequential(torch.nn.Linear(encoder.size,256),torch.nn.GELU(),
            torch.nn.Linear(256,128),torch.nn.GELU(),torch.nn.Linear(128,82))
    @property
    def parameter_count(self):return self.encoder.parameter_count+sum(p.numel() for p in self.head.parameters())
    @property
    def trainable_parameters(self):return sum(p.numel() for p in self.head.parameters())
    def logits(self,x):
        import torch
        with torch.inference_mode():return self.head(torch.from_numpy(self.encoder.encode(x))).numpy()
    def generate(self,x):return np.stack([a.argmax(-1) for a in np.split(self.logits(x),np.cumsum(self.groups)[:-1],axis=-1)],axis=-1)
    def fit(self,x,y,steps=1200,seed=0,active=9):
        import torch
        encoded=torch.from_numpy(self.encoder.encode(x));labels=torch.as_tensor(y,dtype=torch.long)
        opt=torch.optim.Adam(self.head.parameters(),lr=.003);rng=np.random.default_rng(seed);loss=None
        for _ in range(steps):
            idx=rng.integers(0,len(x),size=64);out=self.head(encoded[idx]);offset=0;loss=0
            for i,n in enumerate(self.groups):
                if i<active:loss=loss+torch.nn.functional.cross_entropy(out[:,offset:offset+n],labels[idx,i])/active
                offset+=n
            opt.zero_grad();loss.backward();opt.step()
        return {'final_batch_loss':float(loss.detach()),'gradient_steps':steps,'minibatch_rows_processed':steps*64,'active_heads':active}
    def save(self,path):
        path=Path(path);path.mkdir(parents=True,exist_ok=False)
        arrays={k:v.detach().numpy() for k,v in self.head.state_dict().items()}
        np.savez_compressed(path/'head.npz',**arrays)
        (path/'backend.json').write_text(json.dumps({'backend':self.identifier,'seed':self.seed,'encoder_path':self.encoder.path,'encoder_revision':self.encoder.revision,'encoder_parameters':self.encoder.parameter_count,'trainable_parameters':self.trainable_parameters,'size':self.encoder.size},indent=2))
    @classmethod
    def load(cls,path,encoder):
        import torch
        meta=json.loads((Path(path)/'backend.json').read_text())
        if meta['encoder_revision']!=encoder.revision or meta['size']!=encoder.size:raise ValueError('Backbone mismatch')
        model=cls(encoder,meta['seed'])
        with np.load(Path(path)/'head.npz',allow_pickle=False) as a:
            weights={k:torch.from_numpy(a[k].copy()) for k in model.head.state_dict()}
            if any(not torch.isfinite(v).all() for v in weights.values()):raise ValueError('Nonfinite checkpoint')
            model.head.load_state_dict(weights,strict=True)
        return model
