"""R6 input/output codec and safe checkpoints; contains no solver or optimizer."""
from pathlib import Path
import hashlib,json,math,re
DOMAINS=('mathematics','physics','coding')
PAD,BOS,EOS,MINUS=0,1,2,3
VOCAB=14
MAX_INPUT=256
MAX_OUTPUT=12
FEATURE_SCALES=(10.,10.,20.,20.,5.)+(10.,10.,2.,1.)*3

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def read(path):
    return json.loads(Path(path).read_text(),parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+'\n')
def encode_answer(answer):
    if not isinstance(answer,str) or not re.fullmatch(r'-?(0|[1-9][0-9]*)',answer):raise ValueError('Canonical integer target required')
    if str(int(answer))!=answer:raise ValueError('Noncanonical integer target')
    ids=[MINUS if c=='-' else int(c)+4 for c in answer]+[EOS]
    if len(ids)>MAX_OUTPUT:raise ValueError('Answer outside declared output length')
    return ids

def encode_batch(rows,device,include_targets=True):
    import torch
    if not rows:raise ValueError('Empty batch')
    texts=[r['input'].encode('ascii') for r in rows]
    if any(not s or len(s)>MAX_INPUT for s in texts):raise ValueError('Input outside declared ASCII length; no silent truncation')
    lengths=torch.tensor([len(s) for s in texts],dtype=torch.long)
    tokens=torch.zeros(len(rows),max(map(len,texts)),dtype=torch.long,device=device)
    features=torch.zeros(len(rows),17,dtype=torch.float32,device=device)
    domains=torch.tensor([DOMAINS.index(r['domain']) for r in rows],device=device)
    physics=torch.zeros(len(rows),4,device=device);answers=[]
    for i,(row,text) in enumerate(zip(rows,texts)):
        tokens[i,:len(text)]=torch.tensor([b+1 for b in text],device=device)
        if row['domain']=='physics':
            if len(row['features'])!=17:raise ValueError('17 declared raw physical features required')
            features[i]=torch.tensor(row['features'],device=device)
            if include_targets:physics[i]=torch.tensor(row['target']['state'],device=device)
            answers.append([EOS])
        else:answers.append(encode_answer(row['target']['answer']) if include_targets else [EOS])
    width=max(map(len,answers))
    target=torch.zeros(len(rows),width,dtype=torch.long,device=device)
    decoder=torch.zeros_like(target);decoder[:,0]=BOS
    for i,answer in enumerate(answers):
        target[i,:len(answer)]=torch.tensor(answer,device=device)
        if len(answer)>1:decoder[i,1:len(answer)]=target[i,:len(answer)-1]
    return {'tokens':tokens,'lengths':lengths,'features':features,'domains':domains,
            'target_input':decoder,'target_output':target,'physics_target':physics,
            'physics_mask':domains==1,'text_mask':domains!=1}

def save_checkpoint(path,model,metadata):
    path=Path(path)
    if path.exists():raise FileExistsError('Immutable checkpoint already exists')
    tensors={}
    for name,value in model.state_dict().items():
        value=value.detach().cpu()
        if not bool(value.isfinite().all()):raise ValueError('Nonfinite checkpoint')
        tensors[name]={'shape':list(value.shape),'dtype':str(value.dtype),'values':value.tolist()}
    write(path,{'format':'mindscape-r6-safe-json-v1','metadata':metadata,'tensors':tensors,'scientifically_qualified':False})
    return sha(path)

def load_checkpoint(path,model):
    import torch
    data=read(path)
    if data.get('format')!='mindscape-r6-safe-json-v1':raise ValueError('R6 safe JSON required')
    expected=model.state_dict()
    if set(expected)!=set(data['tensors']):raise ValueError('Checkpoint tensor keys differ')
    state={}
    for name,target in expected.items():
        item=data['tensors'][name]
        if item['shape']!=list(target.shape) or item['dtype']!=str(target.dtype):raise ValueError('Tensor schema differs')
        value=torch.tensor(item['values'],dtype=target.dtype,device=target.device)
        if value.shape!=target.shape or not bool(value.isfinite().all()):raise ValueError('Invalid tensor data')
        state[name]=value
    model.load_state_dict(state,strict=True)
    return data['metadata']
