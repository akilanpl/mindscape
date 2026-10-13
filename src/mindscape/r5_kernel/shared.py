"""Existing V5 graph adapter; semantics and negative results remain domain-qualified."""
from .contract import plain,digest

def encode(domain,public):
    fields={'mathematics':{'values','operation','destination','source','amount'},'coding':{'x','constants','operations','target_leaf','increment','entry'},'physics':{'positions','velocities','masses','destination','impulse','gravity','scene_seed'}}
    if domain not in fields or set(public)!=fields[domain]:raise ValueError('Shared public adapter fields')
    if len(str(public))>4096:raise ValueError('Shared adapter input bound')
    from mindscape.v5.data import graph
    from mindscape.v5.schema import validate
    return validate(graph({'domain':domain,'public':public}))

def execute(request,resources,verification):
    if request.operation!='transition' or set(request.observation)!={'entities','observed','relations'} or set(request.action)!={'encoding'}:raise ValueError('Shared graph contract required')
    from mindscape.v5.schema import validate
    graph=validate({'domain':request.domain,**request.observation,'action':request.action['encoding']})
    model,row=resources.shared()
    import torch
    from mindscape.v5.model import tensors
    with torch.inference_mode():effect,valid=model(*tensors([graph]))
    if not torch.isfinite(effect).all() or not torch.isfinite(valid).all():raise ValueError('Nonfinite shared prediction')
    value={'normalized_effect':effect[0,:len(graph['entities'])].cpu().tolist(),'validity_probability':float(valid.sigmoid()[0].item()),'kind':'uncorrected_shared_transition_prediction'}
    return dict(prediction={'kind':'neural_prediction','value':value},execution={'kind':'not_executed'},verification={'success':None,'method':'No execution targets supplied; infrastructure only'},provenance={'method':'V5_GraphTransition_R5.1_checkpoint','neural':True,'shared_parameters':True,'parameters':6967,'domain_encoder':request.domain,'domain_head':request.domain,'core_parameter_ids':[id(p) for p in model.core.parameters()],'checkpoint_sha256':row['sha256'],'scientific_status':'NO_GO','checkpoint_identity':row['identity']})
