"""Fresh engineering probes only: backward passes with no optimizer and unchanged weights."""
import copy
from .examples import probes
from .shared import encode

def run(resources):
    import torch
    from mindscape.r5_1.engineering import run as connectivity
    from mindscape.v5.model import tensors
    from .contract import digest
    # Reuse the verified gradient/check specification with new public probe graphs, not old holdouts.
    tasks=[]
    for domain,public in probes().items():
        for _ in range(2):tasks.append({'domain':domain,'split':'composition','public':copy.deepcopy(public)})
    specification={'seed':13,'expected_parameters':6967,'encoder_feature_widths':{'mathematics':6,'coding':10,'physics':8},'effect_widths':{'mathematics':1,'coding':1,'physics':4},'minimum_gradient_norm':1e-12,'minimum_action_or_relation_sensitivity':1e-7,'maximum_permutation_error':1e-5}
    receipt=connectivity({'protocol':{'engineering':specification},'tasks':tasks})
    model,row=resources.shared();before={k:v.detach().clone() for k,v in model.state_dict().items()};ids=tuple(id(p) for p in model.core.parameters());records=[]
    for domain,public in probes().items():
        inputs=tensors([encode(domain,public)]);model.zero_grad(set_to_none=True)
        effect,valid=model(*inputs);loss=effect.square().sum()+valid.square().sum();loss.backward()
        norms={n:float(p.grad.norm().item()) for n,p in model.core.named_parameters() if p.grad is not None}
        if not norms or any(not torch.isfinite(p.grad).all() for p in model.core.parameters() if p.grad is not None):raise ValueError('Pinned shared core gradient failure')
        if not all(v>0 for v in norms.values()) or tuple(id(p) for p in model.core.parameters())!=ids:raise ValueError('Disconnected/replaced shared parameters')
        records.append({'domain':domain,'same_core':True,'core_gradient_norms':norms,'graph_hash':digest(encode(domain,public))})
    model.zero_grad(set_to_none=True)
    if any(not torch.equal(before[k],v) for k,v in model.state_dict().items()):raise ValueError('Engineering probe changed weights')
    return {'status':'PASS','fresh_probe_connectivity':receipt,'checkpoint_sha256':row['sha256'],'shared_parameter_identity':records,'optimizer_steps':0,'weights_unchanged':True,'learning_or_transfer_claim':False,'historical_holdout_examples_used':False}
