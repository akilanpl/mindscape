"""Read-only demonstration of the preserved 54,014-parameter R6 research model.

This model has no CBSE natural-language interface. Its output is never used as the
verified assistant's answer. There are no optimizer imports or training calls.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
RESEARCH=Path(os.environ.get('MINDSCAPE_R6_REPOSITORY', str(ROOT/'work/mindscape-r6')))
ARTIFACTS=Path(os.environ.get('MINDSCAPE_R6_ARTIFACTS',str(ROOT/'outputs/r6')))


def native_status() -> dict:
    return {
        'status':'research-only', 'method':'native Mindscape R6 graph/GRU neural inference',
        'cbse_natural_language_supported':False, 'trained_parameter_count':54014,
        'architecture':{'text':'byte embedding → GRU → autoregressive integer decoder',
                        'physics':'17 typed property nodes, three learned message-passing rounds, four-state regression',
                        'scope':'numeric graph masked outside physics; math/coding use the text GRU'},
        'historical_results':{'math_iid_exact_percent':1.0416666666666667,'math_ood_exact_percent':0,
                              'coding_iid_exact_percent':0,'coding_ood_exact_percent':0,
                              'physics_iid_mse':45.7424,'physics_ood_mse':3393.04,
                              'fit':'r6-p1-scaled best-dev, step600, one seed'},
        'limitations':['Math and coding outputs collapse to repeated integers.',
                       'Physics law probes and out-of-distribution generalization remain weak.',
                       'R6.1 pooled-encoder correction has zero trained recovery evidence.',
                       'No Class 12 mastery, general program execution or pretrained knowledge is claimed.'],
        'optimizer_updates_this_demo':0,
    }


def run_demo() -> dict:
    started=time.perf_counter()
    if not RESEARCH.exists():
        return {**native_status(),'demo_status':'unavailable','reason':'Preserved R6 source not mounted.'}
    if str(RESEARCH) not in sys.path:sys.path.insert(0,str(RESEARCH))
    try:
        import torch
        from research.r6.architectures.model import make_model,predict
        from research.r6.common import load_checkpoint
        index=json.loads((ARTIFACTS/'checkpoint-index.json').read_text())
        item=next(v for v in index if v['fit_id']=='r6-p1-scaled' and v['role']=='best')
        checkpoint=Path(item['path'])
        if not checkpoint.exists(): checkpoint=ARTIFACTS/'p1-scaled/fit/best-dev.json'
        before=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if before != item['sha256']:raise ValueError('Preserved checkpoint hash mismatch')
        model=make_model(item['config']).cpu().eval()
        for parameter in model.parameters():parameter.requires_grad_(False)
        metadata=load_checkpoint(checkpoint,model)
        # These are illustrative supported-schema examples, not a new held-out benchmark.
        # Predict() receives only input schema; answers below are external references.
        examples=[
            {'domain':'mathematics','input':'Calculate (7) + (8).','features':[]},
            {'domain':'coding','input':'Predict the final printed integer.\nx = 3\nx = x + 4\nprint(x)','features':[]},
            {'domain':'physics','input':'Initial vx=0, vy=0, x=0, y=0; mass=2. Stage 1: Fx=4, Fy=0, dt=1.',
             'features':[0,0,0,0,2,4,0,1,1,0,0,0,0,0,0,0,0]},
        ]
        references=[{'answer':'15'},{'answer':'7'},{'state':[2,0,1,0],'units':['m/s','m/s','m','m']}]
        infer_start=time.perf_counter()
        predictions=predict(model,examples)
        latency=time.perf_counter()-infer_start
        after=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if after != before:raise RuntimeError('Checkpoint changed during read-only demonstration')
        assert all(p.grad is None for p in model.parameters())
        return {**native_status(),'demo_status':'completed','checkpoint_sha256':before,
                'checkpoint_preserved':after==before,'checkpoint_metadata':metadata,
                'device':'cpu','torch_version':torch.__version__,
                'parameter_count_actual':sum(p.numel() for p in model.parameters()),
                'examples':[{'input':r,'native_prediction':p,'external_reference':e,
                             'reference_method':'Independent arithmetic/program/formula reference; not native inference'}
                            for r,p,e in zip(examples,predictions,references)],
                'timing_seconds':{'inference':latency,'total':time.perf_counter()-started},
                'scientific_interpretation':'Actual preserved model inference. Examples are illustrative, not a fresh accuracy estimate. Native predictions may be incorrect.'}
    except (ImportError,FileNotFoundError) as exc:
        return {**native_status(),'demo_status':'unavailable','reason':str(exc)}

if __name__=='__main__':print(json.dumps(run_demo(),indent=2,allow_nan=False))
