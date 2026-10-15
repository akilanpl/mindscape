"""Bounded two-example concept-dependent engineering task, not physical evidence."""
import argparse,time
from pathlib import Path
import torch
from .model import make_model
from .runner import write

def run(out):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    results=[]
    for condition in ('graph-concept-on','graph-concept-off'):
        model=make_model(condition,17)
        # Identical numeric observations. Only explicit concept ID distinguishes labels.
        # Artificial diagnostic semantics; NOT relabeled physics/campaign data.
        problems=[{'domain':'physics','mass':2.,'force':[0.,0.],'dt':1.,'velocity':[0.,0.],'concept_ids':[c]} for c in (10,11)]
        inputs=[model.prepare(p) for p in problems];targets=[torch.tensor([s,s]) for s in (-1.,1.)]
        def predict():
            with torch.no_grad():return [model(p)['physics_delta'].tolist() for p in inputs]
        initial=predict();opt=torch.optim.Adam(model.parameters(),lr=.001);count=[0]
        hook=opt.register_step_post_hook(lambda *args:count.__setitem__(0,count[0]+1));start=time.perf_counter()
        for i in range(200):
            opt.zero_grad(set_to_none=True);loss=torch.nn.functional.mse_loss(model(inputs[i%2])['physics_delta'],targets[i%2]);loss.backward();opt.step()
        hook.remove();final=predict()
        results.append({'condition':condition,'initial':initial,'final':final,'targets':[x.tolist() for x in targets],
         'final_mse':sum((x-y)**2 for a,b in zip(final,targets) for x,y in zip(a,b.tolist()))/4,
         'gate':float(model.core.concepts.gate.detach()),'updates':count[0],'seconds':time.perf_counter()-start})
    write(out/'evidence/concept-dependent-probe.json',{'scope':'artificial two-example concept-ID task; not physics or generalization evidence','device':'cpu','actual_optimizer_updates':400,'runs':results})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
