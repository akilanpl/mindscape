"""Bounded CPU engineering optimization; separate from every MPS fit budget."""
import argparse,json,time
from pathlib import Path
import torch
from .model import make_model
from .evaluation import evaluate
from .runner import write,CONDITIONS
from mindscape.r56.compact_gpu import domain_loss
from mindscape.r56.learning import save_checkpoint,load_checkpoint

def run(out):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    rows=json.loads((out/'data-v1/train/17.json').read_text())
    dev=json.loads((out/'data-v1/dev/17.json').read_text())
    dest=out/'evidence/tiny-proof';dest.mkdir(exist_ok=False)
    results=[];began=time.perf_counter()
    # Fixed budget announced before results: 3 conditions x 3 sample counts x 320 steps.
    for condition in CONDITIONS:
        for count in (4,8,16):
            pools={d:[] for d in ('math','coding','physics')}
            for d in pools:
                candidates=[r for r in rows if r['problem']['domain']==d]
                seen=set();first=[];rest=[]
                for r in candidates:
                    label=json.dumps(r['labels'],sort_keys=True)
                    if label not in seen:first.append(r);seen.add(label)
                    else:rest.append(r)
                pools[d]=first+rest
            subset=[pools[('math','coding','physics')[i%3]][i//3] for i in range(count)]
            model=make_model(condition,17);prepared=[model.prepare(r['problem']) for r in subset]
            def metrics(items):
                with torch.no_grad():return evaluate(model.predict,items)
            # Training set may repeat; give evaluate distinct normalized examples only.
            from .evaluation import canonical_id
            unique=list({canonical_id(r['problem']):r for r in subset}.values())
            initial=metrics(unique);initialdev=metrics(dev)
            path=dest/f'{condition}-{count}-initial.json';h=save_checkpoint(model,path,{'optimizer_steps':0})
            clone=make_model(condition,17);load_checkpoint(clone,path,h)
            with torch.no_grad():
                a=model(prepared[0]);b=clone(prepared[0])
                assert all(torch.equal(a[k],b[k]) for k in a if torch.is_tensor(a[k]))
            opt=torch.optim.Adam(model.parameters(),lr=.001);actual=[0]
            hook=opt.register_step_post_hook(lambda *args:actual.__setitem__(0,actual[0]+1))
            losses=[];start=time.perf_counter()
            for step in range(320):
                i=step%count;opt.zero_grad(set_to_none=True);loss=domain_loss(model(prepared[i]),subset[i],torch,'cpu')
                loss.backward();assert torch.isfinite(loss)
                assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
                opt.step();losses.append(float(loss.detach()))
            hook.remove();assert actual[0]==320
            result={'condition':condition,'count':count,'optimizer_updates':actual[0],'seconds':time.perf_counter()-start,'device':'cpu','scope':'ENGINEERING_ONLY',
              'initial':initial,'final':metrics(unique),'initial_dev':initialdev,'final_dev':metrics(dev),'losses':losses,'initial_reload_bitwise_equal':True,
              'initial_sha256':h,'parameter_count':sum(p.numel() for p in model.parameters()),
              'parameters_with_gradient_last_step':sum(p.numel() for p in model.parameters() if p.grad is not None)}
            write(dest/f'{condition}-{count}.json',result);results.append({k:v for k,v in result.items() if k not in ('initial','final','initial_dev','final_dev','losses') } | {key:result[key]['metrics'] for key in ('initial','final','initial_dev','final_dev')})
            print(json.dumps(results[-1]),flush=True)
    write(dest/'summary.json',{'actual_cpu_optimizer_updates':sum(r['optimizer_updates'] for r in results),'mps_updates':0,'seconds':time.perf_counter()-began,'runs':results,'candidate_selection':'single candidate; no final exploratory targets read'})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);run(p.parse_args().output)
