"""Owner-authorized exploratory fits. No official gate or evaluator imports."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

CONDITIONS=('graph-concept-on','graph-concept-off','matched-gru')
SEEDS=(17,29,43)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False)

def verify_identity(out):
    frozen=json.loads((out/'candidate-freeze.json').read_text())
    root=Path(frozen['source_root'])
    if frozen['status']!='CANDIDATE_FROZEN' or not frozen['source_hashes']:raise RuntimeError('Candidate not frozen')
    if Path(__file__).resolve()!=root/'src/mindscape/r57/runner.py':raise RuntimeError('Wrong runner source')
    for relative,digest in frozen['source_hashes'].items():
        if sha(root/relative)!=digest:raise RuntimeError('Source changed: '+relative)
    if sha(out/'data-v1/manifest.json')!=frozen['data_manifest_sha256']:raise RuntimeError('Data manifest changed')
    return frozen

def require_mps():
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK','0')!='0':raise RuntimeError('CPU fallback forbidden')
    import torch
    if platform.system()!='Darwin' or platform.machine()!='arm64' or not torch.backends.mps.is_available():
        raise RuntimeError('NATIVE_MPS_EXECUTION_REQUIRED; no training started')
    torch.use_deterministic_algorithms(True)
    return torch

def fit(out,seed,condition):
    frozen=verify_identity(out);torch=require_mps()
    if seed not in SEEDS or condition not in CONDITIONS:raise ValueError('Unknown fit')
    from .model import make_model
    from mindscape.r56.compact_gpu import domain_loss
    from mindscape.r56.learning import save_checkpoint,load_checkpoint
    data=out/f'data-v1/train/{seed}.json'
    manifest=json.loads((out/'data-v1/manifest.json').read_text())
    if sha(data)!=manifest['files'][f'train/{seed}.json']:raise RuntimeError('Training file changed')
    rows=json.loads(data.read_text())
    if len(rows)!=128 or any(r['split']!='train' or r['seed']!=seed for r in rows):raise RuntimeError('Expected exact128 legacy records')
    run=out/f'checkpoints/{seed}-{condition}';run.mkdir(exist_ok=False)
    updates=0;began=time.perf_counter()
    try:
        model=make_model(condition,seed).to('mps');prepared=[model.prepare(r['problem']) for r in rows]
        def summary():
            values={d:[] for d in ('math','coding','physics')}
            with torch.no_grad():
                for p,r in zip(prepared,rows):values[r['problem']['domain']].append(float(domain_loss(model(p),r,torch,'mps').cpu()))
            return {d:sum(v)/len(v) for d,v in values.items()}
        initial_loss=summary()
        meta={'seed':seed,'condition':condition,'optimizer_steps':0,'scope':'EXPLORATORY_NOT_SCIENTIFICALLY_QUALIFIED'}
        initial_sha=save_checkpoint(model,run/'initial.json',meta)
        restored=make_model(condition,seed).to('mps');load_checkpoint(restored,run/'initial.json',initial_sha)
        with torch.no_grad():
            a=model(prepared[0]);b=restored(prepared[0])
            if any(not torch.equal(a[k],b[k]) for k in a if torch.is_tensor(a[k])):raise RuntimeError('Initial checkpoint reload differs')
        del restored
        optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        actual_steps=[0]
        hook=optimizer.register_step_post_hook(lambda *args:actual_steps.__setitem__(0,actual_steps[0]+1))
        torch.mps.synchronize();start=time.perf_counter()
        logpath=out/f'losses/{seed}-{condition}.jsonl'
        with logpath.open('x',buffering=1) as log:
            for step in range(2000):
                index=step%128;optimizer.zero_grad(set_to_none=True)
                prediction=model(prepared[index]);loss=domain_loss(prediction,rows[index],torch,'mps')
                if loss.device.type!='mps' or not bool(torch.isfinite(loss)):raise RuntimeError('Invalid loss')
                loss.backward();grads=[p.grad for p in model.parameters() if p.grad is not None]
                if not grads or any(g.device.type!='mps' for g in grads) or not bool(torch.stack([g.isfinite().all() for g in grads]).all()):raise RuntimeError('Invalid gradients')
                optimizer.step();updates=actual_steps[0]
                if not bool(torch.stack([p.isfinite().all() for p in model.parameters()]).all()):raise RuntimeError('Nonfinite weights')
                log.write(json.dumps({'update':updates,'record_index':index,'domain':rows[index]['problem']['domain'],'preupdate_loss':float(loss.detach().cpu())})+'\n')
                if updates%250==0:print(json.dumps({'seed':seed,'condition':condition,'updates':updates}),flush=True)
        torch.mps.synchronize();seconds=time.perf_counter()-start;hook.remove()
        steps={n:int(optimizer.state[p]['step'].item()) for n,p in model.named_parameters() if 'step' in optimizer.state[p]}
        if updates!=2000 or max(steps.values())!=2000:raise RuntimeError('Optimizer step verification failed')
        final_loss=summary()
        result={**meta,'optimizer_steps':updates,'optimizer_updates':updates,'step_post_hook_count':actual_steps[0],
          'adam_steps_by_parameter':steps,'initial_sha256':initial_sha,'initial_training_loss':initial_loss,'final_training_loss':final_loss,
          'seconds':seconds,'device':'mps','torch':str(torch.__version__),'python':sys.version,'platform':platform.platform(),
          'deterministic':torch.are_deterministic_algorithms_enabled(),'cpu_fallback':False,'training_sha256':sha(data),
          'parameters_with_adam_state':sum(p.numel() for n,p in model.named_parameters() if n in steps),'unused_parameter_names':[n for n,p in model.named_parameters() if n not in steps],'parameter_count':sum(p.numel() for p in model.parameters()),'trainable_count':sum(p.numel() for p in model.parameters() if p.requires_grad),
          'mps_current_bytes':torch.mps.current_allocated_memory(),'mps_driver_bytes':torch.mps.driver_allocated_memory(),
          'memory_measurement':'end-of-fit allocation, not peak','source_commit':frozen['source_commit'],'loss_sha256':sha(logpath),
          'candidate_freeze_sha256':sha(out/'candidate-freeze.json'),'official_updates':0,'checkpoint_policy':'final-update-2000-only','initial_reload_bitwise_equal':True}
        result['checkpoint_sha256']=save_checkpoint(model,run/'final.json',result)
        restored=make_model(condition,seed).to('mps');load_checkpoint(restored,run/'final.json',result['checkpoint_sha256'])
        with torch.no_grad():
            for p in prepared[:3]:
                a=model(p);b=restored(p)
                if any(not torch.equal(a[k],b[k]) for k in a if torch.is_tensor(a[k])):raise RuntimeError('Final checkpoint reload differs')
        result['status']='COMPLETED_EXPLORATORY';result['final_reload_bitwise_equal']=True
        write(run/'result.json',result)
    except BaseException as exc:
        write(run/'failure.json',{'error':str(exc),'traceback':traceback.format_exc(),'actual_optimizer_updates':updates,'elapsed_seconds':time.perf_counter()-began})
        raise

def completed(run):
    out=run.parent.parent;frozen=verify_identity(out)
    result=json.loads((run/'result.json').read_text())
    if result['status']!='COMPLETED_EXPLORATORY' or result['optimizer_updates']!=2000 or result['optimizer_steps']!=2000 or result['step_post_hook_count']!=2000:raise RuntimeError('Incomplete existing fit; not resumed')
    if run.name!=f"{result['seed']}-{result['condition']}" or result['source_commit']!=frozen['source_commit'] or result['candidate_freeze_sha256']!=sha(out/'candidate-freeze.json'):raise RuntimeError('Existing fit identity differs')
    if result['training_sha256']!=sha(out/f"data-v1/train/{result['seed']}.json"):raise RuntimeError('Training data changed')
    for name,key in [('initial.json','initial_sha256'),('final.json','checkpoint_sha256')]:
        if sha(run/name)!=result[key]:raise RuntimeError('Existing checkpoint changed')
    log=out/f'losses/{run.name}.jsonl'
    if sha(log)!=result['loss_sha256']:raise RuntimeError('Loss evidence changed')
    entries=[json.loads(line) for line in log.read_text().splitlines()]
    if len(entries)!=2000 or [r['update'] for r in entries]!=list(range(1,2001)):raise RuntimeError('Loss update sequence differs')
    return result

def batch(out):
    verify_identity(out);require_mps()
    lock=out/'batch.running';lock.mkdir(exist_ok=False)
    try:
        for seed in SEEDS:
            for condition in CONDITIONS:
                run=out/f'checkpoints/{seed}-{condition}'
                if run.exists():
                    completed(run);continue
                subprocess.run([sys.executable,'-m','mindscape.r57.runner','--output',str(out),'--fit',str(seed),condition],check=True)
        # Evaluation is a separate process after every candidate fit is finished.
        frozen=json.loads((out/'candidate-freeze.json').read_text())
        subprocess.run([sys.executable,'-m','mindscape.r57.scoreboard','--output',str(out),'--legacy',frozen['legacy_root'],'--original-source',frozen['original_source']],check=True)
    finally:lock.rmdir()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--fit',nargs=2)
    args=parser.parse_args()
    if args.fit:
        seed=int(args.fit[0]);condition=args.fit[1]
        if seed not in SEEDS or condition not in CONDITIONS:raise ValueError('Unknown fit')
        fit(args.output,seed,condition)
    else:batch(args.output)
