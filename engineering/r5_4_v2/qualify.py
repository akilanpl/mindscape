"""Proposed v2 engineering microbenchmark. Explicit review approval required; never a fit."""
import argparse
import json
import math
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
from common import REPO, HERE, load, verify, rows, write_new, sha, decision
sys.path.insert(0,str(REPO/'src'))


def cpu_copy(torch, mapping):
    return {name: tensor.detach().to('cpu').clone() for name,tensor in mapping.items()}


def parameter_hash(model):
    import hashlib
    h = hashlib.sha256()
    for name,p in model.named_parameters():
        h.update(name.encode())
        h.update(p.detach().to('cpu').contiguous().numpy().tobytes())
    return h.hexdigest()


def memory(torch):
    return dict(process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                sampled_live_tensor_bytes=torch.mps.current_allocated_memory(),
                sampled_driver_allocated_bytes=torch.mps.driver_allocated_memory())


def worker(config_index, output):
    verify()
    plan = load(HERE/'plan.json')
    start_record=load(output/'START.json')
    if start_record['approved_engineering_only'] is not True or start_record['training_authorized'] is not False or start_record['v2_lock_sha256']!=sha(HERE/'LOCK.json'):
        raise RuntimeError('Missing engineering-only invocation approval/provenance')
    kind,seed = plan['configurations'][config_index]
    result = dict(label='R54_V2_DISPOSABLE_OPTIMIZER_QUALIFICATION',config_index=config_index,kind=kind,seed=seed,
                  records=[],campaign_optimizer_updates=0,scratch_optimizer_steps=0,status='R54_V2_ENGINEERING_NO_GO')
    import torch
    if (platform.system()!='Darwin' or platform.machine()!='arm64' or not torch.backends.mps.is_available()
            or os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK','0')!='0'):
        result.update(reason='Native MPS unavailable; no CPU fallback',status='R54_V2_ENGINEERING_UNAVAILABLE')
        write_new(output/f'worker-{config_index}.json',result)
        return
    from mindscape.r5_4.model import Transition, COUNTS
    from mindscape.r5_4.training import objective
    initialization_start = time.perf_counter()
    train = load(REPO/'experiments/r5_4/frozen/train.json')
    batches = {'first32':train[:32], 'first32_four_digit_training_tasks':[
        t for t in train if len(t['left'])==len(t['right'])==4][:32]}
    if any(len(batch)!=32 or any(t['split']!='train' for t in batch) for batch in batches.values()):
        raise RuntimeError('Training batch identity failure')
    torch.manual_seed(seed)
    model = Transition(kind).to(device='mps',dtype=torch.float32)
    snapshots = {name:p.detach().clone() for name,p in model.named_parameters()}
    initial_hash = parameter_hash(model)
    initial_cpu = cpu_copy(torch,snapshots)
    optimizer = torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.0001)
    # This optimizer owns only the disposable model. Actual step work is counted, never disguised as zero.
    torch.mps.synchronize()
    result.update(initialization_seconds=time.perf_counter()-initialization_start,parameter_count=COUNTS[kind],
                  initial_parameter_sha256=initial_hash,batch_ids={k:[t['task_id'] for t in v] for k,v in batches.items()},
                  runtime=dict(mps_available=True,mps_built=torch.backends.mps.is_built(),cpu_fallback=False,
                               torch=torch.__version__,platform=platform.platform(),machine=platform.machine()),
                  v2_lock_sha256=sha(HERE/'LOCK.json'))
    raw_gradients={}
    try:
        for call_id, position in enumerate(rows(plan)):
            torch.mps.synchronize()
            start = time.perf_counter_ns()
            optimizer.zero_grad(set_to_none=True)
            loss,_ = objective(model,batches[position['batch']],401)
            if not torch.isfinite(loss).item():
                raise RuntimeError('Nonfinite loss')
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all().item() for p in model.parameters()):
                raise RuntimeError('Missing/nonfinite gradient')
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
            result['scratch_optimizer_steps']+=1
            torch.mps.synchronize()
            # Same loss-to-host and progress serialization cost as frozen fitting; explicitly qualification log.
            loss_value=float(loss.item())
            with (output/f'worker-{config_index}.scratch-progress.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(microbenchmark_call_id=call_id,objective_update_mode=401,
                                             loss=loss_value,dtype=str(loss.dtype)))+'\n')
            torch.mps.synchronize()
            elapsed=(time.perf_counter_ns()-start)/1e9
            after_step_hash=parameter_hash(model)
            grad_metadata=[]
            for name,p in model.named_parameters():
                if p.device.type!='mps' or p.dtype!=torch.float32 or not torch.isfinite(p).all().item():
                    raise RuntimeError('Parameter placement/dtype/nonfinite')
                g=p.grad
                if g.device.type!='mps' or g.dtype!=torch.float32 or not torch.isfinite(g).all().item() or not math.isfinite(g.norm().item()) or g.norm().item()<=0:
                    raise RuntimeError('Gradient placement/dtype/norm')
                grad_metadata.append(dict(name=name,shape=list(g.shape),dtype=str(g.dtype),device=str(g.device),norm=float(g.norm().item())))
            raw_gradients[str(call_id)]=cpu_copy(torch,{n:p.grad for n,p in model.named_parameters()})
            if after_step_hash==initial_hash:
                raise RuntimeError('Optimizer write workload did not change disposable parameters')
            state_metadata=[]
            for state in optimizer.state.values():
                if state['step'].item()!=1:
                    raise RuntimeError('Scratch optimizer counter not reset')
                for name,tensor in state.items():
                    if not torch.isfinite(tensor).all().item() or (name!='step' and tensor.device.type!='mps'):
                        raise RuntimeError('Optimizer state nonfinite/device')
                state_metadata.append({name:dict(dtype=str(t.dtype),device=str(t.device),shape=list(t.shape)) for name,t in state.items()})
            with torch.no_grad():
                for name,p in model.named_parameters():
                    p.copy_(snapshots[name])
                for state in optimizer.state.values():
                    for tensor in state.values():
                        tensor.zero_()  # retain allocated buffers, restore zero moments and step0
            torch.mps.synchronize()
            if parameter_hash(model)!=initial_hash or any(not torch.equal(p,snapshots[name]) for name,p in model.named_parameters()):
                raise RuntimeError('Scratch weights did not restore exactly')
            if any(t.count_nonzero().item()!=0 for state in optimizer.state.values() for t in state.values()):
                raise RuntimeError('Scratch optimizer state did not reset')
            record=dict(position,call_id=call_id,full_update_seconds=elapsed,loss=loss_value,loss_dtype=str(loss.dtype),
                gradients=grad_metadata,optimizer_state_metadata=state_metadata,scratch_step_counter_before=0,
                scratch_step_counter_after=1,optimizer_state_zero_after_restore=True,
                initial_parameter_sha256=initial_hash,after_step_parameter_sha256=after_step_hash,
                restored_parameter_sha256=initial_hash,weights_restored_exactly=True,
                synchronized_before_and_after=True,memory=memory(torch))
            result['records'].append(record)
            with (output/f'worker-{config_index}.progress.jsonl').open('a') as stream:
                stream.write(json.dumps(record,sort_keys=True,allow_nan=False)+'\n')
                stream.flush()
                os.fsync(stream.fileno())
            if record['memory']['process_peak_rss_bytes']>4294967296:
                raise RuntimeError('4 GiB RSS exceeded')
            del loss
        result['status']='R54_V2_MICROBENCHMARK_COMPLETE'
    except Exception as error:
        result['reason']=repr(error)
    finally:
        final_hash=parameter_hash(model)
        result['final_parameter_sha256']=final_hash
        result['final_weights_restored']=final_hash==initial_hash
        result['peak_process_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Storage-only CPU copies for independent verification; no CPU inference/backward or fallback.
        tensor_path=output/f'worker-{config_index}.restoration-and-gradients.pt'
        torch.save(dict(initial=initial_cpu,restored=cpu_copy(torch,dict(model.named_parameters())),gradients=raw_gradients,optimizer_state=optimizer.state_dict()),tensor_path)
        result['tensor_evidence_sha256']=sha(tensor_path)
        if final_hash!=initial_hash:
            result['status']='R54_V2_ENGINEERING_NO_GO'
        verify()
        write_new(output/f'worker-{config_index}.json',result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--approve-engineering-qualification-v2',action='store_true')
    parser.add_argument('--worker',type=int,choices=range(6),help=argparse.SUPPRESS)
    args=parser.parse_args()
    if not args.approve_engineering_qualification_v2:
        raise SystemExit('R54_V2_FROZEN_AWAITING_QUALIFICATION_AUTHORIZATION: review approval required; no work executed')
    verify()
    if args.worker is not None:
        worker(args.worker,args.output)
        return
    plan=load(HERE/'plan.json')
    args.output.mkdir(parents=True,exist_ok=False)
    write_new(args.output/'START.json',dict(label='R54_V2_ENGINEERING_ONLY',v2_lock_sha256=sha(HERE/'LOCK.json'),
              original_readiness='ENGINEERING_NO_GO',approved_engineering_only=True,training_authorized=False,plan=plan))
    results=[]
    for i in range(6):
        start=time.perf_counter()
        try:
            subprocess.run([sys.executable,str(Path(__file__).resolve()),'--output',str(args.output.resolve()),
                            '--worker',str(i),'--approve-engineering-qualification-v2'],cwd=REPO,
                           env=dict(os.environ,PYTORCH_ENABLE_MPS_FALLBACK='0'),check=True,timeout=90)
            result=load(args.output/f'worker-{i}.json')
            result['parent_observed_worker_wall_seconds']=time.perf_counter()-start
        except (subprocess.SubprocessError,OSError,ValueError) as error:
            result=dict(config_index=i,status='R54_V2_ENGINEERING_NO_GO',reason=repr(error))
        results.append(result)
        if result['status']!='R54_V2_MICROBENCHMARK_COMPLETE':
            break
        result['assessment']=decision(result['records'],result['parent_observed_worker_wall_seconds'],result['peak_process_rss_bytes'],plan)
        if not result['assessment']['timing_pass'] or not result['assessment']['memory_pass']:
            break
    complete=(len(results)==6 and all(r['status']=='R54_V2_MICROBENCHMARK_COMPLETE' and r['assessment']['timing_pass'] and r['assessment']['memory_pass'] for r in results))
    parent_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    complete=complete and parent_rss<=4294967296
    write_new(args.output/'RESULT.json',dict(status='R54_V2_ENGINEERING_READY_TRAINING_NOT_AUTHORIZED' if complete else 'R54_V2_ENGINEERING_NO_GO',
        original_readiness='ENGINEERING_NO_GO',results=results,training_authorized=False,campaign_optimizer_updates=0,
        scratch_optimizer_steps=sum(r.get('scratch_optimizer_steps',0) for r in results),parent_peak_rss_bytes=parent_rss,
        v2_lock_sha256=sha(HERE/'LOCK.json'),independent_audit_status='PENDING'))
    write_new(args.output/'RECEIPT.json',dict(sha256={p.name:sha(p) for p in sorted(args.output.iterdir()) if p.is_file()}))
    # The independent audit does not update frozen results; final readiness is a separate audited verdict.
    from audit_qualification import audit
    try:
        audited=audit(args.output)
    except Exception as error:
        audited=dict(status='R54_V2_ENGINEERING_INVALID_EVIDENCE',error=repr(error),training_authorized=False)
    write_new(args.output/'INDEPENDENT_AUDIT.json',audited)
    write_new(args.output/'FINAL_RECEIPT.json',dict(sha256={p.name:sha(p) for p in sorted(args.output.iterdir()) if p.is_file()}))
    print(json.dumps(audited,indent=2,sort_keys=True))


if __name__=='__main__':
    main()
