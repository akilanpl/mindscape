"""Independent read-only v2 qualification verification, tensor inspection only; never a fit."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from common import REPO,HERE,load,sha,verify
from audit_diagnostic import SHAPES,require,compare


def independent_distribution(values):
    import statistics
    ordered=sorted(values)
    require(values and all(type(v) is float and math.isfinite(v) and v>=0 for v in values),'Invalid update timings')
    q=(len(values)-1)*.95
    lo,hi=math.floor(q),math.ceil(q)
    return dict(samples_seconds=values,n=len(values),mean=statistics.mean(values),median=statistics.median(values),
                sample_stdev=statistics.stdev(values) if len(values)>1 else None,
                p95=ordered[lo]*(hi-q)+ordered[hi]*(q-lo) if lo!=hi else ordered[lo],minimum=min(values),maximum=max(values))


def independent_rows(plan):
    for phase,n in [('cold',1),('warmup',3),('measured',8)]:
        for trial in range(n):
            ordered=plan['batches'] if trial%2==0 else plan['batches'][::-1]
            for slot,batch in enumerate(ordered):
                yield dict(phase=phase,trial=trial,slot=slot,batch=batch)


def tensor_hash(mapping):
    h=hashlib.sha256()
    for name,t in mapping.items():
        h.update(name.encode())
        h.update(t.contiguous().numpy().tobytes())
    return h.hexdigest()


def audit(directory):
    verify()
    directory=Path(directory)
    require(all(p.is_file() and not p.is_symlink() for p in directory.iterdir()),'Unexpected evidence entry')
    receipt=load(directory/'RECEIPT.json')
    known={p.name for p in directory.iterdir() if p.is_file()}-{'RECEIPT.json','INDEPENDENT_AUDIT.json','FINAL_RECEIPT.json'}
    require(set(receipt['sha256'])==known,'Qualification receipt inventory mismatch')
    for name,expected in receipt['sha256'].items():
        require(Path(name).name==name and not (directory/name).is_symlink() and sha(directory/name)==expected,'Qualification artifact hash mismatch')
    if (directory/'FINAL_RECEIPT.json').exists():
        final=load(directory/'FINAL_RECEIPT.json')
        require(set(final['sha256'])=={p.name for p in directory.iterdir() if p.is_file()}-{'FINAL_RECEIPT.json'},'Final receipt inventory mismatch')
        for name,expected in final['sha256'].items():
            require(Path(name).name==name and sha(directory/name)==expected,'Final receipt hash mismatch')
    plan=load(HERE/'plan.json')
    start=load(directory/'START.json')
    require(start['v2_lock_sha256']==sha(HERE/'LOCK.json') and start['approved_engineering_only'] is True and start['training_authorized'] is False,'Qualification approval/provenance mismatch')
    compare(start['plan'],plan)
    result=load(directory/'RESULT.json')
    require(result['v2_lock_sha256']==sha(HERE/'LOCK.json') and result['original_readiness']=='ENGINEERING_NO_GO' and result['training_authorized'] is False and result['campaign_optimizer_updates']==0,'Qualification scope mismatch')
    if len(result['results'])!=6 or any(w['status']!='R54_V2_MICROBENCHMARK_COMPLETE' for w in result['results']):
        require(result['status']=='R54_V2_ENGINEERING_NO_GO','Incomplete work cannot qualify')
        return dict(status='R54_V2_ENGINEERING_NO_GO',independent_audit='PARTIAL_REJECTION_ONLY',
            original_readiness='ENGINEERING_NO_GO',training_authorized=False,campaign_optimizer_updates=0,
            reason='Failed or incomplete qualification; no readiness claim, full tensor/metric verification unrun',
            recorded_scratch_optimizer_steps=result['scratch_optimizer_steps'],receipt_sha256=sha(directory/'RECEIPT.json'))
    train=load(REPO/'experiments/r5_4/frozen/train.json')
    batch_ids={'first32':[t['task_id'] for t in train[:32]],'first32_four_digit_training_tasks':[
        t['task_id'] for t in train if len(t['left'])==len(t['right'])==4][:32]}
    import torch  # CPU storage-only inspection, no model construction, forwards, backward or optimizer
    assessments=[]
    sampled_gpu=[]
    count=0
    for index,(kind,seed) in enumerate(plan['configurations']):
        w=load(directory/f'worker-{index}.json')
        combined=result['results'][index]
        compare({k:combined[k] for k in w},w)
        require((w['config_index'],w['kind'],w['seed'])==(index,kind,seed) and w['status']=='R54_V2_MICROBENCHMARK_COMPLETE','Qualification attribution/status mismatch')
        require(w['v2_lock_sha256']==sha(HERE/'LOCK.json') and w['campaign_optimizer_updates']==0 and w['scratch_optimizer_steps']==24,'Scratch/campaign count mismatch')
        require(w['runtime']['mps_available'] is True and w['runtime']['mps_built'] is True and w['runtime']['cpu_fallback'] is False and w['runtime']['machine']=='arm64','Native placement mismatch')
        require(w['batch_ids']==batch_ids and w['parameter_count']==sum(math.prod(s) for s in SHAPES[kind].values()),'Batch/parameter identity mismatch')
        planned=list(independent_rows(plan))
        require(len(w['records'])==24,'Incomplete trial records')
        progress=[json.loads(line) for line in (directory/f'worker-{index}.progress.jsonl').read_text().splitlines()]
        compare(progress,w['records'])
        scratch_progress=[json.loads(line) for line in (directory/f'worker-{index}.scratch-progress.jsonl').read_text().splitlines()]
        require(len(scratch_progress)==24,'Missing step progress')
        tensor_path=directory/f'worker-{index}.restoration-and-gradients.pt'
        require(sha(tensor_path)==w['tensor_evidence_sha256'],'Tensor evidence hash mismatch')
        tensors=torch.load(tensor_path,map_location='cpu',weights_only=True)
        require({k:list(v.shape) for k,v in tensors['initial'].items()}==SHAPES[kind],'Initial tensor shapes mismatch')
        require(set(tensors['initial'])==set(tensors['restored']),'Restoration tensor inventory mismatch')
        require(all(t.dtype==torch.float32 and torch.isfinite(t).all().item() for t in tensors['initial'].values()),'Initial tensor dtype/finite mismatch')
        require(all(torch.equal(t,tensors['restored'][k]) for k,t in tensors['initial'].items()),'Independent bitwise restoration failure')
        require(tensor_hash(tensors['initial'])==tensor_hash(tensors['restored'])==w['initial_parameter_sha256']==w['final_parameter_sha256'] and w['final_weights_restored'] is True,'Restoration hash mismatch')
        require(set(tensors['gradients'])=={str(i) for i in range(24)},'Gradient tensor evidence incomplete')
        opt=tensors['optimizer_state']
        require(len(opt['state'])==6 and len(opt['param_groups'])==1,'Optimizer state incomplete')
        group=opt['param_groups'][0]
        require(group['lr']==.003 and group['weight_decay']==.0001 and group['betas']==(.9,.999),'Frozen optimizer policy mismatch')
        require(all(torch.isfinite(t).all().item() and t.count_nonzero().item()==0 for state in opt['state'].values() for t in state.values()),'Independent optimizer restoration failure')
        previous_rss=0
        for call_id,(row,expected) in enumerate(zip(w['records'],planned)):
            compare({k:row[k] for k in expected},expected)
            require(row['call_id']==call_id and row['synchronized_before_and_after'] is True,'Call ID/synchronization mismatch')
            compare(scratch_progress[call_id],dict(microbenchmark_call_id=call_id,objective_update_mode=401,loss=row['loss'],dtype=row['loss_dtype']))
            require(math.isfinite(row['loss']) and row['loss_dtype']=='torch.float32','Loss finite/dtype failure')
            require(row['weights_restored_exactly'] is True and row['optimizer_state_zero_after_restore'] is True and row['scratch_step_counter_before']==0 and row['scratch_step_counter_after']==1,'Scratch restoration assertions failed')
            require(row['restored_parameter_sha256']==row['initial_parameter_sha256']==w['initial_parameter_sha256'] and row['after_step_parameter_sha256']!=row['initial_parameter_sha256'],'Optimizer update workload/restoration mismatch')
            gradients=tensors['gradients'][str(call_id)]
            require({k:list(v.shape) for k,v in gradients.items()}==SHAPES[kind],'Gradient shape mismatch')
            require({m['name']:m['shape'] for m in row['gradients']}==SHAPES[kind],'Gradient metadata incomplete')
            for meta in row['gradients']:
                g=gradients[meta['name']]
                require(g.dtype==torch.float32 and torch.isfinite(g).all().item() and g.norm().item()>0,'Independent gradient dtype/finite failure')
                require(meta['device']=='mps:0' and meta['dtype']=='torch.float32' and math.isfinite(meta['norm']) and meta['norm']>0,'Recorded gradient residency/norm failure')
                require(math.isclose(g.norm().item(),meta['norm'],rel_tol=1e-5,abs_tol=1e-8),'CPU/MPS float32 gradient norm discrepancy exceeds declared tolerance')
            require(len(row['optimizer_state_metadata'])==6,'State metadata incomplete')
            for state in row['optimizer_state_metadata']:
                require(set(state)=={'step','exp_avg','exp_avg_sq'} and state['exp_avg']['device']==state['exp_avg_sq']['device']=='mps:0','Optimizer compute placement mismatch')
                require(all(meta['dtype']=='torch.float32' for meta in state.values()),'Optimizer state dtype mismatch')
            mem=row['memory']
            require(all(type(mem[k]) is int and mem[k]>=0 for k in ('process_peak_rss_bytes','sampled_live_tensor_bytes','sampled_driver_allocated_bytes')),'Memory metadata units failure')
            require(mem['process_peak_rss_bytes']>=previous_rss and mem['sampled_live_tensor_bytes']<=mem['sampled_driver_allocated_bytes'],'Memory counter inconsistency')
            previous_rss=mem['process_peak_rss_bytes']
            sampled_gpu.append(mem)
            count+=1
        require(w['peak_process_rss_bytes']>=previous_rss,'Final RSS inconsistent')
        wall=combined['parent_observed_worker_wall_seconds']
        require(type(wall) is float and math.isfinite(wall) and wall>=sum(r['full_update_seconds'] for r in w['records']) and wall<=90,'Worker wall/startup accounting invalid')
        stats={batch:{phase:independent_distribution([r['full_update_seconds'] for r in w['records'] if r['batch']==batch and r['phase']==phase]) for phase in ('cold','warmup','measured')} for batch in plan['batches']}
        worst=max(d['measured']['p95'] for d in stats.values())
        projected=(wall+800*worst)*1.5
        assessment=dict(by_batch=stats,full_probe_worker_wall_seconds=wall,worst_warm_p95_seconds=worst,
            projected_fit_seconds=projected,process_peak_rss_bytes=w['peak_process_rss_bytes'],
            timing_pass=projected<=720,memory_pass=w['peak_process_rss_bytes']<=4294967296,original_readiness='ENGINEERING_NO_GO')
        compare(combined['assessment'],assessment)
        assessments.append(assessment)
    require(count==result['scratch_optimizer_steps']==144,'Total scratch workload mismatch')
    passed=all(a['timing_pass'] and a['memory_pass'] for a in assessments) and result['parent_peak_rss_bytes']<=4294967296
    expected_status='R54_V2_ENGINEERING_READY_TRAINING_NOT_AUTHORIZED' if passed else 'R54_V2_ENGINEERING_NO_GO'
    require(result['status']==expected_status,'Qualification decision mismatch')
    verdict=dict(status=expected_status,independent_audit='VERIFIED',original_readiness='ENGINEERING_NO_GO',
        assessments=assessments,training_authorized=False,campaign_optimizer_updates=0,scratch_optimizer_steps=144,
        maximum_process_rss_bytes=max([a['process_peak_rss_bytes'] for a in assessments]+[result['parent_peak_rss_bytes']]),
        sampled_max_live_tensor_bytes=max(m['sampled_live_tensor_bytes'] for m in sampled_gpu),
        sampled_max_driver_allocated_bytes=max(m['sampled_driver_allocated_bytes'] for m in sampled_gpu),
        v2_lock_sha256=sha(HERE/'LOCK.json'),receipt_sha256=sha(directory/'RECEIPT.json'),
        limitations=['Per-trial native optimizer write/hash assertions are recorded; raw initial/restored parameters and clipped gradients are independently inspected.',
                    'Diagnostic estimates are conservative resource screens, not a completed 800-update fit or evidence of learning.'])
    if (directory/'INDEPENDENT_AUDIT.json').exists():
        compare(load(directory/'INDEPENDENT_AUDIT.json'),verdict)
    return verdict


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('directory',type=Path)
    args=parser.parse_args()
    print(json.dumps(audit(args.directory),indent=2,sort_keys=True,allow_nan=False))
