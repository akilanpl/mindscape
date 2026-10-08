"""Resume missing learning cases on inspected Cloud hardware within eight hours."""
import fcntl
import gc
import json
import os
import signal
import time
from pathlib import Path

import emergency_mps as study
from cloud_runtime import CloudCoder, backend, resources, validate_frozen

from mindscape.coding.instrumentation import StageMeter
from mindscape.coding.policy import CodingPolicy, ToolCodingPolicy
from mindscape.coding.sandbox import WasiSandbox
from mindscape.coding.schema import CodeTask

ROOT=Path('results/coding/cloud_continuation_v1')
ROOT.mkdir(parents=True,exist_ok=True)


def solve(task,client,condition,options=None,seed=11,budget=100):
    sandbox=WasiSandbox('work/coding/runtime')
    policy=(ToolCodingPolicy(client,sandbox,study.memory(budget),**(options or {}))
            if condition.startswith('mindscape') else CodingPolicy(client,sandbox,condition))
    row=policy.solve(task,seed)
    timing=row['attempts'][0]['latency']
    row.update(condition=condition,split=task.metadata['split'],neural_device=timing['device'],
               model_precision=timing['precision'],generation_cache_enabled=False,
               execution_host='cloud:'+resources()['host'],
               physical_batch_sizes=[a['latency']['actual_batch_size'] for a in row['attempts']])
    return row


def main():
    validate_frozen()
    start_path=ROOT/'budget.json'
    if not start_path.exists():
        study.write(start_path,{'start_unix':time.time(),'maximum_seconds':28800})
    budget=json.loads(start_path.read_text());end=budget['start_unix']+28800
    if time.time()>=end-3600:
        raise RuntimeError('Insufficient remaining original Cloud budget for neural continuation')
    data=json.loads((study.BASE/'final_dataset_v1/dataset.json').read_text())
    curve=study.BASE/'completion_gradient_v1/rows.jsonl'
    done={tuple(r['key']) for r in study.read(curve)}
    if len(done)!=len(study.read(curve)):
        raise RuntimeError('Duplicate saved curve keys')
    name,precision=backend()
    study.write(ROOT/'hardware.json',{**resources(),'device':name,'precision':precision,
                'requested_original_precision':'float16','arithmetic_difference':precision!='float16',
                'reason':'CUDA uses frozen float16; CPU uses original supported float32 reference path; hardware/precision strata remain separate'})
    coder=CloudCoder(study.MODEL,name,precision)
    coder.deadline=min(time.time()+600,end-3600)
    coder.use_adapter(study.adapter('mindscape_b'))
    # Same exposed diagnostic tasks; no locked-outcome tuning or main-key rescoring.
    diagnostic_ids=[r['task_id'] for r in json.loads((study.BASE/'emergency_mps_v1/profile.json').read_text())['comparisons']]
    task_map={r['task_id']:CodeTask(**r) for split in ('test','ood_test') for r in data[split]}
    requests=[]
    for task_id in diagnostic_ids[:16]:
        capture=study.CaptureCoder()
        try:
            solve(task_map[task_id],capture,'mindscape_b')
        except study.CaptureRequest:
            requests.append(capture.request)
    trials=[]
    for size in (1,4,8,16):
        began=time.perf_counter()
        try:
            values=coder.generate_batch(requests[:size])
        except TimeoutError:
            break
        except RuntimeError as exc:
            if 'memory' not in str(exc).lower():
                raise
            break
        elapsed=time.perf_counter()-began
        trials.append({'batch_size':size,'seconds':elapsed,'requests_per_second':size/elapsed,
                       'first_response':values[0][0],'resources':resources()})
    if not trials:
        raise RuntimeError('No completed real Cloud inference benchmark')
    selected=max(trials,key=lambda r:r['requests_per_second'])['batch_size']
    study.write(ROOT/'profile.json',{'device':name,'precision':precision,'selected_batch_size':selected,
                'trials':trials,'first_response_batch_agreement':len({r['first_response'] for r in trials})==1,
                'interpretation':'Initial-request throughput, not complete-episode throughput'})
    original=coder.generate_batch
    def measured(requests):
        values=original(requests)
        study.append(ROOT/'resources.jsonl',{'time_unix':time.time(),'actual_batch_size':len(requests),
                     'device':name,'precision':precision,**resources()})
        return values
    coder.generate_batch=measured
    study.solve=solve;study.ROOT=ROOT;study.END=end
    for condition in ('mindscape_c','mindscape_b'):
        for seed in (11,23,37):
            for n in (10,25,50,100):
                if time.time()>=end-3600:
                    break
                pending=[CodeTask(**r) for split in ('test','ood_test') for r in data[split]
                         if (condition,seed,n,split,r['task_id']) not in done]
                if not pending:
                    continue
                coder.use_adapter(study.adapter(condition,seed,n))
                fragment=ROOT/f'curve_{condition}_{seed}_{n}.jsonl'
                try:
                    study.run_group(coder,pending,condition,fragment,selected,end-3600,
                                    {'seed':seed,'budget':n,'gradient_examples':n,'retrieval_examples':n,
                                     'adapter':str(study.adapter(condition,seed,n))})
                finally:
                    for row in study.read(fragment):
                        key=(condition,seed,n,row['split'],row['task_id'])
                        if key not in done:
                            row['key']=list(key);study.append(curve,row);done.add(key)
                    study.write(ROOT/'progress.json',{'learning_completed':len(done),'remaining':1920-len(done),
                                'time_remaining_seconds':end-time.time(),'updated_unix':time.time()})
    if len(done)==1920:
        study.write(curve.parent/'complete.json',{'complete':True,'episodes':1920,'mixed_hardware_precision':True})
    latency=study.BASE/'latency_probe_v1/rows.jsonl';latency.parent.mkdir(parents=True,exist_ok=True)
    reps=[CodeTask(**data[split][i]) for i in range(2) for split in ('test','ood_test')]
    def cutoff(signum,frame):
        raise TimeoutError('Reserved final audit/report budget reached')
    signal.signal(signal.SIGALRM,cutoff)
    for condition in ('model_only','structured','mindscape_b','mindscape_c'):
        coder.use_adapter(study.adapter(condition))
        for task in reps:
            if time.time()>=end-1800:
                break
            if any(r['condition']==condition and r['task_id']==task.task_id for r in study.read(latency)):
                continue
            signal.setitimer(signal.ITIMER_REAL,end-1800-time.time())
            try:
                with StageMeter() as meter:
                    row=solve(task,coder,condition)
            except TimeoutError:
                break
            finally:
                signal.setitimer(signal.ITIMER_REAL,0)
            row.update(stages=meter.summary(),cache_enabled=False,gpu_used=name=='cuda',
                       peak_rss_bytes=resources()['peak_process_rss_bytes'],**resources())
            study.append(latency,row)
    measured_count=len(study.read(latency))
    study.write(latency.parent/'scope.json',{'completed':measured_count,'planned':16,'not_run':16-measured_count,
                'status':'complete' if measured_count==16 else 'partial'})
    if measured_count==16:
        study.write(latency.parent/'complete.json',{'episodes':16,'fresh_generation':True,'cache_hits':0,'tasks_per_condition':4})
    study.write(ROOT/'neural_stop.json',{'learning_completed':len(done),'dedicated_latency':measured_count,
                'time_unix':time.time(),'budget_end_unix':end,'physical_batches':coder.batch_records})
    del coder;gc.collect()


if __name__=='__main__':
    with (ROOT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            main()
        except Exception as exc:
            study.write(ROOT/'failure.json',{'type':type(exc).__name__,'message':str(exc),'time_unix':time.time(),'pid':os.getpid()})
            raise
