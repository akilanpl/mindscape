"""Resume frozen native-MPS evidence without the superseded operational cutoff."""
import fcntl
import gc
import hashlib
import json
import os
import resource
import signal
import time
from pathlib import Path

import emergency_mps as study
import torch

from mindscape.coding.batching import ResidentCoder
from mindscape.coding.instrumentation import StageMeter
from mindscape.coding.schema import CodeTask

ROOT = study.BASE / 'research_continuation_v1'
ROOT.mkdir(parents=True, exist_ok=True)
CONDITIONS = ('model_only', 'structured', 'mindscape_b', 'mindscape_c')
STOP = False


def stop_requested(signum, frame):
    global STOP
    STOP = True
    print('Stop requested: finish the active checkpoint group, then exit.', flush=True)


def validate_rows(path, expected, key):
    rows = study.read(path)
    keys = [key(r) for r in rows]
    if len(keys) != len(set(keys)) or not set(keys) <= expected:
        raise RuntimeError(f'Duplicate or unexpected evidence keys: {path}')
    return set(keys)


def main():
    # Kernel-held lock releases automatically on normal exit, signal, or crash.
    with (ROOT / 'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        signal.signal(signal.SIGINT, stop_requested)
        signal.signal(signal.SIGTERM, stop_requested)
        protocol = json.loads((study.ROOT / 'locked_protocol.json').read_text())
        if (protocol['batch_size'], protocol['precision'], protocol['seed'], protocol['training_budget']) != (16, 'float16', 11, 100):
            raise RuntimeError('Unexpected frozen configuration')
        for path, digest in protocol['source_hashes'].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
                raise RuntimeError(f'Frozen source changed: {path}')
        for condition, digest in protocol['checkpoints'].items():
            if hashlib.sha256((study.adapter(condition) / 'adapter_model.safetensors').read_bytes()).hexdigest() != digest:
                raise RuntimeError(f'Frozen checkpoint changed: {condition}')
        task_path = study.BASE / 'completion_lockbox_v1/tasks.json'
        if hashlib.sha256(task_path.read_bytes()).hexdigest() != protocol['lockbox_sha256']:
            raise RuntimeError('Frozen tasks changed')
        tasks = [CodeTask(**r) for r in json.loads(task_path.read_text())]
        data = json.loads((study.BASE / 'final_dataset_v1/dataset.json').read_text())
        locked_path = study.BASE / 'completion_lockbox_eval_v1/rows.jsonl'
        curve_path = study.BASE / 'completion_gradient_v1/rows.jsonl'
        expected_locked = {(c, t.task_id) for c in CONDITIONS for t in tasks}
        expected_curve = {(c, s, n, split, r['task_id']) for c in CONDITIONS for s in (11,23,37)
                          for n in (10,25,50,100) for split in ('test','ood_test') for r in data[split]}
        validate_rows(locked_path, expected_locked, lambda r: (r['condition'],r['task_id']))
        done = validate_rows(curve_path, expected_curve, lambda r: tuple(r['key']))
        study.write(ROOT / 'launch.json', {'pid':os.getpid(), 'started_unix':time.time(),
                    'operational_deadline_removed_by_user':True, 'frozen_protocol_sha256':
                    hashlib.sha256((study.ROOT/'locked_protocol.json').read_bytes()).hexdigest(),
                    'initial_curve_rows':len(done), 'initial_curve_sha256':hashlib.sha256(curve_path.read_bytes()).hexdigest()})
        coder = ResidentCoder(study.MODEL, precision='float16')
        coder.change_precision('float16')
        coder.cpu_checkpoint_state = None
        coder.cpu_checkpoint_buffers = None
        gc.collect()
        torch.mps.empty_cache()
        generate = coder.generate_batch
        def measured(requests):
            result = generate(requests)
            study.append(ROOT/'resources.jsonl', {'time_unix':time.time(),'actual_batch_size':len(requests),
                         'device':str(coder.device),'precision':coder.precision,
                         'mps_live_bytes':torch.mps.current_allocated_memory(),
                         'mps_driver_bytes':torch.mps.driver_allocated_memory(),
                         'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
            return result
        coder.generate_batch = measured
        for condition in ('mindscape_c','mindscape_b','model_only','structured'):
            if STOP:
                return
            coder.use_adapter(study.adapter(condition))
            study.run_group(coder,tasks,condition,locked_path,16,None,
                            {'seed':11,'budget':100,'one_shot_configuration':True,
                             'adapter':str(study.adapter(condition)), 'protocol':'emergency_mps_v1/locked_protocol.json'})
        if validate_rows(locked_path,expected_locked,lambda r:(r['condition'],r['task_id'])) != expected_locked:
            raise RuntimeError('Locked evidence incomplete')
        study.write(locked_path.parent/'complete.json', {'complete':True,'tasks':100,'conditions':4,'episodes':400,
                    'configuration_frozen_before_outcomes':True})
        study.write(locked_path.parent/'scope.json', {'completed':400,'planned':400,'not_run':0,
                    'primary_100_task_comparison_complete':True,'four_condition_comparison_complete':True})
        study.write(locked_path.parent/'primary_complete.json', {'tasks':100,'conditions':['model_only','mindscape_c'],
                    'actual_saved_episodes':400,'four_condition_complete':True})
        for condition in ('mindscape_c','mindscape_b'):
            for seed in (11,23,37):
                for budget in (10,25,50,100):
                    if STOP:
                        return
                    pending = [CodeTask(**r) for split in ('test','ood_test') for r in data[split]
                               if (condition,seed,budget,split,r['task_id']) not in done]
                    if not pending:
                        continue
                    coder.use_adapter(study.adapter(condition,seed,budget))
                    partial = ROOT/f'curve_{condition}_{seed}_{budget}.jsonl'
                    study.run_group(coder,pending,condition,partial,16,None,
                                    {'seed':seed,'budget':budget,'adapter':str(study.adapter(condition,seed,budget)),
                                     'gradient_examples':budget,'retrieval_examples':budget})
                    for row in study.read(partial):
                        key = (condition,seed,budget,row['split'],row['task_id'])
                        if key not in done:
                            row['key'] = list(key)
                            study.append(curve_path,row)
                            done.add(key)
                    study.write(ROOT/'progress.json', {'curve_completed':len(done),'curve_remaining':1920-len(done)})
        if validate_rows(curve_path,expected_curve,lambda r:tuple(r['key'])) != expected_curve:
            raise RuntimeError('Learning evidence incomplete')
        study.write(curve_path.parent/'complete.json', {'complete':True,'episodes':1920,'mixed_hardware_precision':True})
        study.write(study.ROOT/'curve_scope.json', {'completed':1920,'planned':1920,'not_run':0,'status':'complete',
                    'cpu_rows_preserved':1122,'hardware_precision_stratification_required':True})
        latency_path = study.BASE/'latency_probe_v1/rows.jsonl'
        latency_path.parent.mkdir(parents=True,exist_ok=True)
        reps = [CodeTask(**data[split][index]) for index in range(2) for split in ('test','ood_test')]
        expected_latency = {(c,t.task_id) for c in CONDITIONS for t in reps}
        latency_done = validate_rows(latency_path,expected_latency,lambda r:(r['condition'],r['task_id']))
        for condition in CONDITIONS:
            coder.use_adapter(study.adapter(condition))
            coder.deadline = None
            for task in reps:
                if STOP:
                    return
                if (condition,task.task_id) in latency_done:
                    continue
                with StageMeter() as meter:
                    row = study.solve(task,coder,condition)
                row.update(stages=meter.summary(),cache_enabled=False,gpu_used=True,
                           peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                           mps_allocated_bytes=torch.mps.current_allocated_memory(),
                           mps_driver_bytes=torch.mps.driver_allocated_memory())
                study.append(latency_path,row)
        if validate_rows(latency_path,expected_latency,lambda r:(r['condition'],r['task_id'])) != expected_latency:
            raise RuntimeError('Dedicated latency incomplete')
        study.write(latency_path.parent/'complete.json', {'episodes':16,'fresh_generation':True,'cache_hits':0,
                    'tasks_per_condition':4,'percentiles':'P50/P95; small descriptive sample'})
        study.write(latency_path.parent/'scope.json', {'completed':16,'planned':16,'not_run':0,'status':'complete'})
        study.write(ROOT/'complete.json', {'lockbox':400,'learning':1920,'dedicated_latency':16,'time_unix':time.time()})
        print('NEURAL EVIDENCE COMPLETE',flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        study.write(ROOT/'failure.json', {'type':type(exc).__name__,'message':str(exc),'time_unix':time.time()})
        raise
