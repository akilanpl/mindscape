"""Separate bounded timing diagnostic. No optimizer, fit, official gate or evaluation."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / 'src'))


def sha(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def inspect_freeze():
    lock_path = REPO / 'experiments/r5_4/frozen/LOCK.json'
    lock = json.loads(lock_path.read_text())
    # Hash-only integrity reads: evaluation bytes are never decoded or given to a model.
    for name, expected in lock['sha256'].items():
        if sha(REPO / name) != expected:
            raise RuntimeError(f'Frozen artifact changed: {name}')
    return dict(lock_sha256=sha(lock_path), version=lock['version'], locked_files=len(lock['sha256']),
                official_engineering_sha256=sha(REPO / 'experiments/r5_4/engineering.json'),
                diagnostic_source_sha256=sha(__file__), schedule_sha256=sha(HERE / 'schedule.json'))


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def sequence(schedule, worker):
    base = schedule['worker_orders'][worker]
    for phase, count in [('cold', schedule['cold_rounds']), ('warmup', schedule['warmup_rounds']),
                         ('measured', schedule['measured_rounds'])]:
        for trial in range(count):
            order = base if trial % 2 == 0 else base[::-1]
            for slot, kind in enumerate(order):
                yield dict(phase=phase, trial=trial, slot=slot, order=order, kind=kind)


def distribution(values):
    if not values or any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError('Invalid timing distribution')
    ordered = sorted(values)
    def percentile(q):
        index = (len(ordered) - 1) * q
        lower, upper = math.floor(index), math.ceil(index)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)
    return dict(n=len(values), samples_seconds=values, mean=statistics.mean(values),
                median=statistics.median(values), sample_stdev=statistics.stdev(values) if len(values) > 1 else None,
                p90=percentile(.90), p95=percentile(.95), minimum=min(values), maximum=max(values))


def memory(torch):
    # macOS ru_maxrss is bytes. Driver allocation includes allocator caches/non-tensor allocations.
    return dict(process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                mps_live_tensor_bytes=torch.mps.current_allocated_memory(),
                mps_driver_allocated_bytes=torch.mps.driver_allocated_memory(),
                gpu_values_are_sampled_not_peak=True)


def checks(torch, model, snapshots):
    parameters = []
    for name, p in model.named_parameters():
        grad = p.grad
        ok = (p.device.type == 'mps' and p.dtype == torch.float32 and torch.isfinite(p).all().item()
              and grad is not None and grad.device.type == 'mps' and grad.dtype == torch.float32
              and torch.isfinite(grad).all().item() and grad.norm().item() > 0)
        unchanged = torch.equal(p.detach(), snapshots[name])
        parameters.append(dict(name=name, shape=list(p.shape), dtype=str(p.dtype), device=str(p.device),
                               gradient_dtype=str(grad.dtype) if grad is not None else None,
                               gradient_device=str(grad.device) if grad is not None else None,
                               gradient_norm=float(grad.norm().item()) if grad is not None else None,
                               checks_passed=bool(ok), weights_unchanged=unchanged))
    return dict(parameters=parameters, passed=all(p['checks_passed'] and p['weights_unchanged'] for p in parameters))


def worker_run(index, output, schedule):
    import torch
    original_optimizer_init = torch.optim.Optimizer.__init__
    def forbidden_optimizer(*args, **kwargs):
        raise RuntimeError('Optimizer construction forbidden by diagnostic')
    torch.optim.Optimizer.__init__ = forbidden_optimizer
    result = dict(label='NONTRAINING_MPS_DIAGNOSTIC_WORKER', worker=index,
                  base_order=schedule['worker_orders'][index], records=[], optimizer_instances=0,
                  optimizer_updates=0, cpu_fallback=False, status='UNRUN')
    try:
        result['freeze_before'] = inspect_freeze()
        result['runtime'] = dict(torch=torch.__version__, python=sys.version, platform=platform.platform(),
            machine=platform.machine(), mps_available=torch.backends.mps.is_available(),
            mps_built=torch.backends.mps.is_built(), fallback_env=os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0'))
        if (platform.system() != 'Darwin' or platform.machine() != 'arm64'
                or not result['runtime']['mps_available'] or result['runtime']['fallback_env'] != '0'):
            result['reason'] = 'Native arm64 MPS unavailable or fallback enabled; no CPU execution'
            return result
        from mindscape.r5_4.model import Transition, COUNTS
        from mindscape.r5_4.training import objective
        train = json.loads((REPO / 'experiments/r5_4/frozen/train.json').read_text())[:32]
        if len(train) != 32 or any(task['split'] != 'train' for task in train):
            raise RuntimeError('Training-only batch mismatch')
        result['batch_ids'] = [t['task_id'] for t in train]
        result['train_sha256'] = sha(REPO / 'experiments/r5_4/frozen/train.json')
        models, snapshots = {}, {}
        for kind in schedule['worker_orders'][index]:
            torch.manual_seed(11)
            models[kind] = Transition(kind).to(device='mps', dtype=torch.float32)
            if sum(p.numel() for p in models[kind].parameters()) != COUNTS[kind]:
                raise RuntimeError('Parameter count mismatch')
            snapshots[kind] = {name: p.detach().clone() for name, p in models[kind].named_parameters()}
        torch.mps.synchronize()
        result['initial_memory'] = memory(torch)
        for position in sequence(schedule, index):
            kind, model = position['kind'], models[position['kind']]
            model.zero_grad(set_to_none=True)  # new backward, no gradient accumulation
            torch.mps.synchronize()
            start = time.perf_counter_ns()
            loss, exposure = objective(model, train, 401)
            objective_return = time.perf_counter_ns()
            loss.backward()
            backward_return = time.perf_counter_ns()
            torch.mps.synchronize()
            finished = time.perf_counter_ns()
            # Total exactly includes original objective + backward + final synchronization.
            # Submission segments are CPU wall intervals, not independently synchronized GPU times.
            record = dict(position, total_seconds=(finished - start) / 1e9,
                objective_return_wall_seconds=(objective_return - start) / 1e9,
                backward_submission_wall_seconds=(backward_return - objective_return) / 1e9,
                final_sync_wall_seconds=(finished - backward_return) / 1e9,
                synchronized_before_and_after=True, objective_update=401, batch_size=32,
                parameter_count=COUNTS[kind], loss=float(loss.item()), loss_dtype=str(loss.dtype),
                memory_after_timing=memory(torch), checks=checks(torch, model, snapshots[kind]),
                visited_context_count=len(exposure))
            torch.mps.synchronize()
            record['memory_after_checks'] = memory(torch)
            record['rss_limit_exceeded'] = record['memory_after_checks']['process_peak_rss_bytes'] > 4 * 1024**3
            result['records'].append(record)
            # Partial evidence persists if later trials fail; never overwrite another run.
            with (output / f'worker-{index}.progress.jsonl').open('a') as stream:
                stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
            if not math.isfinite(record['loss']) or not record['checks']['passed']:
                raise RuntimeError('Gradient/parameter/finite/unchanged check failure')
            if record['rss_limit_exceeded']:
                raise RuntimeError('Process RSS exceeded frozen 4 GiB limit; stop diagnostic')
            del loss
        result['status'] = 'DIAGNOSTIC_COMPLETE'
    except Exception as error:
        result.update(status='DIAGNOSTIC_FAILED', reason=repr(error))
    finally:
        result['freeze_after'] = inspect_freeze()
        if result.get('freeze_before') != result['freeze_after']:
            result.update(status='DIAGNOSTIC_FAILED', reason='Frozen artifacts changed')
        result['peak_process_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result['rss_limit_exceeded'] = result['peak_process_rss_bytes'] > 4 * 1024**3
        torch.optim.Optimizer.__init__ = original_optimizer_init
    return result


def summarize(workers, schedule):
    if len(workers) != 4 or [w.get('worker') for w in workers] != list(range(4)) or any(w['status'] != 'DIAGNOSTIC_COMPLETE' for w in workers):
        exceeded = any(w.get('rss_limit_exceeded', False) for w in workers)
        partial = {kind: [r['total_seconds'] for w in workers for r in w.get('records', []) if r['kind'] == kind] for kind in ('markov', 'recurrent')}
        return dict(status='ENGINEERING_NO_GO' if exceeded else 'INCOMPLETE_NATIVE_DIAGNOSTIC',
                    partial_timing_distributions={kind: distribution(values) if values else None for kind, values in partial.items()},
                    cold_start_established=False, original_readiness='ENGINEERING_NO_GO',
                    process_rss_limit_exceeded=True if exceeded else None,
                    next_action='Propose separately frozen v2 engineering plan; no threshold relaxation' if exceeded else 'Review failed/incomplete evidence; no retries or qualification substitution')
    for w in workers:
        expected = list(sequence(schedule, w['worker']))
        if len(w['records']) != len(expected) or any(
                {k: row[k] for k in position} != position
                for row, position in zip(w['records'], expected)):
            raise ValueError('Incomplete/order-mismatched measurement schedule')
    per_worker, pooled = {}, {}
    for kind in ('markov', 'recurrent'):
        pooled[kind] = {}
        for phase in ('cold', 'warmup', 'measured'):
            pooled[kind][phase] = distribution([r['total_seconds'] for w in workers for r in w['records']
                                                if r['kind'] == kind and r['phase'] == phase])
        pooled[kind]['measured_by_slot'] = {str(slot): distribution([
            r['total_seconds'] for w in workers for r in w['records']
            if r['kind'] == kind and r['phase'] == 'measured' and r['slot'] == slot]) for slot in (0, 1)}
        pooled[kind]['mean_slot0_minus_slot1_seconds'] = pooled[kind]['measured_by_slot']['0']['mean'] - pooled[kind]['measured_by_slot']['1']['mean']
        pooled[kind]['measured_by_worker_base_order'] = {str(first): distribution([
            r['total_seconds'] for w in workers if w['base_order'][0] == first for r in w['records']
            if r['kind'] == kind and r['phase'] == 'measured']) for first in ('markov', 'recurrent')}
        pooled[kind]['mean_markov_first_minus_recurrent_first_seconds'] = (pooled[kind]['measured_by_worker_base_order']['markov']['mean'] - pooled[kind]['measured_by_worker_base_order']['recurrent']['mean'])
    cold_above = True
    warm_projected_within = True
    for w in workers:
        stats = {}
        for kind in ('markov', 'recurrent'):
            phases = {phase: distribution([r['total_seconds'] for r in w['records']
                if r['kind'] == kind and r['phase'] == phase]) for phase in ('cold', 'warmup', 'measured')}
            cold, mean, median, p95 = (phases['cold']['mean'], phases['measured']['mean'],
                                     phases['measured']['median'], phases['measured']['p95'])
            stats[kind] = dict(phases, cold_minus_warm_median_seconds=cold-median,
                cold_to_warm_median_ratio=cold/median, projected_warm_mean_seconds=mean*1200,
                projected_warm_p95_seconds=p95*1200,
                first_call_plus_799_warm_mean_times_margin_seconds=(cold+799*mean)*1.5)
            cold_above &= cold > median
            warm_projected_within &= p95*1200 <= 720
        per_worker[str(w['worker'])] = stats
    memory_exceeded = any(w['rss_limit_exceeded'] for w in workers)
    return dict(status='NONTRAINING_DIAGNOSIS_ONLY', per_worker=per_worker, pooled=pooled,
        cold_start_support=bool(cold_above and warm_projected_within and not memory_exceeded),
        all_worker_cold_above_warm_median=bool(cold_above), all_worker_warm_p95_projection_within_limit=bool(warm_projected_within),
        process_rss_limit_exceeded=memory_exceeded, original_readiness='ENGINEERING_NO_GO',
        warning='Diagnostic projections exclude optimizer steps and are not replacement qualification.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', type=int, choices=range(4), help=argparse.SUPPRESS)
    args = parser.parse_args()
    schedule = json.loads((HERE / 'schedule.json').read_text())
    if args.worker is not None:
        write_new(args.output / f'worker-{args.worker}.json', worker_run(args.worker, args.output, schedule))
        return
    before = inspect_freeze()
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / 'diagnostic-source.py').open('x') as stream:
        stream.write(Path(__file__).read_text())
        stream.flush()
        os.fsync(stream.fileno())
    write_new(args.output / 'schedule.json', schedule)
    write_new(args.output / 'START.json', dict(freeze=before, started_unix=time.time(), purpose='NONTRAINING_ONLY'))
    import torch
    runtime = dict(mps_available=torch.backends.mps.is_available(), mps_built=torch.backends.mps.is_built(),
                   fallback_env=os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0'), torch=torch.__version__)
    workers = []
    if not runtime['mps_available'] or runtime['fallback_env'] != '0':
        diagnosis = dict(status='NATIVE_DIAGNOSTIC_UNRUN', runtime=runtime,
            reason='MPS unavailable in this execution session; no CPU fallback or timings substituted',
            original_readiness='ENGINEERING_NO_GO', timings=None, process_rss_limit_exceeded=None,
            parent_peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    else:
        for index in range(4):
            try:
                subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', str(index),
                    '--output', str(args.output.resolve())], check=True, timeout=schedule['worker_timeout_seconds'],
                    cwd=REPO, env=dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK='0'))
                w = json.loads((args.output / f'worker-{index}.json').read_text())
            except (subprocess.SubprocessError, OSError, ValueError) as error:
                w = dict(worker=index, status='DIAGNOSTIC_FAILED', reason=repr(error), records=[])
            workers.append(w)
            if w['status'] != 'DIAGNOSTIC_COMPLETE':
                break  # fail closed, no retries after driver/check/memory/timeout failure
        diagnosis = summarize(workers, schedule)
    after = inspect_freeze()
    if before != after:
        raise RuntimeError('Original freeze changed during diagnostic')
    report = dict(label='R5.4-MPS-DIAGNOSTIC-v1-NONTRAINING', schedule=schedule, runtime=runtime,
        diagnosis=diagnosis, workers=workers, freeze_before=before, freeze_after=after,
        original_scientific_freeze_intact=True, optimizer_instances=0, optimizer_updates=0,
        official_engineering_gate_reruns=0, evaluated_ood_tasks=0, completed_unix=time.time())
    write_new(args.output / 'REPORT.json', report)
    write_new(args.output / 'RECEIPT.json', {'sha256': {p.name: sha(p) for p in sorted(args.output.iterdir()) if p.is_file()}})
    print(json.dumps(diagnosis, indent=2))


if __name__ == '__main__':
    main()
