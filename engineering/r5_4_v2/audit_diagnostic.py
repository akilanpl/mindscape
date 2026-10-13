"""Independent read-only audit of existing diagnostic evidence. Never imports torch or run.py."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

REPO = Path(__file__).resolve().parents[2]
SHAPES = {
    'markov': {'cell.0.weight': [32, 22], 'cell.0.bias': [32], 'cell.2.weight': [32, 32],
               'cell.2.bias': [32], 'head.weight': [12, 32], 'head.bias': [12]},
    'recurrent': {'cell.weight_ih': [48, 22], 'cell.weight_hh': [48, 16], 'cell.bias_ih': [48],
                  'cell.bias_hh': [48], 'head.weight': [12, 16], 'head.bias': [12]},
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(test, message):
    if not test:
        raise ValueError(message)


def compare(actual, expected, location='root'):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), f'{location}: keys mismatch')
        for key in expected:
            compare(actual[key], expected[key], f'{location}.{key}')
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), f'{location}: length mismatch')
        for i, value in enumerate(expected):
            compare(actual[i], value, f'{location}[{i}]')
    elif type(expected) is float:
        require(type(actual) is float and math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), f'{location}: numerical mismatch')
    else:
        require(type(actual) is type(expected) and actual == expected, f'{location}: discrete mismatch')


def stats(values):
    require(bool(values) and all(type(v) is float and math.isfinite(v) and v >= 0 for v in values), 'Invalid timing')
    sorted_values = sorted(values)
    def quantile(q):
        index = (len(values)-1)*q
        lo, hi = int(math.floor(index)), int(math.ceil(index))
        return sorted_values[lo]*(1-(index-lo)) + sorted_values[hi]*(index-lo)
    return dict(n=len(values), samples_seconds=values, mean=statistics.mean(values), median=statistics.median(values),
        sample_stdev=statistics.stdev(values) if len(values) > 1 else None, p90=quantile(.9), p95=quantile(.95),
        minimum=min(values), maximum=max(values))


def planned_rows(worker):
    base = ['markov', 'recurrent'] if worker % 2 == 0 else ['recurrent', 'markov']
    for phase, n in [('cold', 1), ('warmup', 3), ('measured', 8)]:
        for trial in range(n):
            order = base if trial % 2 == 0 else base[::-1]
            for slot, kind in enumerate(order):
                yield dict(phase=phase, trial=trial, slot=slot, kind=kind, order=order)


def summary(workers):
    pooled, per_worker = {}, {}
    cold_above, p95_ok = True, True
    for kind in SHAPES:
        pooled[kind] = {phase: stats([r['total_seconds'] for w in workers for r in w['records'] if r['kind'] == kind and r['phase'] == phase]) for phase in ('cold', 'warmup', 'measured')}
        pooled[kind]['measured_by_slot'] = {str(slot): stats([r['total_seconds'] for w in workers for r in w['records'] if r['kind'] == kind and r['phase'] == 'measured' and r['slot'] == slot]) for slot in (0, 1)}
        pooled[kind]['mean_slot0_minus_slot1_seconds'] = pooled[kind]['measured_by_slot']['0']['mean'] - pooled[kind]['measured_by_slot']['1']['mean']
        pooled[kind]['measured_by_worker_base_order'] = {first: stats([r['total_seconds'] for w in workers if w['base_order'][0] == first for r in w['records'] if r['kind'] == kind and r['phase'] == 'measured']) for first in SHAPES}
        pooled[kind]['mean_markov_first_minus_recurrent_first_seconds'] = pooled[kind]['measured_by_worker_base_order']['markov']['mean'] - pooled[kind]['measured_by_worker_base_order']['recurrent']['mean']
    for w in workers:
        per_worker[str(w['worker'])] = {}
        for kind in SHAPES:
            groups = {phase: stats([r['total_seconds'] for r in w['records'] if r['kind'] == kind and r['phase'] == phase]) for phase in ('cold', 'warmup', 'measured')}
            cold, warm = groups['cold']['mean'], groups['measured']
            per_worker[str(w['worker'])][kind] = dict(groups, cold_minus_warm_median_seconds=cold-warm['median'],
                cold_to_warm_median_ratio=cold/warm['median'], projected_warm_mean_seconds=warm['mean']*1200,
                projected_warm_p95_seconds=warm['p95']*1200,
                first_call_plus_799_warm_mean_times_margin_seconds=(cold+799*warm['mean'])*1.5)
            cold_above &= cold > warm['median']
            p95_ok &= warm['p95']*1200 <= 720
    exceeded = any(w['rss_limit_exceeded'] for w in workers)
    return dict(status='NONTRAINING_DIAGNOSIS_ONLY', per_worker=per_worker, pooled=pooled,
        cold_start_support=bool(cold_above and p95_ok and not exceeded), all_worker_cold_above_warm_median=bool(cold_above),
        all_worker_warm_p95_projection_within_limit=bool(p95_ok), process_rss_limit_exceeded=exceeded,
        original_readiness='ENGINEERING_NO_GO', warning='Diagnostic projections exclude optimizer steps and are not replacement qualification.')


def audit(directory):
    directory = Path(directory)
    require(directory.is_dir(), 'Missing diagnostic directory')
    receipt = json.loads((directory/'RECEIPT.json').read_text())
    expected_names = {'REPORT.json','START.json','schedule.json','diagnostic-source.py'} | {
        f'worker-{i}{suffix}' for i in range(4) for suffix in ('.json','.progress.jsonl')}
    require(set(receipt['sha256']) == expected_names, 'Incomplete receipt inventory')
    require({p.name for p in directory.iterdir()} == expected_names | {'RECEIPT.json'}, 'Unexpected/missing artifacts')
    for name, expected in receipt['sha256'].items():
        require(not (directory/name).is_symlink() and sha(directory/name) == expected, f'Artifact hash mismatch: {name}')
    source_hash = sha(REPO/'diagnostics/r5_4_mps_v1/run.py')
    schedule_hash = sha(REPO/'diagnostics/r5_4_mps_v1/schedule.json')
    require(sha(directory/'diagnostic-source.py') == source_hash, 'Diagnostic source differs from committed v1')
    canonical_schedule = json.dumps(json.loads((REPO/'diagnostics/r5_4_mps_v1/schedule.json').read_text()), indent=2, sort_keys=True, allow_nan=False) + '\n'
    require((directory/'schedule.json').read_bytes() == canonical_schedule.encode(), 'Diagnostic schedule differs from committed v1 semantics/serialization')
    lock_path = REPO/'experiments/r5_4/frozen/LOCK.json'
    lock = json.loads(lock_path.read_text())
    for name, expected in lock['sha256'].items():
        require(sha(REPO/name) == expected, f'Frozen source/data mismatch: {name}')
    expected_freeze = dict(version='R5.4-v1', locked_files=19, lock_sha256=sha(lock_path),
        official_engineering_sha256=sha(REPO/'experiments/r5_4/engineering.json'),
        diagnostic_source_sha256=source_hash, schedule_sha256=schedule_hash)
    report = json.loads((directory/'REPORT.json').read_text())
    compare(report['freeze_before'], expected_freeze)
    compare(report['freeze_after'], expected_freeze)
    compare(json.loads((directory/'START.json').read_text())['freeze'], expected_freeze)
    for key in ('optimizer_instances','optimizer_updates','official_engineering_gate_reruns','evaluated_ood_tasks'):
        require(type(report[key]) is int and report[key] == 0, f'Forbidden work: {key}')
    require(report['original_scientific_freeze_intact'] is True, 'Missing freeze assertion')
    require(report['runtime']['mps_available'] is True and report['runtime']['mps_built'] is True and report['runtime']['fallback_env'] == '0', 'Native runtime mismatch')
    schedule = json.loads((directory/'schedule.json').read_text())
    compare(report['schedule'], schedule)
    train = json.loads((REPO/'experiments/r5_4/frozen/train.json').read_text())[:32]
    ids = [t['task_id'] for t in train]
    workers = []
    memories = []
    norms = 0
    for index in range(4):
        w = json.loads((directory/f'worker-{index}.json').read_text())
        require(w['worker'] == index and w['status'] == 'DIAGNOSTIC_COMPLETE', 'Worker identity/completeness failure')
        require(w['base_order'] == schedule['worker_orders'][index], 'Worker order mismatch')
        compare(w['freeze_before'], expected_freeze)
        compare(w['freeze_after'], expected_freeze)
        require(w['batch_ids'] == ids and w['train_sha256'] == lock['sha256']['experiments/r5_4/frozen/train.json'], 'Training batch identity mismatch')
        require(w['runtime']['mps_available'] is True and w['runtime']['mps_built'] is True and w['runtime']['machine'] == 'arm64' and w['runtime']['fallback_env'] == '0', 'Worker native placement mismatch')
        require(w['optimizer_instances'] == w['optimizer_updates'] == 0 and w['cpu_fallback'] is False, 'Worker optimization/fallback failure')
        planned = list(planned_rows(index))
        require(len(w['records']) == len(planned) == 24, 'Worker call denominator mismatch')
        progress = [json.loads(line) for line in (directory/f'worker-{index}.progress.jsonl').read_text().splitlines()]
        compare(progress, w['records'])
        previous_rss = w['initial_memory']['process_peak_rss_bytes']
        memories.append(w['initial_memory'])
        for row, position in zip(w['records'], planned):
            compare({k: row[k] for k in position}, position)
            kind = row['kind']
            require(row['synchronized_before_and_after'] is True and row['objective_update'] == 401 and row['batch_size'] == 32, 'Measurement boundary/objective mismatch')
            require(type(row['loss']) is float and math.isfinite(row['loss']) and row['loss_dtype'] == 'torch.float32', 'Loss dtype/finite failure')
            require(row['parameter_count'] == sum(math.prod(shape) for shape in SHAPES[kind].values()), 'Parameter count mismatch')
            require(type(row['total_seconds']) is float and row['total_seconds'] > 0 and math.isfinite(row['total_seconds']), 'Timing invalid')
            segments = [row[k] for k in ('objective_return_wall_seconds','backward_submission_wall_seconds','final_sync_wall_seconds')]
            require(all(type(v) is float and math.isfinite(v) and v >= 0 for v in segments), 'Timing segment invalid')
            require(math.isclose(sum(segments), row['total_seconds'], rel_tol=0, abs_tol=1e-12), 'Timing segment sum mismatch')
            require(row['checks']['passed'] is True, 'Native parameter/gradient assertions failed')
            parameters = row['checks']['parameters']
            require(len(parameters) == 6 and {p['name']: p['shape'] for p in parameters} == SHAPES[kind], 'Parameter inventory/shape mismatch')
            for p in parameters:
                require(p['checks_passed'] is True and p['weights_unchanged'] is True, 'Unchanged/finite gradient assertion failure')
                require(p['dtype'] == p['gradient_dtype'] == 'torch.float32' and p['device'] == p['gradient_device'] == 'mps:0', 'Tensor dtype/placement mismatch')
                require(type(p['gradient_norm']) is float and math.isfinite(p['gradient_norm']) and p['gradient_norm'] > 0, 'Gradient norm failure')
                norms += 1
            for mem_key in ('memory_after_timing','memory_after_checks'):
                mem = row[mem_key]
                require(all(type(mem[k]) is int and mem[k] >= 0 for k in ('process_peak_rss_bytes','mps_live_tensor_bytes','mps_driver_allocated_bytes')), 'Memory units/type mismatch')
                require(mem['gpu_values_are_sampled_not_peak'] is True, 'GPU peak misattribution')
                require(mem['mps_live_tensor_bytes'] <= mem['mps_driver_allocated_bytes'], 'GPU allocation inconsistency')
                require(mem['process_peak_rss_bytes'] >= previous_rss, 'RSS high-water mark decreased')
                previous_rss = mem['process_peak_rss_bytes']
                memories.append(mem)
            require(row['rss_limit_exceeded'] is (previous_rss > 4294967296), 'RSS threshold attribution failure')
        require(type(w['peak_process_rss_bytes']) is int and w['peak_process_rss_bytes'] >= previous_rss, 'Final RSS inconsistent')
        require(w['rss_limit_exceeded'] is (w['peak_process_rss_bytes'] > 4294967296), 'Final RSS decision mismatch')
        workers.append(w)
    compare(report['workers'], workers)
    recomputed = summary(workers)
    compare(report['diagnosis'], recomputed)
    return dict(status='VERIFIED_RECORDED_NATIVE_DIAGNOSTIC', workers=4, total_objective_backward_calls=96,
        measured_warm_calls_per_model=32, parameter_gradient_assertions_checked=norms,
        receipt_sha256=sha(directory/'RECEIPT.json'), artifact_sha256={**receipt['sha256'], 'RECEIPT.json':sha(directory/'RECEIPT.json')},
        freeze=expected_freeze, diagnosis=recomputed,
        process_peak_rss_bytes=max(w['peak_process_rss_bytes'] for w in workers),
        worker_peak_rss_bytes=[w['peak_process_rss_bytes'] for w in workers],
        sampled_max_live_tensor_bytes=max(m['mps_live_tensor_bytes'] for m in memories),
        sampled_max_driver_allocated_bytes=max(m['mps_driver_allocated_bytes'] for m in memories),
        limitations=['Existing boolean tensor assertions and finite norms verified against frozen source and records; original raw gradient/parameter tensors were not saved, so tensor assertions are not independently regenerated.',
                    'File hashes and consistency are not hardware attestation or proof of actual process independence.',
                    'Cold-start effects are supported; the original 0.611934667-second event is not fully causally explained.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.directory), indent=2, sort_keys=True, allow_nan=False))
