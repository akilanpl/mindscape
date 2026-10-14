"""v2 hashes, frozen plan, schedule and statistics; no neural execution."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        import os
        os.fsync(stream.fileno())


def verify():
    lock = load(HERE/'LOCK.json')
    actual_scripts={str(p.relative_to(REPO)) for p in HERE.glob('*.py')}
    locked_scripts={name for name in lock['sha256'] if name.startswith('engineering/r5_4_v2/') and name.endswith('.py')}
    if actual_scripts!=locked_scripts:
        raise ValueError('Incomplete engineering source inventory')
    for name, expected in lock['sha256'].items():
        if sha(REPO/name) != expected:
            raise ValueError(f'v2 freeze mismatch: {name}')
    original = load(REPO/'experiments/r5_4/frozen/LOCK.json')
    if sha(REPO/'experiments/r5_4/frozen/LOCK.json') != lock['v1_lock_sha256']:
        raise ValueError('Original v1 lock changed')
    for name, expected in original['sha256'].items():
        if sha(REPO/name) != expected:
            raise ValueError(f'Original v1 artifact changed: {name}')
    dirty = subprocess.check_output(['git','status','--porcelain','--',*lock['sha256'],*original['sha256'],
                                    'engineering/r5_4_v2/LOCK.json'],cwd=REPO,text=True)
    if dirty:
        raise ValueError('Frozen qualification inputs must be committed and clean')
    return lock


def rows(plan):
    batches = plan['batches']
    for phase, count in [('cold',1),('warmup',3),('measured',8)]:
        for trial in range(count):
            order = batches if trial % 2 == 0 else batches[::-1]
            for slot, batch in enumerate(order):
                yield dict(phase=phase, trial=trial, slot=slot, batch=batch)


def distribution(values):
    if not values or any(type(v) is not float or not math.isfinite(v) or v < 0 for v in values):
        raise ValueError('Invalid timing sample')
    ordered = sorted(values)
    pos = (len(values)-1)*.95
    lo, hi = math.floor(pos), math.ceil(pos)
    return dict(samples_seconds=values,n=len(values),mean=statistics.mean(values),median=statistics.median(values),
                sample_stdev=statistics.stdev(values) if len(values)>1 else None,
                p95=ordered[lo]+(ordered[hi]-ordered[lo])*(pos-lo),minimum=min(values),maximum=max(values))


def decision(records, worker_wall_seconds, peak_rss, plan):
    expected = list(rows(plan))
    if len(records) != 24 or any({k: row[k] for k in pos} != pos for row,pos in zip(records,expected)):
        raise ValueError('Incomplete qualification schedule')
    by_batch = {batch: {phase: distribution([r['full_update_seconds'] for r in records if r['batch']==batch and r['phase']==phase])
                        for phase in ('cold','warmup','measured')} for batch in plan['batches']}
    worst_p95 = max(v['measured']['p95'] for v in by_batch.values())
    projection = 1.5*(worker_wall_seconds+800*worst_p95)
    return dict(by_batch=by_batch,full_probe_worker_wall_seconds=worker_wall_seconds,
                worst_warm_p95_seconds=worst_p95,projected_fit_seconds=projection,
                process_peak_rss_bytes=peak_rss,timing_pass=projection<=720,memory_pass=peak_rss<=4294967296,
                original_readiness='ENGINEERING_NO_GO')
