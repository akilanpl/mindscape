"""Native Mac launch: verify MPS, resume missing evidence, then verify the package."""
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
STATE = Path('results/coding/research_continuation_v1')


def main():
    if not torch.backends.mps.is_available():
        raise RuntimeError('This execution environment cannot access MPS; launch in native Mac Terminal. CPU fallback is forbidden.')
    allocation = torch.ones(1, device='mps')
    torch.mps.synchronize()
    if allocation.device.type != 'mps':
        raise RuntimeError('Actual MPS allocation failed')
    with (STATE/'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = {**os.environ, 'PYTHONPATH':str(REPO/'src'),
           'MINDSCAPE_DEVICE':'mps', 'MINDSCAPE_PRECISION':'float16'}
    env.pop('PYTORCH_ENABLE_MPS_FALLBACK', None)
    # Watch progress without killing valid work or imposing a study deadline.
    with subprocess.Popen([sys.executable,'scripts/coding/complete_research.py'], env=env) as worker:
        last_warning = 0
        while worker.poll() is None:
            time.sleep(60)
            progress = STATE/'local_progress.json'
            if not progress.exists():
                continue
            launch = STATE/'launch.json'
            if not launch.exists():
                continue
            receipt = json.loads(launch.read_text())
            if receipt['pid'] != worker.pid:
                continue
            updated = max(json.loads(progress.read_text())['updated_unix'], receipt['started_unix'])
            now = time.time()
            if now-updated > 900 and now-last_warning > 900:
                last_warning = now
                print(f'WATCHDOG: no durable learning progress for {now-updated:.0f}s; '
                      f'worker PID {worker.pid}. Inspect MPS/memory; no automatic termination.', flush=True)
        if worker.returncode:
            raise subprocess.CalledProcessError(worker.returncode, worker.args)

    handoff = STATE/'finalization_handoff.json'
    if handoff.exists():
        preserved = STATE/'finalization_handoff.superseded_by_local_resume.json'
        if preserved.exists():
            raise RuntimeError('Preserved handoff already exists; inspect before modifying')
        handoff.rename(preserved)
    subprocess.run([sys.executable,'scripts/coding/finalize_research.py'], env=env, check=True)
    print('Evidence and package verified. Completion tag remains withheld for independent final audit and push.', flush=True)


if __name__ == '__main__':
    main()
