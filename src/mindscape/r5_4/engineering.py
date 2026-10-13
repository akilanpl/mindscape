"""Native MPS qualification, backward only, never an optimizer or CPU fallback."""
import os
import platform
import resource
import time
import torch
from .data import ROOT, operand
from .model import Transition, COUNTS
from .training import objective


def check(tasks):
    result = dict(status='ENGINEERING_NO_GO', platform=platform.platform(), machine=platform.machine(),
                  torch=torch.__version__, mps_built=torch.backends.mps.is_built(),
                  mps_available=torch.backends.mps.is_available(), optimizer_updates=0, cpu_fallback=False)
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0') != '0':
        result['reason'] = 'MPS fallback must be disabled'
        return result
    if not result['mps_available'] or platform.system() != 'Darwin' or platform.machine() != 'arm64':
        result['reason'] = 'Native arm64 MPS unavailable; gradient and feasibility checks unrun'
        return result
    measurements = []
    try:
        for kind in COUNTS:
            torch.manual_seed(11)
            model = Transition(kind).to('mps')
            before = {k: v.detach().clone() for k, v in model.state_dict().items()}
            torch.mps.synchronize()
            start = time.monotonic()
            loss, _ = objective(model, tasks[:32], 401)
            loss.backward()
            torch.mps.synchronize()
            elapsed = time.monotonic() - start
            if any(p.device.type != 'mps' or p.grad is None or p.grad.device.type != 'mps'
                   or not torch.isfinite(p.grad).all().item() or p.grad.norm().item() == 0
                   for p in model.parameters()):
                raise RuntimeError('MPS gradient/device failure')
            if any(not torch.equal(v, before[k]) for k, v in model.state_dict().items()):
                raise RuntimeError('Qualification mutated parameters')
            measurements.append(dict(model=kind, parameters=COUNTS[kind], dtype='torch.float32',
                                     backward_seconds=elapsed, projected_fit_seconds=elapsed * 800 * 1.5))
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if peak > 4 * 1024**3 or any(m['projected_fit_seconds'] > 720 for m in measurements):
            raise RuntimeError('Frozen resource headroom insufficient')
        result.update(status='ENGINEERING_READY', gradients='finite, nonzero, MPS resident',
                      measurements=measurements, peak_process_rss_bytes=peak)
    except Exception as error:
        result.update(reason=str(error), measurements=measurements)
    return result
