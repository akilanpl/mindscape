"""One-shot real Kaggle CUDA capability probe. Never trains outside Kaggle."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require_kaggle():
    if not Path('/kaggle/working').is_dir() or not (
        os.environ.get('KAGGLE_KERNEL_RUN_TYPE') or os.environ.get('KAGGLE_KERNEL_INTEGRATIONS')
    ):
        raise RuntimeError('Real Kaggle kernel required; local neural fitting is forbidden')


def main(output='/kaggle/working/cuda-probe'):
    require_kaggle()
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    receipt = {'schema_version': 1, 'status': 'failed', 'optimizer_updates': 0,
               'device': 'cuda:0', 'started_unix': time.time(), 'artifacts': {}}
    try:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        import torch
        receipt.update(python=platform.python_version(), torch=str(torch.__version__),
            cuda_runtime=torch.version.cuda, cuda_available=torch.cuda.is_available(),
            cuda_device_count=torch.cuda.device_count(), cpu_count=os.cpu_count(),
            kaggle_run_type=os.environ.get('KAGGLE_KERNEL_RUN_TYPE'),
            nvidia_device_nodes=[str(p) for p in Path('/dev').glob('nvidia*')],
            memory=Path('/proc/meminfo').read_text() if Path('/proc/meminfo').exists() else None)
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=20)
            receipt.update(nvidia_smi=result.stdout, nvidia_smi_error=result.stderr,
                           nvidia_smi_returncode=result.returncode)
        except (OSError, subprocess.SubprocessError) as error:
            receipt['nvidia_smi_error'] = type(error).__name__
        print(json.dumps({'environment_before_updates': receipt}), flush=True)
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable; CPU substitution forbidden')
        torch.cuda.set_device(0)
        torch.manual_seed(17)
        torch.cuda.manual_seed_all(17)
        torch.use_deterministic_algorithms(True)
        receipt.update({
            'python': platform.python_version(), 'platform': platform.platform(),
            'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
            'cudnn': torch.backends.cudnn.version(),
            'cuda_available': True, 'cuda_device_count': torch.cuda.device_count(),
            'selected_device': 0, 'gpu_devices': [
                {'index': i, 'name': torch.cuda.get_device_name(i),
                 'total_memory_bytes': torch.cuda.get_device_properties(i).total_memory,
                 'compute_capability': list(torch.cuda.get_device_capability(i))}
                for i in range(torch.cuda.device_count())],
            'cpu_count': os.cpu_count(),
            'memory': Path('/proc/meminfo').read_text() if Path('/proc/meminfo').exists() else None,
            'storage': {str(p): dict(zip(('total','used','free'), shutil.disk_usage(p)))
                        for p in (Path('/kaggle/working'), Path('/kaggle/input')) if p.exists()},
            'kaggle_run_type': os.environ.get('KAGGLE_KERNEL_RUN_TYPE'),
        })
        model = torch.nn.Linear(4, 1).to('cuda:0')
        initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        x = torch.arange(16, device='cuda:0', dtype=torch.float32).reshape(4, 4) / 16
        y = x.sum(dim=1, keepdim=True)
        losses = []
        for _ in range(5):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(x), y)
            loss.backward()
            optimizer.step()
            receipt['optimizer_updates'] += 1
            losses.append(float(loss.detach().cpu()))
        torch.cuda.synchronize()
        changed = any(not torch.equal(initial[k], v) for k, v in model.state_dict().items())
        if not changed:
            raise RuntimeError('CUDA optimizer did not change weights')
        checkpoint = out / 'cuda_probe.pt'
        torch.save({'model_state_dict': model.state_dict(), 'optimizer_updates': 5}, checkpoint)
        reloaded = torch.nn.Linear(4, 1).to('cuda:0')
        payload = torch.load(checkpoint, map_location='cuda:0', weights_only=True)
        reloaded.load_state_dict(payload['model_state_dict'])
        reload_ok = all(torch.equal(a, b) for a, b in zip(model.parameters(), reloaded.parameters()))
        if not reload_ok or payload['optimizer_updates'] != 5:
            raise RuntimeError('Checkpoint reload verification failed')
        receipt.update(status='passed', weights_changed=changed, reload_succeeded=reload_ok,
                       losses=losses, max_cuda_memory_bytes=torch.cuda.max_memory_allocated(0))
        receipt['artifacts'][checkpoint.name] = sha256(checkpoint)
        try:
            receipt['nvidia_smi'] = subprocess.check_output(['nvidia-smi'], text=True)
        except (OSError, subprocess.SubprocessError) as error:
            receipt['nvidia_smi_error'] = type(error).__name__
    except Exception as error:
        receipt['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        receipt['finished_unix'] = time.time()
        (out / 'cuda_probe_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
        print(json.dumps({'cuda_probe_receipt': receipt}), flush=True)


if __name__ == '__main__':
    main()
