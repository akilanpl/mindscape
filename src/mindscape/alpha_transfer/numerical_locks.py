"""Concrete byte/runtime locks; candidate source commitments are externally signed."""
import hashlib
import os
from pathlib import Path
import platform
import sys
import numpy as np
import torch
from .readiness import digest

MANDATORY=('src/mindscape/alpha_transfer/numerical.py','src/mindscape/alpha_transfer/numerical_scoring.py','src/mindscape/alpha_transfer/numerical_authority.py','src/mindscape/alpha_transfer/numerical_locks.py','src/mindscape/alpha_transfer/readiness.py','src/mindscape/alpha_transfer/controller.py','src/mindscape/v5/model.py','src/mindscape/v5/schema.py','src/mindscape/v5/execute.py','src/mindscape/v5/data.py','src/mindscape/v3/physics_world/accelerator.py','configs/r5_kernel/shared-study-proposal.json')


def checked_path(root,relative):
    root=Path(root).resolve(); value=Path(relative)
    if value.is_absolute() or '..' in value.parts or not value.parts: raise ValueError('Unsafe source path')
    path=root/value
    cursor=root
    for part in value.parts:
        cursor=cursor/part
        if cursor.is_symlink(): raise ValueError('Symlink source path')
    if not path.is_file() or not path.resolve().is_relative_to(root): raise ValueError('Source path missing/outside root')
    return path


def verify_source(root,manifest):
    if not isinstance(manifest,dict) or not set(MANDATORY)<=set(manifest): raise ValueError('Mandatory source coverage missing')
    for relative,sha in manifest.items():
        if hashlib.sha256(checked_path(root,relative).read_bytes()).hexdigest()!=sha: raise ValueError('Source byte substitution')
    return True


def runtime_snapshot():
    fallback=os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK','0')
    if fallback!='0': raise ValueError('MPS fallback forbidden')
    executable=Path(sys.executable).resolve()
    return dict(python_executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),python_version=sys.version,torch_version=torch.__version__,numpy_version=np.__version__,platform=platform.platform(),machine=platform.machine(),mps_fallback='0')


def verify_locks(spec,mode='storage'):
    if mode not in ('storage','native'): raise ValueError('Runtime mode')
    manifest=spec['source_manifest']
    if digest(manifest)!=spec['bindings']['source']: raise ValueError('Source manifest binding')
    verify_source(spec['repository'],manifest)
    observed=runtime_snapshot()
    if observed!=spec['runtime_snapshot'] or digest(observed)!=spec['bindings']['runtime']: raise ValueError('Runtime substitution')
    if mode=='native':
        from mindscape.v3.physics_world.accelerator import require_mps
        require_mps()
    return True
