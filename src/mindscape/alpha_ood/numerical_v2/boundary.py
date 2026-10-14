"""Current bytes/runtime must equal externally approved candidate commitments."""
import hashlib
import os
import platform
import sys
from pathlib import Path, PurePosixPath
import numpy as np
import torch
from ..benchmark import digest

REPO=Path(__file__).resolve().parents[4]
MANDATORY=frozenset({
    *('src/mindscape/alpha_ood/numerical_v2/'+name+'.py' for name in
      ('__init__','adapter','boundary','campaign','custody','evaluation')),
    'src/mindscape/alpha_ood/models.py','src/mindscape/alpha_ood/benchmark.py',
    'src/mindscape/alpha_ood/scoring.py','configs/alpha_ood/protocol-v1.json',
    'src/mindscape/r5_2_math/arithmetic.py','src/mindscape/r5_2_math/model.py',
    'src/mindscape/v3/physics_world/accelerator.py',
})


def verify_source_manifest(manifest,binding,*,root=REPO):
    """Flat repo-relative file -> lowercaseSHA256; no self-referential lock file."""
    if not isinstance(manifest,dict) or not MANDATORY.issubset(manifest):
        raise ValueError('source manifest mandatory coverage missing')
    if digest(manifest)!=binding['source_hash']:
        raise ValueError('source inventory commitment mismatch')
    root=Path(root).resolve()
    for relative,expected in manifest.items():
        if not isinstance(relative,str) or not relative or '\\' in relative:
            raise ValueError('invalid source path')
        parts=PurePosixPath(relative).parts
        if PurePosixPath(relative).is_absolute() or any(p in ('..','.') for p in parts) or str(PurePosixPath(relative))!=relative:
            raise ValueError('source traversal/noncanonical path')
        if not isinstance(expected,str) or len(expected)!=64 or any(c not in '0123456789abcdef' for c in expected):
            raise ValueError('source hash schema')
        current=root
        for part in parts:
            current=current/part
            if current.is_symlink():
                raise ValueError('source symlink forbidden')
        if not current.is_file() or root not in current.resolve().parents:
            raise ValueError('source file missing/outside repository')
        if hashlib.sha256(current.read_bytes()).hexdigest()!=expected:
            raise ValueError('active source bytes differ: '+relative)
    return digest(manifest)


def runtime_snapshot():
    executable=Path(sys.executable).resolve()
    return {'python_executable':str(executable),'python_executable_sha256':hashlib.sha256(executable.read_bytes()).hexdigest(),
            'python_version':sys.version,'torch_version':str(torch.__version__),
            'numpy_version':str(np.__version__),'platform':platform.platform(),
            'machine':platform.machine(),'mps_fallback':os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK','unset'),
            'default_dtype':str(torch.get_default_dtype())}


def verify_runtime_policy(policy,binding):
    if not isinstance(policy,dict) or set(policy)!= {'snapshot','device','dtype','status'}:
        raise ValueError('external runtime policy missing/schema')
    if digest(policy)!=binding['runtime_policy_hash']:
        raise ValueError('runtime policy commitment mismatch')
    if policy['status']!='PROPOSED_REQUIRES_REVIEW' or policy['device']!='mps' or policy['dtype']!='float32':
        raise ValueError('reviewed native float32 candidate policy required')
    if policy['snapshot']!=runtime_snapshot():
        raise ValueError('current executable/library/platform snapshot differs')
    if policy['snapshot']['mps_fallback'] not in ('unset','0') or torch.get_default_dtype()!=torch.float32:
        raise ValueError('fallback/dtype violates native numerical policy')
    from mindscape.v3.physics_world.accelerator import require_mps
    require_mps()
    return digest(policy)


def verify_boundary(manifest,policy,binding):
    verify_source_manifest(manifest,binding)
    verify_runtime_policy(policy,binding)
