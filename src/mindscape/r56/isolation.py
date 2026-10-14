"""Fail-closed staging and container contracts; never launches a training campaign."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

IMAGE = re.compile(r'^[a-zA-Z0-9./_-]+@sha256:[0-9a-f]{64}$')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_file(root, name):
    root=Path(root).resolve()
    if not isinstance(name,str) or not name or Path(name).is_absolute() or '..' in Path(name).parts:
        raise ValueError('Only safe relative input paths allowed')
    raw=root/name
    # Reject links at every path component, including a link which resolves inside.
    cursor=root
    for part in Path(name).parts:
        cursor=cursor/part
        if cursor.is_symlink():raise ValueError('Symlink input forbidden')
    if not raw.is_file() or not raw.resolve().is_relative_to(root):
        raise ValueError('Staging input must be a regular confined file')
    return raw


def stage_training(package, destination, seed, expected_package_hash):
    """Copy ONLY approved train-file bytes after hash/type checks; never seal targets here.

    No evaluator data, target generators, verifiers or repository source mount is
    permitted in the trainer boundary. This is engineering staging, not fitting.
    """
    package=Path(package).resolve();manifest=json.loads(checked_file(package,'manifest.json').read_text())
    payload=dict(manifest);claimed=payload.pop('package_sha256',None)
    actual=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    if claimed!=actual or claimed!=expected_package_hash:
        raise ValueError('Independently pinned proposal identity mismatch')
    if seed not in (17,29,43):raise ValueError('Unknown reserved seed')
    entry=manifest['index'][str(seed)]
    source=checked_file(package,entry['training_file'])
    if sha(source)!=entry['sha256']:raise ValueError('Training input tampered')
    rows=json.loads(source.read_bytes())
    if not isinstance(rows,list) or len(rows)!=512 or any(r.get('split')!='train' for r in rows):
        raise ValueError('Unexpected non-training input')
    destination=Path(destination)
    if destination.exists():raise FileExistsError('Staging destination must be new')
    destination.mkdir(parents=True,mode=0o700)
    try:
        target=destination/'train.json'
        with source.open('rb') as src,target.open('xb') as dst:shutil.copyfileobj(src,dst)
        if sha(target)!=entry['sha256']:raise ValueError('Training source changed during copy')
        target.chmod(0o444)
        receipt={'package_sha256':expected_package_hash,'seed':seed,'files':{'train.json':sha(target)},'heldout_mounts':[],
                 'source_modules_mounted':[], 'training_authorized':False}
        (destination/'staging.json').write_text(json.dumps(receipt,indent=2))
        (destination/'staging.json').chmod(0o444)
        destination.chmod(0o555)
        return receipt
    except Exception:
        # Keep partial staging as failure evidence; never treat it as qualified.
        raise


def verify_stage(stage):
    stage=Path(stage).resolve()
    receipt=json.loads(checked_file(stage,'staging.json').read_text())
    expected={'train.json','staging.json'}
    if {p.name for p in stage.iterdir()}!=expected:raise ValueError('Unexpected trainer mount contents')
    if receipt.get('heldout_mounts')!=[] or receipt.get('source_modules_mounted')!=[]:
        raise ValueError('Trainer mount includes forbidden data/source')
    if sha(checked_file(stage,'train.json'))!=receipt['files']['train.json']:
        raise ValueError('Staging tampered')
    return receipt


def container_command(image, stage, probe, role='trainer', evaluator_data=None, checkpoint_dir=None):
    """Shell-free command. Trainer never mounts checkpoint feedback or evaluator data."""
    if not IMAGE.fullmatch(image):raise ValueError('Reviewed immutable image digest required')
    if role not in ('trainer','evaluator'):raise ValueError('Unknown isolated role')
    if Path(probe).is_symlink():raise ValueError('Probe symlink forbidden')
    stage=Path(stage).resolve();probe=Path(probe).resolve()
    if ',' in str(stage) or ',' in str(probe):raise ValueError('Mount path delimiter forbidden')
    if probe.is_symlink() or not probe.is_file():raise ValueError('Probe must be trusted regular source')
    verify_stage(stage)
    if role=='trainer' and (evaluator_data is not None or checkpoint_dir is not None):raise ValueError('Trainer cannot mount evaluator data')
    args=['docker','run','--rm','--network','none','--read-only','--cap-drop','ALL',
          '--security-opt','no-new-privileges','--user','10001:10001','--pids-limit','32',
          '--memory','512m','--cpus','1','--ulimit','nofile=64:64',
          '--tmpfs','/tmp:rw,noexec,nosuid,size=16m','--workdir','/tmp',
          '--mount',f'type=bind,src={stage},dst=/training,readonly',
          '--mount',f'type=bind,src={probe},dst=/probe.py,readonly']
    if role=='evaluator':
        if evaluator_data is None or checkpoint_dir is None:raise ValueError('Evaluator-controlled data and frozen checkpoint mounts required')
        data=Path(evaluator_data)
        if data.is_symlink() or not data.is_dir() or ',' in str(data.resolve()):raise ValueError('Invalid evaluator vault')
        checkpoint=Path(checkpoint_dir)
        if checkpoint.is_symlink() or not checkpoint.is_dir() or ',' in str(checkpoint.resolve()):raise ValueError('Invalid frozen checkpoint vault')
        for root in (data,checkpoint):
            if any(p.is_symlink() for p in root.rglob('*')):raise ValueError('Vault symlinks forbidden')
        args+=['--mount',f'type=bind,src={data.resolve()},dst=/sealed-evaluation,readonly',
               '--mount',f'type=bind,src={checkpoint.resolve()},dst=/checkpoints,readonly']
    return args+[image,'python','-I','-B','/probe.py']+(['evaluator'] if role=='evaluator' else [])


def run_probe(image, stage, probe):
    command=container_command(image,stage,probe)
    if not shutil.which('docker'):
        return {'qualified':False,'status':'BLOCKED','reason':'Docker executable/runtime unavailable; OS boundary NOT demonstrated',
                'command_contract':command,'optimizer_updates':0}
    # Exclude host env credentials and inherited descriptors; never invoke a shell.
    env={k:os.environ[k] for k in ('PATH','SYSTEMROOT') if k in os.environ}
    completed=subprocess.run(command,env=env,close_fds=True,capture_output=True,text=True,timeout=30)
    if completed.returncode:
        return {'qualified':False,'status':'FAILED','exit_code':completed.returncode,'stderr':completed.stderr[:2000]}
    result=json.loads(completed.stdout)
    required=('heldout_denied','traversal_denied','symlink_denied','environment_clean',
              'inherited_descriptors_clean','readonly_training','nonroot','no_network','training_readable')
    passed=all(result.get(name) is True for name in required)
    return {'qualified':False,'boundary_probe_passed':passed,'status':'TRAINER_PROBE_ONLY' if passed else 'FAILED',
            'probe':result,'image':image,'training_authorized':False,
            'scope':'Container file/env/FD/network probe only; not a scientific fit or evaluator qualification'}


def run_isolation_pair(image, stage, probe, evaluator_data, checkpoint_dir, expected_target_hash, expected_checkpoint_hash, expected_probe_hash):
    """Positive evaluator+negative trainer controls; always an engineering receipt.

    Requires independently owned immutable staging. Path validation cannot prevent
    malicious same-UID host races; this implementation does NOT qualify that case.
    """
    if sha(probe)!=expected_probe_hash:raise ValueError('Probe source identity mismatch')
    trainer=run_probe(image,stage,probe)
    if not trainer.get('boundary_probe_passed'):
        return {'qualified':False,'boundary_probe_passed':False,'status':trainer['status'],'trainer':trainer}
    command=container_command(image,stage,probe,role='evaluator',evaluator_data=evaluator_data,checkpoint_dir=checkpoint_dir)
    env={k:os.environ[k] for k in ('PATH','SYSTEMROOT') if k in os.environ}
    evaluated=subprocess.run(command,env=env,close_fds=True,capture_output=True,text=True,timeout=30)
    positive=json.loads(evaluated.stdout) if evaluated.returncode==0 else {}
    positive_pass=(positive.get('target_readable') is True and positive.get('checkpoint_readable') is True
        and positive.get('target_sha256')==expected_target_hash and positive.get('checkpoint_sha256')==expected_checkpoint_hash)
    unchanged=sha(probe)==expected_probe_hash
    return {'qualified':False,'boundary_probe_passed':bool(positive_pass and unchanged),
            'status':'PAIRED_ENGINEERING_PROBE' if positive_pass and unchanged else 'FAILED',
            'trainer':trainer,'evaluator_positive':positive,'probe_sha256':expected_probe_hash,
            'image':image,'scientific_authorization':False,
            'remaining_requirement':'Independent immutable staging/daemon ownership and actual evaluator fitting deployment qualification'}
