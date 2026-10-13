"""Pinned evidence, strict receipts and original source compatibility. No fits or predictors."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile

REPO=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
READY='R54_V2_ENGINEERING_READY_TRAINING_NOT_AUTHORIZED'


class GateError(ValueError):
    pass


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    def no_duplicate(pairs):
        result={}
        for key,value in pairs:
            if key in result:
                raise GateError('Duplicate JSON field')
            result[key]=value
        return result
    def invalid_constant(value):
        raise GateError(f'Nonfinite JSON value: {value}')
    return json.loads(Path(path).read_text(),object_pairs_hook=no_duplicate,parse_constant=invalid_constant)


def require(value,message):
    if not value:
        raise GateError(message)


def no_symlink_path(path):
    path=Path(path).absolute()
    require('..' not in path.parts,'Path traversal prohibited')
    for part in [*reversed(path.parents),path]:
        require(not part.is_symlink(),'Symlinked path prohibited')
    return path


def _bytes_at(directory_fd,name):
    require(Path(name).name==name and name not in ('.','..'),'Unsafe evidence name')
    fd=os.open(name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=directory_fd)
    try:
        require(stat.S_ISREG(os.fstat(fd).st_mode),'Nonregular evidence entry')
        with os.fdopen(os.dup(fd),'rb') as stream:
            return stream.read()
    finally:
        os.close(fd)


def pinned_snapshot(directory,manifest,destination):
    path=no_symlink_path(directory)
    require(path.is_dir(),'Missing qualification evidence')
    # File reads use O_NOFOLLOW/openat; only exact reviewed names/bytes can enter the private snapshot.
    fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY)
    try:
        for component in path.parts[1:]:
            next_fd=os.open(component,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd)
            os.close(fd)
            fd=next_fd
    except Exception:
        os.close(fd)
        raise
    try:
        require(set(os.listdir(fd))==set(manifest),'Missing, partial or substituted evidence inventory')
        for name,expected in manifest.items():
            data=_bytes_at(fd,name)
            require(hashlib.sha256(data).hexdigest()==expected,f'Pinned evidence hash mismatch: {name}')
            with (Path(destination)/name).open('xb') as stream:
                stream.write(data)
        return dict(original_path=str(path),sha256=dict(manifest))
    finally:
        os.close(fd)


def validate_receipts(directory):
    directory=Path(directory)
    names={p.name for p in directory.iterdir()}
    require({'START.json','RESULT.json','RECEIPT.json','FINAL_RECEIPT.json','INDEPENDENT_AUDIT.json'}<=names,'Complete receipt and audit required')
    base=load(directory/'RECEIPT.json')['sha256']
    final=load(directory/'FINAL_RECEIPT.json')['sha256']
    require(set(base)==names-{'RECEIPT.json','FINAL_RECEIPT.json','INDEPENDENT_AUDIT.json'},'Base receipt incomplete')
    require(set(final)==names-{'FINAL_RECEIPT.json'},'Final receipt incomplete')
    for hashes in (base,final):
        for name,expected in hashes.items():
            require(Path(name).name==name and sha(directory/name)==expected,'Stale or substituted receipt')
    start,result,saved=map(load,[directory/'START.json',directory/'RESULT.json',directory/'INDEPENDENT_AUDIT.json'])
    require(start['approved_engineering_only'] is True and start['training_authorized'] is False,'Engineering authorization scope mismatch')
    require(result['status']==saved['status']==READY and saved['independent_audit']=='VERIFIED','Exact approved qualification/audit required')
    require(type(result['scratch_optimizer_steps']) is int and result['scratch_optimizer_steps']==saved['scratch_optimizer_steps']==144,'Disposable step count mismatch')
    require(type(result['campaign_optimizer_updates']) is int and result['campaign_optimizer_updates']==saved['campaign_optimizer_updates']==0,'Qualification cannot contain scientific updates')
    require(result['training_authorized'] is False and saved['training_authorized'] is False,'Engineering approval is not scientific authorization')
    require(len(result['results'])==len(saved['assessments'])==6,'Incomplete configuration evidence')
    return dict(qualification_status=READY,receipt_sha256=sha(directory/'RECEIPT.json'),
                final_receipt_sha256=sha(directory/'FINAL_RECEIPT.json'),start_sha256=sha(directory/'START.json'),
                saved_audit_sha256=sha(directory/'INDEPENDENT_AUDIT.json'),engineering_approval={
                    'recorded_approved_engineering_only':True,'recorded_training_authorized':False,
                    'scope':'144 disposable optimizer microbenchmarks only','provenance':'Original START.json local CLI approval assertion; not cryptographic human identity attestation'})


def verify_sources():
    pin=load(HERE/'qualification-pin.json')
    v1=REPO/'experiments/r5_4/frozen/LOCK.json'
    v2=REPO/'engineering/r5_4_v2/LOCK.json'
    require(sha(v1)==pin['v1_lock_sha256'] and sha(v2)==pin['v2_lock_sha256'],'Incompatible or stale v1/v2 locks')
    inventories=[load(v1)['sha256'],load(v2)['sha256']]
    for inventory in inventories:
        for name,expected in inventory.items():
            path=no_symlink_path(REPO/name)
            require(sha(path)==expected,f'Tampered scientific/engineering source: {name}')
    require(sha(REPO/'experiments/r5_4/engineering.json')==pin['original_engineering_sha256'],'Original NO_GO changed')
    integration_lock=load(HERE/'LOCK.json')
    for name,expected in integration_lock['sha256'].items():
        path=no_symlink_path(REPO/name)
        require(sha(path)==expected,f'Integration review freeze mismatch: {name}')
    scripts={str(p.relative_to(REPO)) for p in HERE.glob('*.py')}
    require(scripts=={n for n in integration_lock['sha256'] if n.startswith('integration/r5_4_gate_v2/') and n.endswith('.py')},'Untracked integration implementation')
    paths=set().union(*map(set,inventories),set(integration_lock['sha256']),{'integration/r5_4_gate_v2/LOCK.json'})
    dirty=subprocess.check_output(['git','status','--porcelain','--',*sorted(paths)],cwd=REPO,text=True)
    require(not dirty,'Review inputs must be committed cleanly')
    return dict(v1_lock_sha256=sha(v1),v2_lock_sha256=sha(v2),integration_lock_sha256=sha(HERE/'LOCK.json'),
                compatible_scientific_inventory=load(v1)['sha256'])


def independent_reaudit(snapshot):
    # A fresh storage-only process prevents stale/substituted Python module caches.
    result=subprocess.run([sys.executable,str(REPO/'engineering/r5_4_v2/audit_qualification.py'),str(snapshot)],
        cwd=REPO,check=True,capture_output=True,text=True,timeout=60)
    return json.loads(result.stdout)


def verify_qualification(directory):
    sources=verify_sources()
    pin=load(HERE/'qualification-pin.json')
    work=REPO/'work'
    work.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='r54-qualified-evidence-',dir=work) as temp:
        provenance=pinned_snapshot(directory,pin['artifact_sha256'],temp)
        receipt=validate_receipts(temp)
        require(receipt['receipt_sha256']==pin['receipt_sha256'] and receipt['final_receipt_sha256']==pin['final_receipt_sha256'],'Evidence differs from reviewed qualification')
        audited=independent_reaudit(temp)
        require(audited['status']==READY and audited['independent_audit']=='VERIFIED','Independent native re-audit rejected')
        require(audited==load(temp+'/INDEPENDENT_AUDIT.json'),'Independent native verdict changed')
        # Recheck exact source bytes and observed original evidence after the read-only audit.
        require(verify_sources()==sources,'Sources changed during evidence audit')
        with tempfile.TemporaryDirectory(prefix='r54-evidence-recheck-',dir=work) as second:
            pinned_snapshot(directory,pin['artifact_sha256'],second)
    return dict(sources,**receipt,original_evidence_path=provenance['original_path'],
                qualification_source_snapshot_commit=pin['qualification_source_snapshot_commit'],
                qualification_run_commit='NOT_RECORDED_IN_ORIGINAL_ARTIFACT',
                independent_audit='VERIFIED',scientific_updates=0,disposable_steps=144,
                original_v1_readiness='ENGINEERING_NO_GO',qualification_assessments=audited['assessments'],
                qualified_runtime=pin['qualified_runtime'])
