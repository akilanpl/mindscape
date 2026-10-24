"""Private, frozen R6 CPU evaluator package and one-shot Kaggle controller.

Building performs no network calls. Execution submits only the prepared CPU
kernel; checkpoint dataset upload is the trusted controller's separate action.
"""
import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import time
import zipfile

SOURCE_FILES = ('research/r6/common.py', 'research/r6/architectures/model.py',
    'research/r6/datasets/mathematics.py', 'research/r6/datasets/physics.py',
    'research/r6/datasets/coding.py', 'research/r6/evaluation/worker.py')
LEGACY_BASELINE_DATASET='akilan2006/mindscape-r57c-final-checkpoints-20261015-v1'
LEGACY_CHECKPOINT_SHA256='a67179976979fd8a2722d3d159d93127fd3f87b01c84c9c8bc977da8926ec6b5'
# Canonical inventories from the already verified, immutable R5.7c outputs.
LEGACY_SOURCE_INVENTORY_SHA256='ff16a2cf799100b65a240e0b91b11f17cb91f6f03b7f3f2bbeebe5c241e1aca6'
LEGACY_BASELINE_FILES_SHA256='bdb2bf5cd6ed3b3d58c95e7cda9745a5f19398574233259914e369c33ae6f1d5'


def _object_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _validate_baseline_attachment(trainer, sources):
    if not sources:
        return
    proof=trainer.get('baseline_checkpoint_input',{})
    files=proof.get('files_sha256',{})
    if (sources not in ([LEGACY_BASELINE_DATASET],[LEGACY_BASELINE_DATASET+'/1']) or
        proof.get('reference')!=LEGACY_BASELINE_DATASET+'/1' or
        proof.get('is_private') is not True or proof.get('download_verified') is not True or
        proof.get('checkpoint_sha256')!=LEGACY_CHECKPOINT_SHA256 or
        len(files)!=18 or _object_sha(files)!=LEGACY_BASELINE_FILES_SHA256):
        raise ValueError('trainer attachment lacks exact verified historical baseline-only proof')


def _validate_trainer_files(trainer):
    files=trainer.get('payload_files_sha256')
    if not isinstance(files,dict) or not files:
        raise ValueError('trainer payload source inventory required')
    legacy={name:digest for name,digest in files.items() if name.startswith('src/')}
    if legacy:
        frozen=trainer.get('legacy_frozen_source_sha256',{})
        if (len(frozen)!=272 or _object_sha(frozen)!=LEGACY_SOURCE_INVENTORY_SHA256 or legacy!=frozen):
            raise ValueError('legacy source must match complete independently frozen 272-file inventory')
    for name,digest in files.items():
        _hash(digest)
        if name.startswith('src/'):
            continue  # Exact frozen legacy source may contain evaluator modules.
        lower=name.lower()
        if (any(word in lower for word in ('protected','final_iid','final_ood','evaluation/')) or
            lower.startswith('research/r6/datasets/')):
            raise ValueError('trainer payload includes protected evaluation input/code or R6 generators')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(value):
    return json.loads(Path(value).read_text()) if isinstance(value,(str,Path)) else value


def _write(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')


def _hash(value):
    if not isinstance(value,str) or not re.fullmatch('[0-9a-f]{64}',value):
        raise ValueError('SHA256 required')
    return value


def _safe_id(value):
    if not isinstance(value,str) or not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]*',value):
        raise ValueError('unsafe identifier')
    return value


def prepare(repo,dataset_root,checkpoint_index,dataset_receipt,trainer_evidence,owner,slug,output):
    repo,dataset_root,output=map(Path,(repo,dataset_root,output))
    _safe_id(owner);_safe_id(slug)
    if output.exists():
        raise FileExistsError('new evaluator package directory required')
    checkpoints=_json(checkpoint_index)
    checkpoints=checkpoints['checkpoints'] if isinstance(checkpoints,dict) else checkpoints
    dataset=_json(dataset_receipt);trainers=_json(trainer_evidence)
    if len(checkpoints)!=4:
        raise ValueError('exactly initial/best checkpoints for two fits required')
    fits=sorted({x['fit_id'] for x in checkpoints})
    if len(fits)!=2 or {(x['fit_id'],x['role']) for x in checkpoints}!={(f,r) for f in fits for r in ('initial','best')}:
        raise ValueError('two fit initial/best checkpoint grid required')
    expected_files={}
    for checkpoint in checkpoints:
        _safe_id(checkpoint['fit_id'])
        filename=checkpoint['fit_id']+'-'+checkpoint['role']+'.json'
        expected_files[filename]=_hash(checkpoint['sha256'])
        if _sha(checkpoint['path'])!=checkpoint['sha256'] or not isinstance(checkpoint['config'],dict):
            raise ValueError('local checkpoint content/config mismatch')
    if (dataset.get('version')!=1 or dataset.get('is_private') is not True or
        dataset.get('download_verified') is not True or dataset.get('files_sha256')!=expected_files):
        raise ValueError('private version-1 downloaded checkpoint dataset with exact four files required')
    _hash(dataset['provider_metadata_sha256'])
    if not re.fullmatch(re.escape(owner)+r'/[A-Za-z0-9_-]+',dataset['dataset_id']):
        raise ValueError('checkpoint dataset owner/id mismatch')
    if trainers.get('selected_configs_frozen') is not True or trainers.get('budget_actual_updates')!=1200:
        raise ValueError('both bounded fits and correction selection must be frozen')
    evidence=trainers.get('trainers',[])
    ids={x['kernel_id'] for x in evidence}
    if len(evidence)!=2 or len(ids)!=2 or ids!={x['trainer_kernel_id'] for x in checkpoints}:
        raise ValueError('trainer evidence/checkpoint identities mismatch')
    for trainer in evidence:
        metadata=trainer['metadata']
        if (trainer.get('source_verified') is not True or trainer.get('evaluation_data_absent') is not True or
            metadata.get('id')!=trainer['kernel_id'] or metadata.get('is_private') is not True or
            metadata.get('enable_internet') is not False or metadata.get('enable_gpu') is not True or
            metadata.get('kernel_sources')):
            raise ValueError('actual private offline GPU trainer proof required')
        _hash(trainer['source_sha256']);_hash(trainer['fit_receipt_sha256'])
        _validate_baseline_attachment(trainer,metadata.get('dataset_sources',[]))
        _validate_trainer_files(trainer)
    payload={}
    def add(name,content):
        if name.startswith('/') or '..' in Path(name).parts:
            raise ValueError('unsafe payload path')
        payload[name]=content
    for name in SOURCE_FILES:
        add(name,(repo/name).read_bytes())
    for directory in ('research','research/r6','research/r6/architectures','research/r6/datasets','research/r6/evaluation'):
        add(directory+'/__init__.py',b'')
    source_manifest={name:hashlib.sha256(value).hexdigest() for name,value in payload.items()}
    data_entries=[]
    for split in ('final_iid','final_ood'):
        name='protected/'+split+'.json'
        content=(dataset_root/name).read_bytes()
        rows=json.loads(content)
        if not rows or any(r['split']!=split for r in rows):
            raise ValueError('protected dataset split mismatch')
        add(name,content)
        data_entries.append({'path':name,'split':split,'sha256':hashlib.sha256(content).hexdigest()})
    normalization=(dataset_root/'normalization.json').read_bytes()
    if json.loads(normalization).get('computed_from')!='train-only':
        raise ValueError('normalization must be derived from training only')
    add('normalization.json',normalization)
    config=json.dumps(checkpoints[0]['config'],sort_keys=True,indent=2).encode()+b'\n'
    add('config.json',config)
    audit={'provider':'kaggle','device':'cpu','all_fits_locked':True,
        'final_test_selection_frozen':True,'controller_trusted':True,
        'trainer_kernel_ids':sorted(ids),'evaluator_kernel_id':owner+'/'+slug,
        'trainer_evidence':trainers,'checkpoint_dataset':dataset,
        'same_owner_access_limitation':'Trusted owner can edit future runs; no adversarial owner isolation claimed',
        'dataset_version_limitation':'Mount may omit version in metadata; receipt pins downloaded version 1 and exact checkpoint contents'}
    audit_bytes=json.dumps(audit,sort_keys=True,indent=2).encode()+b'\n'
    add('runtime-audit.json',audit_bytes)
    manifest={'schema':'r6-evaluation-v1','config_path':'config.json',
        'config_sha256':hashlib.sha256(config).hexdigest(),
        'normalization_path':'normalization.json','normalization_sha256':hashlib.sha256(normalization).hexdigest(),
        'runtime_audit_path':'runtime-audit.json','runtime_audit_sha256':hashlib.sha256(audit_bytes).hexdigest(),
        'source_manifest':source_manifest,'datasets':data_entries,
        'checkpoints':[{k:x[k] for k in ('fit_id','role','sha256','config','trainer_kernel_id')} |
            {'path':x['fit_id']+'-'+x['role']+'.json'} for x in checkpoints]}
    add('evaluation-manifest.json',json.dumps(manifest,sort_keys=True,indent=2).encode()+b'\n')
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as archive:
        for name,content in sorted(payload.items()):
            info=zipfile.ZipInfo(name,date_time=(2026,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(info,content)
    archive_bytes=stream.getvalue();archive_sha=hashlib.sha256(archive_bytes).hexdigest()
    launcher='''# Private CPU inference only: generated frozen R6 evaluator.
import base64,hashlib,io,json,os,pathlib,sys,zipfile
if sys.platform!='linux' or os.environ.get('KAGGLE_KERNEL_RUN_TYPE')!='Batch':
 raise RuntimeError('Kaggle Batch required')
os.environ['CUDA_VISIBLE_DEVICES']=''
payload=base64.b64decode(PAYLOAD)
if hashlib.sha256(payload).hexdigest()!=PAYLOAD_SHA: raise RuntimeError('payload hash')
root=pathlib.Path('/kaggle/working/r6-evaluation-package')
if root.exists(): raise FileExistsError('immutable evaluator extraction')
root.mkdir()
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
 for info in archive.infolist():
  name=pathlib.PurePosixPath(info.filename)
  if name.is_absolute() or '..' in name.parts: raise RuntimeError('archive path')
  path=root.joinpath(*name.parts)
  path.parent.mkdir(parents=True,exist_ok=True)
  path.write_bytes(archive.read(info))
manifest_path=root/'evaluation-manifest.json'
manifest=json.loads(manifest_path.read_text())
for checkpoint in manifest['checkpoints']:
 matches=list(pathlib.Path('/kaggle/input').rglob(checkpoint['path']))
 if len(matches)!=1 or not matches[0].is_file(): raise RuntimeError('unique mounted checkpoint required')
 if hashlib.sha256(matches[0].read_bytes()).hexdigest()!=checkpoint['sha256']: raise RuntimeError('mounted checkpoint hash')
 checkpoint['path']=str(matches[0].resolve())
manifest_path.write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\\n')
sys.path.insert(0,str(root))
from research.r6.evaluation.worker import run
receipt=run(root,'/kaggle/working/r6-evaluation-output')
print(json.dumps({'status':receipt['status'],'optimizer_updates_executed':0,'evaluations':len(receipt['evaluations'])}))
'''
    launcher=launcher.replace('PAYLOAD_SHA',repr(archive_sha)).replace('PAYLOAD',repr(base64.b64encode(archive_bytes).decode()))
    metadata={'id':owner+'/'+slug,'title':slug,'code_file':'kernel.py','language':'python',
        'kernel_type':'script','is_private':True,'enable_gpu':False,'enable_tpu':False,
        'enable_internet':False,'dataset_sources':[dataset['dataset_id']+'/1'],'competition_sources':[],
        'kernel_sources':[],'model_sources':[]}
    output.mkdir(parents=True)
    (output/'kernel.py').write_text(launcher)
    _write(output/'kernel-metadata.json',metadata)
    receipt={'schema':'r6-evaluator-package-v1','kernel_id':metadata['id'],
        'source_sha256':_sha(output/'kernel.py'),'metadata_sha256':_sha(output/'kernel-metadata.json'),
        'payload_sha256':archive_sha,'payload_files_sha256':{n:hashlib.sha256(v).hexdigest() for n,v in payload.items()},
        'checkpoint_files_sha256':expected_files,'checkpoint_dataset':dataset,
        'optimizer_updates_authorized':0,'evaluation_manifest':manifest}
    _write(output/'package-receipt.json',receipt)
    return receipt


def _cli(arguments):
    result=subprocess.run([os.environ.get('KAGGLE_BIN','kaggle'),*arguments],text=True,
        capture_output=True,timeout=300)
    if result.returncode:
        raise RuntimeError('Kaggle command failed: '+' '.join(arguments)+'\n'+result.stdout+'\n'+result.stderr)
    return result.stdout


def _metadata_matches(actual,expected):
    for key in ('id','is_private','enable_gpu','enable_tpu','enable_internet','kernel_sources','competition_sources','model_sources'):
        if actual.get(key,[] if key.endswith('_sources') else None)!=expected.get(key):
            raise ValueError('provider evaluator metadata mismatch: '+key)
    normalize=lambda value:value[:-2] if value.endswith('/1') else value
    actual_sources=actual.get('dataset_sources',[])
    expected_sources=expected['dataset_sources']
    if len(actual_sources)!=len(expected_sources) or {normalize(v) for v in actual_sources}!={normalize(v) for v in expected_sources}:
        raise ValueError('provider checkpoint dataset attachment mismatch')


def verify_download(output,package):
    output,package=Path(output),Path(package)
    prepared=_json(package/'package-receipt.json')
    result_root=output/'r6-evaluation-output'
    receipt=_json(result_root/'evaluation_receipt.json')
    if receipt.get('status')!='COMPLETED' or receipt.get('device')!='cpu' or receipt.get('optimizer_updates_executed')!=0:
        raise ValueError('actual completed CPU zero-update receipt required')
    expected={(x['fit_id'],x['role']):x['sha256'] for x in prepared['evaluation_manifest']['checkpoints']}
    actual={(x['fit_id'],x['role']):x['checkpoint_sha256'] for x in receipt['evaluations']}
    if actual!=expected or len(receipt['evaluations'])!=4:
        raise ValueError('evaluator checkpoint identities mismatch')
    for filename,digest in receipt['artifacts'].items():
        path=(result_root/filename).resolve()
        if result_root.resolve() not in path.parents or _sha(path)!=digest:
            raise ValueError('downloaded evaluator artifact hash mismatch')
    manifest=_json(output/'r6-evaluation-package/evaluation-manifest.json')
    audit_path=output/'r6-evaluation-package/runtime-audit.json'
    if _sha(output/'r6-evaluation-package/evaluation-manifest.json')!=receipt['evaluation_manifest_sha256']:
        raise ValueError('runtime evaluation manifest hash mismatch')
    if _sha(audit_path)!=manifest['runtime_audit_sha256']:
        raise ValueError('runtime audit hash mismatch')
    planned=prepared['evaluation_manifest']
    for key in ('source_manifest','datasets','config_sha256','normalization_sha256','runtime_audit_sha256'):
        if manifest.get(key)!=planned.get(key):
            raise ValueError('actual runtime package differs from frozen package: '+key)
    runtime=receipt['runtime_audit']
    if (runtime.get('source_dataset_checkpoint_hashes_rechecked_after_evaluation') is not True or
        runtime.get('known_credential_environment_absent') is not True or
        runtime.get('known_credential_files_absent') is not True or
        runtime.get('model_requires_grad') is not False or runtime.get('cuda_available') is not False or
        len(runtime.get('checkpoint_write_open_denied_paths',[]))!=4):
        raise ValueError('actual runtime isolation proof incomplete')
    return receipt


def execute(args):
    package,output=Path(args.package),Path(args.output)
    timeout=getattr(args,'timeout_seconds',7200);poll=getattr(args,'poll_seconds',20)
    if not 1<=poll<=60 or not 60<=timeout<=14400:
        raise ValueError('invalid bounded status polling configuration')
    receipt=_json(package/'package-receipt.json')
    if _sha(package/'kernel.py')!=receipt['source_sha256'] or _sha(package/'kernel-metadata.json')!=receipt['metadata_sha256']:
        raise ValueError('prepared evaluator source changed')
    if output.exists():
        raise FileExistsError('fresh evaluator controller output required')
    output.mkdir(parents=True)
    reservation=package/'SUBMISSION_RESERVED.json'
    descriptor=os.open(reservation,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(descriptor,'w') as stream:
        json.dump({'kernel_id':receipt['kernel_id'],'optimizer_updates':0},stream)
    pushed=_cli(['kernels','push','-p',str(package)])
    (output/'push-response.txt').write_text(pushed)
    if not re.search(r'version\s+1\b',pushed,re.I):
        raise ValueError('one-shot evaluator must be newly created version 1')
    started=time.monotonic()
    statuses=[]
    while True:
        status=_cli(['kernels','status',receipt['kernel_id']])
        statuses.append({'elapsed_seconds':time.monotonic()-started,'status':status})
        _write(output/'status-history.json',statuses)
        if re.search(r'\bcomplete\b',status,re.I):
            break
        if re.search(r'\b(error|failed|cancelled)\b',status,re.I):
            raise RuntimeError('independent evaluator failed; preserve logs, no automatic execution retry')
        if time.monotonic()-started>timeout:
            raise TimeoutError('independent evaluator timeout; do not resubmit running kernel')
        time.sleep(poll)
    provider=output/'provider-source';provider.mkdir()
    _cli(['kernels','pull',receipt['kernel_id'],'-p',str(provider),'-m'])
    metadata=_json(provider/'kernel-metadata.json')
    expected_metadata=_json(package/'kernel-metadata.json')
    _metadata_matches(metadata,expected_metadata)
    files=list(provider.glob('*.py'))
    if len(files)!=1 or _sha(files[0])!=receipt['source_sha256']:
        raise ValueError('authenticated provider evaluator source differs')
    download=output/'downloads'
    _cli(['kernels','output',receipt['kernel_id']+'/1','-p',str(download),
          '--file-pattern',r'^r6-evaluation-output/|^r6-evaluation-package/(evaluation-manifest|runtime-audit)\.json$'])
    actual=verify_download(download,package)
    proof={'status':'VERIFIED','kernel_id':receipt['kernel_id'],'version':1,
        'provider_source_sha256':_sha(files[0]),'provider_metadata_sha256':_sha(provider/'kernel-metadata.json'),
        'control_plane_source_verified':True,'optimizer_updates_executed':0,
        'receipt_sha256':_sha(download/'r6-evaluation-output/evaluation_receipt.json'),
        'source_identity_scope':'full-launcher-and-embedded-payload','metadata':metadata}
    _write(output/'controller-verification.json',proof)
    return proof


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    build=sub.add_parser('prepare')
    for name in ('repo','dataset-root','checkpoint-index','dataset-receipt','trainer-evidence','owner','slug','output'):
        build.add_argument('--'+name,required=True)
    run=sub.add_parser('execute')
    run.add_argument('--package',required=True);run.add_argument('--output',required=True)
    run.add_argument('--timeout-seconds',type=int,default=7200)
    run.add_argument('--poll-seconds',type=int,default=20)
    args=parser.parse_args()
    result=prepare(**{k:v for k,v in vars(args).items() if k!='command'}) if args.command=='prepare' else execute(args)
    print(json.dumps({'status':result.get('status','PREPARED'),'kernel_id':result['kernel_id']}))


if __name__=='__main__':
    main()
