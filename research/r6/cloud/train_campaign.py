"""Private R6 one-shot training packaging/controller. No local neural execution."""
from pathlib import Path
import argparse,base64,hashlib,io,json,os,re,shutil,subprocess,time,zipfile

def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,obj):Path(path).write_text(json.dumps(obj,sort_keys=True,indent=2)+'\n')
def cli(args):
    exe=os.environ.get('KAGGLE_BIN') or shutil.which('kaggle')
    if not exe:raise RuntimeError('Authenticated Kaggle CLI required')
    result=subprocess.run([exe,*args],text=True,capture_output=True)
    if result.returncode:raise RuntimeError('Provider operation failed; no automatic submission retry')
    return result.stdout

def prepare(repo,dataset,prior_outputs,config,owner,slug,output):
    repo,dataset,prior_outputs,output=map(Path,(repo,dataset,prior_outputs,output));config=read(config) if isinstance(config,(str,Path)) else config
    if config['fit_id'] not in ('r6-p1-raw','r6-p1-scaled') or config['updates']!=600:raise ValueError('Bounded two-fit authorization only')
    if owner!='akilan2006' or not re.fullmatch('[a-z0-9-]{6,50}',slug):raise ValueError('Expected verified account/unique short slug')
    output.mkdir(parents=True,exist_ok=False);payload=output/'payload';payload.mkdir()
    for rel in ('research/__init__.py','research/r6/__init__.py','research/r6/architectures/__init__.py','research/r6/training/__init__.py'):
        p=payload/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('')
    core=('research/r6/common.py','research/r6/architectures/model.py','research/r6/training/worker.py')
    for rel in core:shutil.copy2(repo/rel,payload/rel)
    old_manifest=read(prior_outputs/'training-payload-final/manifest.json')
    for rel,digest in old_manifest['source_files_sha256'].items():
        source=repo/rel
        if sha(source)!=digest:raise ValueError('Frozen legacy source changed')
        dest=payload/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
    for name in ('train.json','dev.json','normalization.json'):
        if sha(dataset/name)!=read(dataset/'manifest.json')['files'][name]['sha256']:raise ValueError('Dataset source changed')
        shutil.copy2(dataset/name,payload/name)
    write(payload/'config.json',config)
    checkpoint=prior_outputs/'verified-fits/17-graph-concept-on/final.json'
    legacy=read(prior_outputs/'NUMERICAL_EQUIVALENCE_REPORT.json')
    fit=next(f for f in legacy['fits'] if f['seed']==17 and f['condition']=='graph-concept-on')
    if sha(checkpoint)!=fit['checkpoint_sha256']:raise ValueError('P0 existing checkpoint mismatch')
    # Existing legacy weights are mounted read-only from their verified private dataset.
    oldtrain=read(prior_outputs/'training-payload-final/train/17.json')
    cases=[]
    for domain in ('math','coding','physics'):
        row=next(r for r in oldtrain if r['problem']['domain']==domain)
        heads=next(d for d in fit['domains'] if d['domain']==domain)['heads']
        cases.append({'problem':row['problem'],'heads':heads})
    write(payload/'legacy-p0-cases.json',cases)
    inventory={str(p.relative_to(payload)):sha(p) for p in sorted(payload.rglob('*')) if p.is_file()}
    if any('protected' in name or name.startswith('research/r6/datasets/') or name.startswith('research/r6/evaluation/') for name in inventory):raise ValueError('Trainer exposes final data or generator/evaluator')
    manifest={'schema':'mindscape-r6-training-package-v1','files_sha256':inventory,'historical_source_commit':'d54ff8fe64f30c53b9662ddba90691338aad05f6','fit_id':config['fit_id'],'authorized_updates':600,'evaluation_records':0,'r6_generator_code_included':False,'dataset_train_sha256':sha(dataset/'train.json'),'dataset_dev_sha256':sha(dataset/'dev.json'),'p0_reference_sha256':sha(prior_outputs/'NUMERICAL_EQUIVALENCE_REPORT.json'),'legacy_checkpoint_sha256':sha(checkpoint)}
    write(payload/'manifest.json',manifest)
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(payload.rglob('*')):
            if p.is_file():z.writestr(str(p.relative_to(payload)),p.read_bytes())
    archive=buffer.getvalue();(output/'payload.zip').write_bytes(archive)
    launcher=training_launcher(base64.b64encode(archive).decode(),hashlib.sha256(archive).hexdigest())
    (output/'kernel.py').write_text(launcher)
    metadata={'id':owner+'/'+slug,'title':slug,'code_file':'kernel.py','language':'python','kernel_type':'script','is_private':True,'enable_gpu':True,'enable_tpu':False,'enable_internet':False,'dataset_sources':['akilan2006/mindscape-r57c-final-checkpoints-20261015-v1/1'],'kernel_sources':[],'competition_sources':[],'model_sources':[]}
    write(output/'kernel-metadata.json',metadata)
    receipt={'kernel_id':metadata['id'],'fit_id':config['fit_id'],'kernel_sha256':sha(output/'kernel.py'),'payload_sha256':sha(output/'payload.zip'),'manifest_sha256':sha(payload/'manifest.json'),'config_sha256':sha(payload/'config.json'),'data_sha256':{'train':sha(payload/'train.json'),'dev':sha(payload/'dev.json')},'updates':600,'files_sha256':inventory}
    write(output/'package-receipt.json',receipt);return receipt

def training_launcher(encoded,digest):
    return f'''# Private bounded R6 CUDA execution; no evaluation data attached.
import base64,hashlib,io,json,os,pathlib,sys,zipfile
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
if os.environ.get('KAGGLE_KERNEL_RUN_TYPE')!='Batch':raise RuntimeError('Kaggle Batch required')
raw=base64.b64decode({encoded!r})
if hashlib.sha256(raw).hexdigest()!={digest!r}:raise RuntimeError('Package changed')
root=pathlib.Path('/kaggle/working/r6-training-package');root.mkdir(exist_ok=False)
with zipfile.ZipFile(io.BytesIO(raw)) as archive:
    for name in archive.namelist():
        if not (root/name).resolve().is_relative_to(root):raise RuntimeError('Unsafe archive')
    archive.extractall(root)
sys.path.insert(0,str(root));sys.path.insert(0,str(root/'src'));os.chdir(root)
manifest=json.loads((root/'manifest.json').read_text())
for name,digest in manifest['files_sha256'].items():
    if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Payload hash changed')
from research.r6.training.worker import require_cloud,run
torch=require_cloud()
# R6-P0: verify one existing historical checkpoint, no optimizer or fitting.
from mindscape.r57c.ordinal_model import make_model
matches=list(pathlib.Path('/kaggle/input').rglob('17-graph-concept-on-final.json'))
if len(matches)!=1:raise RuntimeError('Exactly one pinned legacy checkpoint input required')
old=matches[0];old_sha=hashlib.sha256(old.read_bytes()).hexdigest()
if old_sha!=manifest['legacy_checkpoint_sha256']:raise RuntimeError('P0 mounted checkpoint hash mismatch')
data=json.loads(old.read_text());model=make_model('graph-concept-on',17).to('cuda:0');state=model.state_dict()
if set(data['tensors'])!=set(state):raise RuntimeError('P0 tensor keys')
for name,target in state.items():
    item=data['tensors'][name]
    if item['shape']!=list(target.shape) or item['dtype']!=str(target.dtype):raise RuntimeError('P0 tensor schema')
    state[name]=torch.tensor(item['values'],dtype=target.dtype,device='cuda:0')
model.load_state_dict(state,strict=True);model.eval();checks=[]
with torch.inference_mode():
    for case in json.loads((root/'legacy-p0-cases.json').read_text()):
        prediction=model.predict(case['problem'])
        for head in case['heads']:
            value=prediction[head['head']]
            reference=torch.tensor(head['cuda_output'],device='cuda:0',dtype=value.dtype)
            if value.device.type!='cuda' or not torch.allclose(value,reference,atol=1e-5,rtol=1e-4):raise RuntimeError('P0 equivalence failed')
        checks.append(case['problem']['domain'])
if hashlib.sha256(old.read_bytes()).hexdigest()!=old_sha:raise RuntimeError('P0 checkpoint altered')
p0={{'status':'PASSED_EXISTING_CHECKPOINT_EQUIVALENCE','optimizer_updates':0,'device':'cuda:0','checkpoint_sha256':old_sha,'domains':checks,'reference_sha256':manifest['p0_reference_sha256'],'atol':1e-5,'rtol':1e-4}}
pathlib.Path('/kaggle/working/r6-p0-verification.json').write_text(json.dumps(p0,indent=2))
print(json.dumps(p0),flush=True)
del model,state
torch.cuda.empty_cache()
receipt=run(root,'/kaggle/working/r6-fit-output')
print(json.dumps({{'fit_id':receipt['fit_id'],'status':receipt['status'],'optimizer_updates':receipt['optimizer_updates']}}),flush=True)
'''

def submit(package,ledger,quota_receipt):
    package,ledger,quota_receipt=map(Path,(package,ledger,quota_receipt));r=read(package/'package-receipt.json');m=read(package/'kernel-metadata.json')
    if sha(package/'kernel.py')!=r['kernel_sha256'] or m['id']!=r['kernel_id']:raise ValueError('Submitted package changed')
    quota=read(quota_receipt)
    # Controller snapshots authenticated quota immediately before each call.
    if time.time()-quota_receipt.stat().st_mtime>600:raise ValueError('Fresh quota check required')
    gpu=next(q for q in quota if q['resource']=='GPU')
    if float(gpu['remaining'].removesuffix('h'))<1.:raise RuntimeError('At least one free GPU hour required; no paid fallback')
    if not (m['is_private'] is True and m['enable_internet'] is False and m['enable_gpu'] is True) or m['dataset_sources']!=['akilan2006/mindscape-r57c-final-checkpoints-20261015-v1/1'] or any(m[k] for k in ('kernel_sources','model_sources','competition_sources')):raise ValueError('Private offline train-only runtime required')
    ledger.mkdir(parents=True,exist_ok=True)
    reservations=[read(p) for p in ledger.glob('*.json')]
    if sum(x['authorized_updates'] for x in reservations)+600>1200:raise RuntimeError('Global R6 1200-update authorization exhausted')
    with (ledger/(r['fit_id']+'.json')).open('x') as f:json.dump({'fit_id':r['fit_id'],'kernel_id':r['kernel_id'],'authorized_updates':600,'config_sha256':r['config_sha256'],'source_sha256':r['kernel_sha256'],'status':'ATTEMPTED_ONCE'},f)
    response=cli(['kernels','push','-p',str(package),'--accelerator','NvidiaTeslaT4']);(package/'submission-response.txt').write_text(response)
    if not re.search(r'Kernel version 1 successfully pushed',response):raise RuntimeError('No confirmed version1; reservation retained; no retry')
    return r['kernel_id']

def wait(kernel,record,timeout=5400):
    start=time.monotonic();previous=None
    while True:
        status=cli(['kernels','status',kernel]).strip();Path(record).write_text(status+'\n')
        if status!=previous:print(json.dumps({'event':'cloud_status','kernel':kernel,'status':status}),flush=True);previous=status
        if 'KernelWorkerStatus.COMPLETE' in status:return
        if any(x in status for x in ('KernelWorkerStatus.ERROR','KernelWorkerStatus.CANCEL','KernelWorkerStatus.FAIL')):raise RuntimeError('Cloud failure; preserve outputs and counted budget, do not retrain')
        if time.monotonic()-start>timeout:raise TimeoutError('Inspect existing reserved job without retraining')
        time.sleep(20)

def verify_fit(directory):
    directory=Path(directory);r=read(directory/'receipt.json')
    if r['status']!='COMPLETED' or r['optimizer_updates']!=600 or not r['per_parameter_adam_steps_verified'] or not r['checkpoint_reload_verified']:raise ValueError('Exact600 completed updates/reload/accounting required')
    lines=[json.loads(line) for line in (directory/'loss.jsonl').read_text().splitlines()]
    if [x['update'] for x in lines]!=list(range(1,601)):raise ValueError('Loss/update sequence mismatch')
    for name,digest in r['output_hashes'].items():
        if sha(directory/name)!=digest:raise ValueError('Export hash mismatch: '+name)
    if sha(directory/'final.json')!=r['checkpoint_sha256'] or sha(directory/'best-dev.json')!=r['selected_checkpoint_sha256']:raise ValueError('Final/selected checkpoint mismatch')
    return r

def collect(package,output):
    package,output=map(Path,(package,output));r=read(package/'package-receipt.json');kernel=r['kernel_id'];output.mkdir(parents=True,exist_ok=False)
    for attempt in range(1,4):
        target=output/f'download-{attempt}'
        cli(['kernels','output',kernel+'/1','-p',str(target),'--file-pattern','^r6-fit-output/|^r6-p0-verification.json$'])
        try:fit=verify_fit(target/'r6-fit-output');p0=read(target/'r6-p0-verification.json');assert p0['optimizer_updates']==0 and p0['status']=='PASSED_EXISTING_CHECKPOINT_EQUIVALENCE'
        except (ValueError,KeyError,FileNotFoundError,AssertionError):
            if attempt==3:raise
            time.sleep(15);continue
        shutil.copytree(target/'r6-fit-output',output/'fit');shutil.copy2(target/'r6-p0-verification.json',output/'p0-verification.json');break
    provider=output/'provider';cli(['kernels','pull',kernel,'-p',str(provider),'-m']);meta=read(provider/'kernel-metadata.json')
    if sha(provider/meta['code_file'])!=r['kernel_sha256']:raise ValueError('Provider source differs')
    expected=read(package/'kernel-metadata.json')
    for key,wanted in expected.items():
        if key=='code_file':continue
        if key=='dataset_sources':
            if meta.get(key) not in (wanted,[x.removesuffix('/1') for x in wanted]):raise ValueError('Dataset attachment differs')
        elif meta.get(key)!=wanted:raise ValueError('Provider metadata differs: '+key)
    proof={'kernel_id':kernel,'source_sha256':r['kernel_sha256'],'metadata':meta,'source_verified':True,'payload_files_sha256':r['files_sha256'],'evaluation_data_absent':True,'fit_receipt_sha256':sha(output/'fit/receipt.json'),'optimizer_updates':fit['optimizer_updates'],'provider_metadata_sha256':sha(provider/'kernel-metadata.json')}
    write(output/'trainer-evidence.json',proof);return proof

def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='action',required=True)
    q=s.add_parser('prepare')
    for k in ('repo','dataset','prior-outputs','config','owner','slug','output'):q.add_argument('--'+k,required=True)
    q=s.add_parser('submit')
    for k in ('package','ledger','quota-receipt'):q.add_argument('--'+k,required=True)
    q=s.add_parser('wait');q.add_argument('--kernel',required=True);q.add_argument('--record',required=True)
    q=s.add_parser('collect');q.add_argument('--package',required=True);q.add_argument('--output',required=True)
    q=s.add_parser('verify-fit');q.add_argument('--directory',required=True)
    args=vars(p.parse_args());action=args.pop('action');print(json.dumps(globals()[action.replace('-','_')](**args),default=str))
if __name__=='__main__':main()
