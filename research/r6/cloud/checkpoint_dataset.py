"""Prepare and verify a private, immutable R6 checkpoint transport; never fit."""
from pathlib import Path
import argparse,csv,hashlib,io,json,os,shutil,subprocess
from concurrent.futures import ThreadPoolExecutor

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,obj):Path(path).write_text(json.dumps(obj,sort_keys=True,indent=2)+'\n')
def cli(args):
    exe=os.environ.get('KAGGLE_BIN') or shutil.which('kaggle')
    if not exe:raise RuntimeError('Authenticated Kaggle CLI required')
    result=subprocess.run([exe,*args],text=True,capture_output=True)
    if result.returncode:raise RuntimeError('Provider read failed; no cloud mutation performed')
    return result.stdout

def prepare(checkpoint_index,dataset_id,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);entries=read(checkpoint_index)
    if len(entries)!=4 or len({(x['fit_id'],x['role']) for x in entries})!=4:raise ValueError('Four frozen initial/best checkpoints required')
    files={}
    for entry in entries:
        if entry['role'] not in ('initial','best') or sha(entry['path'])!=entry['sha256']:raise ValueError('Frozen checkpoint mismatch')
        data=read(entry['path'])
        if data['format']!='mindscape-r6-safe-json-v1' or data['metadata']['config']!=entry['config']:raise ValueError('Checkpoint config identity mismatch')
        name=entry['fit_id']+'-'+entry['role']+'.json'
        if Path(name).name!=name:raise ValueError('Unsafe filename')
        shutil.copy2(entry['path'],output/name);files[name]=entry['sha256']
    write(output/'dataset-metadata.json',{'title':'Mindscape R6 frozen pilot checkpoints v1','id':dataset_id,'licenses':[{'name':'other'}],'description':'Private exact copies of two bounded native R6 pilot initial and development-selected checkpoints. No final evaluation examples, no optimizer state.'})
    return {'dataset_id':dataset_id,'files_sha256':files,'public':False}

def verify(checkpoint_index,dataset_id,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    expected={x['fit_id']+'-'+x['role']+'.json':x['sha256'] for x in read(checkpoint_index)}
    cli(['datasets','metadata',dataset_id,'-p',str(output/'provider')]);mp=output/'provider/dataset-metadata.json';m=read(mp);m=m.get('info',m)
    if m.get('isPrivate') is not True or m['ownerUser']+'/'+m['datasetSlug']!=dataset_id:raise ValueError('Exact private dataset required')
    raw=cli(['datasets','files',dataset_id+'/1','--csv']);(output/'provider-files-v1.csv').write_text(raw)
    files=list(csv.DictReader(io.StringIO(raw)))
    if len(files)!=4 or {r['name'] for r in files}!=set(expected):raise ValueError('Only four pinned checkpoints may be present')
    target=output/'files';target.mkdir()
    def get(name):
        cli(['datasets','download',dataset_id+'/1','-f',name,'-p',str(target)])
        if sha(target/name)!=expected[name]:raise ValueError('Downloaded checkpoint hash mismatch')
        read(target/name)
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(get,sorted(expected)))
    receipt={'dataset_id':dataset_id,'version':1,'is_private':True,'download_verified':True,'provider_metadata_sha256':sha(mp),'files_sha256':expected}
    write(output/'dataset-receipt.json',receipt);return receipt

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','verify'])
    for name in ('checkpoint-index','dataset-id','output'):p.add_argument('--'+name,required=True)
    args=vars(p.parse_args());action=args.pop('action');print(json.dumps(globals()[action](**args)))
if __name__=='__main__':main()
