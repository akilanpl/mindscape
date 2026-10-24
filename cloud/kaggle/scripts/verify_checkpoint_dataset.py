"""Read-only provider retrieval and verification of one private checkpoint dataset."""
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, io, json, os
from pathlib import Path
import shutil, subprocess
from concurrent.futures import ThreadPoolExecutor

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def verify(checkpoint_index, dataset, output):
    output=Path(output).resolve(); output.mkdir(parents=True,exist_ok=False)
    spec=importlib.util.spec_from_file_location('checkpoint_builder',Path(__file__).with_name('prepare_evaluation.py'))
    builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
    raw=json.loads(Path(checkpoint_index).read_text()); checked=builder.validate_checkpoint_index(raw)
    index=raw['checkpoints'] if isinstance(raw,dict) else raw
    executable=os.environ.get('KAGGLE_BIN') or shutil.which('kaggle')
    if not executable: raise RuntimeError('Authenticated official Kaggle CLI required')
    def cli(args):
        result=subprocess.run([executable,*args],capture_output=True,text=True)
        if result.returncode: raise RuntimeError('Provider operation failed; no cloud mutation performed')
        return result.stdout
    cli(['datasets','metadata',dataset,'-p',str(output/'provider')])
    metadata_path=output/'provider/dataset-metadata.json';meta=json.loads(metadata_path.read_text())
    info=meta.get('info',meta)
    if info.get('isPrivate') is not True or info.get('ownerUser')+'/'+info.get('datasetSlug') != dataset:
        raise PermissionError('Exact private checkpoint dataset required')
    inventory=cli(['datasets','files',dataset+'/1','--csv'])
    (output/'provider-files-v1.csv').write_text(inventory)
    expected={}
    for entry in checked:
        stem=f"{entry['seed']}-{entry['condition']}"
        expected[stem+'-final.json']=entry['sha256'];expected[stem+'-result.json']=entry['receipt_sha256']
    listed=list(csv.DictReader(io.StringIO(inventory)))
    if len(listed)!=18 or {r['name'] for r in listed}!=set(expected):
        raise ValueError('Provider dataset must contain exactly 18 finalized checkpoint/receipt files')
    files=output/'files';files.mkdir()
    def get_file(name):
        cli(['datasets','download',dataset+'/1','-f',name,'-p',str(files)])
        if sha(files/name)!=expected[name]:raise ValueError('Downloaded dataset bytes differ: '+name)
        json.loads((files/name).read_text(),parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
    with ThreadPoolExecutor(max_workers=4) as pool: list(pool.map(get_file,sorted(expected)))
    receipt={'dataset_id':dataset,'version':1,'is_private':True,'provider_metadata_sha256':sha(metadata_path),
             'download_verified':True,'files_sha256':expected,'checkpoints':index,
             'retrieval':'Authenticated per-file version-1 downloads; all18 exact hashes; complete archive endpoint not required'}
    path=output/'dataset-receipt.json';path.write_text(json.dumps(receipt,indent=2)+'\n')
    builder.validate_checkpoint_dataset(path,checked)
    return {'dataset_id':dataset,'verified_files':18,'receipt':str(path),'receipt_sha256':sha(path)}
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint-index','dataset','output'):parser.add_argument('--'+name,required=True)
    print(json.dumps(verify(**vars(parser.parse_args()))))
if __name__=='__main__':main()
