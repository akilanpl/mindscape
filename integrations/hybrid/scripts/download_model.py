"""Download official public Apache-2.0 model into the workspace; no remote code."""
import argparse
import hashlib
import json
from pathlib import Path
import time

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--destination',default='work/models/Qwen3-0.6B')
    parser.add_argument('--receipt',default='outputs/delivery/neural/model-receipt.json')
    parser.add_argument('--revision',help='Exact commit; defaults to checked-in model-lock.json')
    args=parser.parse_args()
    from huggingface_hub import HfApi, snapshot_download
    started=time.perf_counter()
    lock_path=Path(__file__).resolve().parents[1]/'model-lock.json'
    revision=args.revision or (json.loads(lock_path.read_text())['revision'] if lock_path.exists() else None)
    if not revision or len(revision)!=40 or any(c not in '0123456789abcdef' for c in revision):
        raise SystemExit('Supply an exact 40-character --revision or a model-lock.json')
    info=HfApi().model_info('Qwen/Qwen3-0.6B',revision=revision)
    target=Path(args.destination).resolve()
    snapshot_download('Qwen/Qwen3-0.6B',revision=info.sha,local_dir=target,allow_patterns=['*.json','*.safetensors','*.txt','*.jinja','*.model','LICENSE','README.md'],max_workers=4)
    files=[]
    for path in sorted(target.rglob('*')):
        if path.is_file() and '.cache' not in path.parts and path.name != 'model-receipt.json':
            digest=hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda:stream.read(8*1024*1024),b''):
                    digest.update(block)
            files.append({'path':str(path.relative_to(target)),'bytes':path.stat().st_size,'sha256':digest.hexdigest()})
    receipt={'model_id':'Qwen/Qwen3-0.6B','revision':info.sha,'license':'Apache-2.0','official_card':'https://huggingface.co/Qwen/Qwen3-0.6B','license_source':'https://huggingface.co/Qwen/Qwen3-0.6B/blob/'+info.sha+'/LICENSE','destination':str(target),'download_seconds':time.perf_counter()-started,'files':files,'training_executed':False}
    Path(args.receipt).parent.mkdir(parents=True,exist_ok=True)
    Path(args.receipt).write_text(json.dumps(receipt,indent=2))
    (target/'model-receipt.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps({k:v for k,v in receipt.items() if k!='files'}),flush=True)
if __name__=='__main__':
    main()
