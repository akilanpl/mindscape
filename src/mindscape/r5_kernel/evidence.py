"""Immutable artifact registry. No checkpoint or score selection at runtime."""
from pathlib import Path
import hashlib,json
BASE=Path(__file__).resolve().parents[3]
REGISTRY=BASE/'configs/r5_kernel/assets.json'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block:=stream.read(1024*1024):h.update(block)
    return h.hexdigest()
def registry():return json.loads(REGISTRY.read_text())
def asset(name):
    row=registry()['assets'][name];path=Path(row['path']);path=path if path.is_absolute() else BASE/path
    if not path.is_file() or sha(path)!=row['sha256']:raise ValueError('Asset missing or changed: '+name)
    return path,row

def history_check():
    data=registry()
    for name,h in data['history'].items():
        if sha(BASE/name)!=h:raise ValueError('Historical evidence changed: '+name)
    return {'status':'PASS','files':len(data['history']),'scientific_status':data['scientific_status']}
