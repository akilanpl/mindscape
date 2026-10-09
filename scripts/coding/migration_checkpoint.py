"""Archive all research results into Git-sized parts; restore via existing verifier."""
import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

from restore_completion import restore

ROOT=Path('research_checkpoint/2026-10-08')
SNAPSHOT=Path('work/migration_snapshot_v1')
RELEASE='partial-cloud-migration-checkpoint'


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            value.update(block)
    return value.hexdigest()


class Parts:
    def __init__(self):
        self.stream=None
        self.size=0
        self.entries=[]
    def write(self,data):
        original=len(data)
        while data:
            if self.stream is None or self.size==32*1024*1024:
                if self.stream:
                    self.stream.close()
                path=ROOT/f'archive-{len(self.entries):04d}.part'
                self.stream=path.open('wb');self.entries.append(path);self.size=0
            chunk=data[:32*1024*1024-self.size]
            self.stream.write(chunk);self.size+=len(chunk);data=data[len(chunk):]
        return original
    def flush(self):
        if self.stream:
            self.stream.flush()
    def close(self):
        if self.stream:
            self.stream.close()


def freeze():
    if ROOT.exists() or SNAPSHOT.exists():
        raise RuntimeError('Checkpoint already exists; never overwrite verified research archives')
    if RELEASE == 'complete-research-release':
        from completion_gate import verify
        verify()
    ROOT.mkdir(parents=True);SNAPSHOT.mkdir(parents=True)
    sources={'all_results':Path('results'),'datasets':Path('datasets'),
             'wasi_runtime':Path('work/coding/runtime'),'public_benchmark_input':Path('work/coding/humaneval'),
             **{p:Path(p) for p in ('configs','scripts','docs','experiments','tests','demo')}}
    for label,path in sources.items():
        if path.exists():
            ignored=['__pycache__','.DS_Store','*.pyc']
            if label=='wasi_runtime':
                ignored.append('*.cwasm')
            shutil.copytree(path,SNAPSHOT/label,ignore=shutil.ignore_patterns(*ignored))
    if RELEASE == 'complete-research-release':
        recovery = Path('work/stall_recovery_20261009')
        shutil.copytree(recovery, SNAPSHOT/'restart_recovery')
        sources['restart_recovery'] = recovery
        archive = Path('work/Mindscape_recovery_20261009.tar.gz')
        shutil.copy2(archive, SNAPSHOT/'recovery_archive.tar.gz')
    sources={k:v for k,v in sources.items() if v.exists()}
    log_root=SNAPSHOT/'logs';log_root.mkdir()
    for path in Path('work').glob('*.log'):
        shutil.copy2(path,log_root/path.name)
    sources['logs']=Path('work/migration_logs')
    shutil.copytree('src/mindscape',SNAPSHOT/'source/mindscape',ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('README.md','pyproject.toml'):
        shutil.copy2(name,SNAPSHOT/name)
    files={str(p.relative_to(SNAPSHOT)):digest(p) for p in sorted(SNAPSHOT.rglob('*')) if p.is_file()}
    locked=[json.loads(s) for s in Path('results/coding/completion_lockbox_eval_v1/rows.jsonl').read_text().splitlines()]
    curve=[json.loads(s) for s in Path('results/coding/completion_gradient_v1/rows.jsonl').read_text().splitlines()]
    assert len({(r['condition'],r['task_id']) for r in locked})==len(locked)
    assert len({tuple(r['key']) for r in curve})==len(curve)
    manifest={'release':RELEASE,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              'source_paths':{k:str(v) for k,v in sources.items()},'files':files,
              'locked_completed':len(locked),'learning_completed':len(curve),'learning_remaining':1920-len(curve),
              'foundation_weights':'External exact pinned HF downloads; hashes and bootstrap shipped',
              'exclusions':'Virtual environments, generation caches, OS/Python caches, disposable compiled WASM caches outside immutable historical packages, duplicated temporary restore directories'}
    (SNAPSHOT/'MANIFEST.json').write_text(json.dumps(manifest,indent=2))
    shutil.copy2(SNAPSHOT/'MANIFEST.json',ROOT/'SNAPSHOT_MANIFEST.json')
    parts=Parts()
    with gzip.GzipFile(fileobj=parts,mode='wb',compresslevel=6,mtime=0) as zipped, tarfile.open(fileobj=zipped,mode='w|') as archive:
        for path in sorted(SNAPSHOT.rglob('*')):
            if path.is_file():
                archive.add(path,arcname=str(path.relative_to(SNAPSHOT)),recursive=False)
    parts.close()
    archive={'snapshot_manifest_sha256':digest(ROOT/'SNAPSHOT_MANIFEST.json'),
             'parts':[{'path':p.name,'size':p.stat().st_size,'sha256':digest(p)} for p in parts.entries]}
    (ROOT/'ARCHIVE.json').write_text(json.dumps(archive,indent=2))
    print('Frozen',len(files),'files in',len(parts.entries),'parts;',sum(p.stat().st_size for p in parts.entries),'compressed bytes')


def unpack(destination):
    archive=json.loads((ROOT/'ARCHIVE.json').read_text())
    if digest(ROOT/'SNAPSHOT_MANIFEST.json')!=archive['snapshot_manifest_sha256']:
        raise RuntimeError('Snapshot manifest changed')
    scratch=destination.parent/(destination.name+'_snapshot')
    if scratch.exists() or destination.exists():
        raise RuntimeError('Clean restore paths required')
    scratch.mkdir(parents=True)
    joined=scratch.parent/(scratch.name+'.tar.gz')
    with joined.open('wb') as stream:
        for entry in archive['parts']:
            path=ROOT/entry['path']
            if path.stat().st_size!=entry['size'] or digest(path)!=entry['sha256']:
                raise RuntimeError('Archive part corrupted: '+str(path))
            with path.open('rb') as part:
                shutil.copyfileobj(part,stream)
    with tarfile.open(joined,'r:gz') as packed:
        packed.extractall(scratch,filter='data')
    if digest(scratch/'MANIFEST.json')!=archive['snapshot_manifest_sha256']:
        raise RuntimeError('Archive manifest mismatch')
    restore(scratch,destination)
    joined.unlink()
    print('Verified archive parts, complete file inventory, hashes and clean restoration')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['freeze','restore'])
    p.add_argument('--destination',type=Path,default=Path('work/migration_restore_verified'))
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--snapshot', type=Path, default=SNAPSHOT)
    p.add_argument('--release', choices=['partial-cloud-migration-checkpoint','complete-research-release'], default=RELEASE)
    a=p.parse_args()
    ROOT, SNAPSHOT, RELEASE = a.root, a.snapshot, a.release
    freeze() if a.mode=='freeze' else unpack(a.destination)
