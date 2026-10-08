"""Validate and digest non-primary final reference folders without overwriting them."""
import argparse,json,hashlib
from pathlib import Path
from mindscape.data.generation import load_dataset
from mindscape.data.backends import get_backend
from mindscape.evaluation.metrics import aggregate
from mindscape.evaluation.schemas import PredictionRecord

def main():
 p=argparse.ArgumentParser();p.add_argument('results');p.add_argument('--dataset',required=True);a=p.parse_args();root=Path(a.results);out=root/'freeze_manifest.json'
 if out.exists():raise ValueError('Already frozen')
 s,m=load_dataset(a.dataset);backend=get_backend(m['config']['environment']);reference=(root/'references.json').exists();runs=json.loads((root/'references.json').read_text()) if reference else [json.loads(x) for x in (root/'runs.jsonl').read_text().splitlines()];checks=[]
 for r in runs:
  folder=Path(r['directory']);rows=[json.loads(x) for x in (folder/'predictions.jsonl').read_text().splitlines()];byid={e.example_id:e for e in s[r['split']]}
  assert len(rows)==len(byid)
  assert {x['example_id'] for x in rows}==set(byid)
  for x in rows:
   e=byid[x['example_id']];correct=type(x['predicted_answer']) is type(e.target_answer) and x['predicted_answer']==e.target_answer
   assert x['correct']==correct and x['target_answer']==e.target_answer
   valid,goal,error=backend.assess(e,x['predicted_answer'],x['predicted_trajectory'])
   assert x['trajectory_valid']==valid and x['goal_reached']==(goal and correct)
  measured=aggregate([PredictionRecord(**x) for x in rows]);assert measured['accuracy']==r['metrics']['accuracy']
  assert all((folder/f).exists() for f in ['config.yaml','metrics.json','predictions.jsonl','summary.md','plots/outcomes.svg'])
  checks.append(dict(directory=str(folder),examples=len(rows),passed=True))
 files=[]
 for f in sorted(root.rglob('*')):
  if f.is_file():
   h=hashlib.sha256()
   with f.open('rb') as stream:
    for b in iter(lambda:stream.read(1048576),b''):h.update(b)
   files.append(dict(path=str(f.relative_to(root)),bytes=f.stat().st_size,sha256=h.hexdigest()))
 out.write_text(json.dumps(dict(checks=checks,files=files,all_passed=True,kind='non-neural references' if reference else 'historical MLP fresh-data references'),indent=2));print(len(runs),'reference evaluations validated/frozen')
if __name__=='__main__':main()
