"""Verify exported CUDA fits; preserve absent evaluation metrics as null."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from cloud.kaggle.reproducibility import verify_export

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    rows=[]
    for path in sorted(a.runs.rglob('result.json')):
        r=verify_export(path.parent)
        rows.append({k:r[k] for k in ('seed','condition','checkpoint_sha256',
            'optimizer_updates','seconds','initial_training_loss','final_training_loss')})
    ids=[(r['seed'],r['condition']) for r in rows]
    if len(ids)!=len(set(ids)):
        raise ValueError('Duplicate fit identity; review budget accounting')
    a.output.mkdir(parents=True,exist_ok=False)
    report=dict(verified_fits=rows,verified_optimizer_updates=sum(r['optimizer_updates'] for r in rows),
        expected_fits=9,expected_optimizer_updates=18000,cloud_training_completed=len(rows)==9,
        cloud_evaluation_completed=False,evaluation_metrics=None)
    (a.output/'summary.json').write_text(json.dumps(report,indent=2))
    lines=['# Cloud comparison','',f'Verified cloud fits: {len(rows)}/9.',
        'Independent evaluation metrics are unavailable. Training loss does not establish generalization.','']
    for r in rows:
        lines.append(f"- {r['seed']} / {r['condition']}: {r['optimizer_updates']} updates; {r['seconds']:.2f}s.")
    (a.output/'CLOUD_COMPARISON.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
