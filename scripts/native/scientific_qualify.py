"""Standalone nontraining workaround: never dispatch repository legacy workflows."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from mindscape.r56.qualification import prepare

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[2]
    result=prepare(root,args.output)
    print(json.dumps({'proposal_sha256':result['package_sha256'],
        'decision':result['scientific_decision'],'optimizer_updates':0,
        'workload_fits':result['neural_run_count'],'proposed_updates':result['total_optimizer_steps']}))
