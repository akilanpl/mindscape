"""Bounded sequential submission of the frozen nine-fit campaign; no local fitting."""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO))
from cloud.kaggle.reproducibility import SEEDS,CONDITIONS,verify_export
from cloud.kaggle.scripts.workflow import prepare,push,verify_account,invoke


def export_fit(kernel, download_root, verified):
    # Only transport retries; never resubmit/retrain after any error.
    for attempt in range(1,4):
        target=download_root/f'attempt-{attempt}'
        target.mkdir(parents=True,exist_ok=False)
        subprocess.run([os.environ['KAGGLE_BIN'],'kernels','output',kernel+'/1',
            '-p',str(target),'--file-pattern','^fit-output/'],check=True)
        try:
            result=verify_export(target/'fit-output')
        except (ValueError,KeyError,FileNotFoundError) as error:
            print(json.dumps(dict(event='download_verification_failed',kernel=kernel,
                attempt=attempt,error_type=type(error).__name__)),flush=True)
            if attempt==3:raise
            time.sleep(30)
            continue
        shutil.copytree(target/'fit-output',verified)
        return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--owner',required=True)
    p.add_argument('--payload',type=Path,required=True)
    p.add_argument('--probe-receipt',type=Path,required=True)
    p.add_argument('--outputs',type=Path,required=True)
    p.add_argument('--ledger',type=Path,required=True)
    a=p.parse_args();out=a.outputs.resolve()
    verified_root=out/'verified-fits';verified_root.mkdir(exist_ok=True)
    campaign=out/'campaign-sequential-v1';campaign.mkdir(exist_ok=False)
    results=[]
    for seed in SEEDS:
        for condition in CONDITIONS:
            verified=verified_root/f'{seed}-{condition}'
            if verified.exists():
                r=verify_export(verified)
                if r['seed']!=seed or r['condition']!=condition:raise ValueError('Existing fit identity mismatch')
                results.append(r)
                print(json.dumps(dict(event='verified_existing_fit',seed=seed,condition=condition)),flush=True)
                continue
            tag={'graph-concept-on':'concept-on','graph-concept-off':'concept-off','matched-gru':'gru'}[condition]
            slug=f'mindscape-r57c-s{seed}-{tag}-20261015-v1';kernel=f'{a.owner}/{slug}'
            run=campaign/f'{seed}-{condition}';run.mkdir()
            account=run/'account.json';verify_account(account)
            package=prepare('training',a.owner,slug,run/'submission',a.payload,seed,condition)
            print(json.dumps(dict(event='submitting',kernel=kernel,seed=seed,condition=condition)),flush=True)
            push(package,a.probe_receipt,account,'NvidiaTeslaT4',a.ledger)
            start=time.monotonic();last=None
            while True:
                state=invoke(['kernels','status',kernel]).strip()
                if state!=last:
                    print(json.dumps(dict(event='cloud_status',kernel=kernel,state=state)),flush=True);last=state
                if 'KernelWorkerStatus.COMPLETE' in state:break
                if any(x in state for x in ('KernelWorkerStatus.ERROR','KernelWorkerStatus.CANCEL','KernelWorkerStatus.FAIL')):
                    (run/'failed-status.txt').write_text(state)
                    # Preserve failed-run outputs without a new optimizer attempt.
                    subprocess.run([os.environ['KAGGLE_BIN'],'kernels','output',kernel+'/1',
                        '-p',str(run/'failed-export'),'--file-pattern','^fit-output/'],check=False)
                    raise RuntimeError('Cloud fit failed; no automatic retraining/resume')
                if time.monotonic()-start>3600:raise TimeoutError('Status wait ended; cloud fit remains reserved, inspect remotely')
                time.sleep(20)
            r=export_fit(kernel,run/'downloads',verified)
            if r['seed']!=seed or r['condition']!=condition:raise ValueError('Cloud fit identity mismatch')
            results.append(r)
            summary=dict(verified_fits=len(results),verified_optimizer_updates=sum(x['optimizer_updates'] for x in results),
                runs=[dict(seed=x['seed'],condition=x['condition'],checkpoint_sha256=x['checkpoint_sha256'],seconds=x['seconds']) for x in results])
            (campaign/'progress.json').write_text(json.dumps(summary,indent=2))
            print(json.dumps(dict(event='fit_verified',seed=seed,condition=condition,optimizer_updates=r['optimizer_updates'],seconds=r['seconds'],checkpoint_sha256=r['checkpoint_sha256'])),flush=True)
    if len(results)!=9 or sum(r['optimizer_updates'] for r in results)!=18000:raise ValueError('Campaign budget mismatch')
    (campaign/'COMPLETED.json').write_text(json.dumps(dict(status='COMPLETED',fits=9,optimizer_updates=18000),indent=2))
    print(json.dumps(dict(event='campaign_completed',fits=9,optimizer_updates=18000)),flush=True)

if __name__=='__main__':main()
