"""Prepare data and a bounded recovery proposal; never submits cloud jobs."""
import argparse,copy,hashlib,json
from pathlib import Path
from collections import Counter
from research.r6 import common
from research.r6.datasets import mathematics,coding,build

TRAINER_KEYS=('schema','status','authorized_to_submit','seed','architecture','fit_ids','updates_per_fit_max','total_optimizer_update_ceiling','wall_seconds_per_fit_max','total_cuda_fit_seconds_ceiling','optimizer','sampling','train_sha256','dev_sha256','checkpoint_policy','stop_conditions','metrics','trainer_code_sha256')
def trainer_config(config):
    """Never expose protected examples, seeds, hashes or generator code."""
    return {key:config[key] for key in TRAINER_KEYS if key in config}

def run(r6,repo,output):
    r6,repo,output=map(Path,(r6,repo,output));data=output/'pilot-data';data.mkdir(exist_ok=False)
    old=[]
    for f in ['train.json','dev.json','protected/final_iid.json','protected/final_ood.json']:old+=common.read(r6/'dataset-v1'/f)
    seenkeys={r['canonical_key'] for r in old};seentext={r['input'] for r in old}
    counts={'train':16,'dev':32,'final_iid':32,'final_ood':32};seeds={'train':61117,'dev':62131,'final_iid':63149,'final_ood':64151};stats={};files={}
    for split,count in counts.items():
        rows=[];stats[split]={}
        for domain,generator in [('mathematics',mathematics),('coding',coding)]:
            candidates=generator.generate(split,seeds[split],512)
            topics=sorted({r['topic'] for r in candidates});selected=[];answers=set();per=Counter()
            while len(selected)<count:
                progress=False
                for topic in topics:
                    for r in candidates:
                        if r['topic']!=topic or r['canonical_key'] in seenkeys or r['input'] in seentext or r['target']['answer'] in answers:continue
                        build.verify_row(r);seenkeys.add(r['canonical_key']);seentext.add(r['input']);answers.add(r['target']['answer']);selected.append(r);per[topic]+=1;progress=True;break
                    if len(selected)==count:break
                if not progress:raise ValueError('Not enough verified novel diverse records')
            rows+=selected;stats[split][domain]={'count':len(selected),'distinct_answers':len(answers),'topics':dict(per)}
        path=data/('protected/'+split+'.json' if split.startswith('final_') else split+'.json');common.write(path,rows);files[str(path.relative_to(data))]={'sha256':common.sha(path),'count':len(rows)}
    manifest={'schema':'mindscape-r61-tiny-data-v1','seeds':seeds,'files':files,'statistics':stats,'exact_and_canonical_overlap_with_all_r6':0,'across_r61_splits_overlap':0,'selection':'Round-robin topics; unique target answers within each domain/split. Designed diagnostic distribution, not natural corpus prevalence.','oracles':'Math restricted AST fraction and structured oracle; coding restricted AST interpreter and independent template oracle. External to neural predictor.'}
    common.write(data/'manifest.json',manifest)
    source={str(p.relative_to(repo)):common.sha(p) for d in ['research/r6','research/r61'] for p in sorted((repo/d).rglob('*.py')) if '__pycache__' not in p.parts}
    cfg={'schema':'mindscape-r61-pilot-config-v1','status':'PROPOSED_AWAITING_EXPLICIT_USER_BUDGET_APPROVAL','authorized_to_submit':False,'physics_followup':'ON_HOLD_DO_NOT_EXECUTE','seed':61117,'device':'Kaggle CUDA cuda:0, one T4; validate quota and actual hardware before any future submission','architecture':{'baseline':'Unmodified R6 native-relational width48 final-GRU-state encoder','corrected':'Same exact tensors/parameter count, masked-mean of all valid GRU encoder output states','parameter_count_each':54014,'only_intervention':'encoder sequence pooling','pretrained':False,'physics_training':False},'fit_ids':['r61-tiny-baseline-s61117','r61-tiny-pooled-s61117'],'updates_per_fit_max':1200,'total_optimizer_update_ceiling':2400,'wall_seconds_per_fit_max':300,'total_cuda_fit_seconds_ceiling':600,'expected_seconds_per_fit':[30,180],'time_estimate':'Estimate based on R6 T4 receipts; provisioning/queue/quota accounting additional; not guaranteed.','optimizer':{'type':'Adam','lr':0.001,'schedule':'constant','clip_norm':1.0,'precision':'FP32','weight_decay':0.0},'sampling':'Full fixed batch of all 32 tiny training cases each update, matched case order; CE over all non-PAD answer tokens; no physics loss','data':manifest,'source_sha256':source,'train_sha256':files['train.json']['sha256'],'dev_sha256':files['dev.json']['sha256'],'trainer_inputs':['train.json','dev.json','config.json','model and codec source only'],'protected_files_not_in_trainer':['protected/final_iid.json','protected/final_ood.json'],'checkpoint_policy':'Initial and fixed final-update checkpoint; no best-dev selection and no silent resume/retry. Record and flag partial runs; no causal ranking from unequal completed steps.','stop_conditions':['No user approval: no submission or optimizer construction','Not Linux Kaggle Batch CUDA or insufficient approved free quota: abort before optimizer','Nonfinite loss/gradients/weights or package hash mismatch: abort','1200 actual optimizer updates or 300 elapsed fitting seconds per fit, whichever comes first; no auto retry','Any failure consumes actual updates and stops campaign for review; do not silently spend unused quota'],'metrics':['Native greedy train/dev exact accuracy per domain every100 updates','Teacher-forced CE and token accuracy separately labeled','Per-layer gradient norms and initial-to-final weight changes','Distinct valid output count and output entropy','Input sensitivity and semantics-preserving probes','Separate CPU protected scoring on identical cases for both models; zero updates'],'tiny_memorization_acceptance':{'min_train_exact_accuracy_each_domain':0.95,'min_distinct_valid_outputs_each_domain':15,'cannot_claim_generalization':True},'generalization_acceptance':'Predeclared per-domain exact accuracy greater than train-derived frequency baseline plus5 percentage points on fresh IID cases; report OOD and perturbation failures. Single seed remains exploratory.','comparators':['Matched from-scratch unchanged R6 baseline','Frozen original R6 selected checkpoint, inference only, same cases','Constant 0 and train-only majority/frequency predictors','External deterministic reference solvers labeled separately'],'independent_evaluation':'After future approved fitting, private CPU Kaggle batch; read-only frozen checkpoint inputs; no optimizers; fresh protected data absent from trainers; same-account controller trust disclosed','submission_policy':'Private only, internet disabled, no paid resources, no public upload; no submissions have been made'}
    cfg['trainer_code_sha256']={name:common.sha(repo/name) for name in ['research/r6/common.py','research/r6/architectures/model.py','research/r61/model.py','research/r61/pilot_worker.py']}
    common.write(output/'trainer-config.json',trainer_config(cfg))
    cfg['trainer_config_sha256']=common.sha(output/'trainer-config.json')
    common.write(output/'R61_CLOUD_PILOT_CONFIG.json',cfg)
    print(json.dumps({'status':cfg['status'],'data_files':files,'authorized_to_submit':False}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--r6',required=True);p.add_argument('--repo',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.r6,a.repo,a.output)
