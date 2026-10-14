"""Review-only by default. Proposed execution requires separately approved adoption AND scientific training."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from evidence import REPO,HERE,READY,GateError,require,verify_qualification,verify_sources,load,sha,no_symlink_path
from authorization import authorize
from science_reuse import reuse_descriptor,invoke_frozen_science
sys.path.insert(0,str(REPO/'src'))


def compatibility_readiness(proof,approvals,source_commit):
    require(proof['qualification_status']==READY and proof['independent_audit']=='VERIFIED' and proof['scientific_updates']==0 and proof['disposable_steps']==144,'No generic readiness without proven native v2 qualification')
    require(approvals['gate_adoption']['purpose']=='R54_V2_GATE_ADOPTION' and approvals['gate_adoption']['approved'] is True and approvals['scientific_training']['purpose']=='R54_SCIENTIFIC_TRAINING' and approvals['scientific_training']['approved'] is True,'Two explicit approvals required before readiness adaptation')
    return dict(status='ENGINEERING_READY',schema_compatibility='Generic ready label required by unchanged v1 scientific verifier; not a v1 qualification PASS',
        method='R54_V2_GATE_INTEGRATION',qualification_native_status=READY,original_v1_status='ENGINEERING_NO_GO',
        v1_engineering_gate_executed=False,qualification=proof,authorizations=approvals,
        scientific_source_commit=source_commit,scientific_reuse_descriptor=reuse_descriptor())


def native_runtime_check(proof):
    import os,platform,torch
    qualified=proof['qualified_runtime']
    require(platform.system()=='Darwin' and torch.backends.mps.is_available() and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK','0')=='0','Native MPS unavailable or fallback enabled')
    require(str(torch.__version__)==qualified['torch'] and platform.machine()==qualified['machine'] and platform.platform()==qualified['platform'],'Runtime differs from reviewed native qualification; new review required')


def execute(qualification,adoption,training,destination,execution_state=None):
    # No import of model/campaign, optimizer, fit or evaluation before both explicit approvals.
    require(adoption is not None and training is not None,'Execution blocked: separate future gate-adoption and scientific-training approvals absent')
    require(destination is not None and not Path(destination).exists(),'New authorized output directory required')
    no_symlink_path(destination)
    proof=verify_qualification(qualification)
    source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    approvals=authorize(adoption,training,proof,source_commit,destination)
    require(Path.cwd().resolve()==REPO.resolve(),'Execute only from reviewed checkout root')
    native_runtime_check(proof)
    require(verify_sources()['integration_lock_sha256']==proof['integration_lock_sha256'],'Review freeze changed before scientific boundary')
    # Preserve the same frozen dataset qualification and training-task load as the v1 preamble.
    from mindscape.r5_4.freeze import verify_lock
    from mindscape.r5_4.data import qualify,ROOT
    verify_lock()
    require(qualify()['status']=='DATA_READY','Original scientific DATA_NO_GO')
    tasks=load(ROOT/'train.json')
    readiness=compatibility_readiness(proof,approvals,source_commit)
    # Explicitly NOT frozen.campaign.run: that entry point still runs the preserved v1 gate.
    if execution_state is not None:
        execution_state['scientific_body_entered']=True
    return invoke_frozen_science(destination,tasks,readiness)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qualification',type=Path,default=Path('results/r5_4/engineering-v2-native'))
    parser.add_argument('--gate-adoption',type=Path)
    parser.add_argument('--scientific-authorization',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--execute-scientific-training',action='store_true')
    args=parser.parse_args()
    execution_state={'scientific_body_entered':False}
    try:
        if args.execute_scientific_training:
            result=execute(args.qualification,args.gate_adoption,args.scientific_authorization,args.output,execution_state)
        else:
            proof=verify_qualification(args.qualification)
            result=dict(status='R54_V2_GATE_INTEGRATION_REVIEW_READY',mode='NONTRAINING_REVIEW_ONLY',
                proof=proof,science_reuse=reuse_descriptor(),gate_adopted=False,scientific_training_authorized=False,
                scientific_optimizer_updates=0,neural_ood_predictions=0)
    except (GateError,ValueError,OSError,KeyError,TypeError,subprocess.SubprocessError) as error:
        result=dict(status='R54_V2_GATE_INTEGRATION_BLOCKED',error=str(error),scientific_body_entered=execution_state['scientific_body_entered'],
                    scientific_training_started='UNDETERMINED_INSPECT_PRESERVED_PROGRESS' if execution_state['scientific_body_entered'] else False)
        print(json.dumps(result,indent=2,sort_keys=True))
        raise SystemExit(2)
    print(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))


if __name__=='__main__':
    main()
