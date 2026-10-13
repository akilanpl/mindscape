"""Proposed read-only post-campaign gate audit; never fitting or neural prediction replay."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from evidence import REPO,GateError,require,load,verify_qualification,sha
from authorization import validate_record,parse_time,ADOPTION,TRAINING
from science_reuse import reuse_descriptor
sys.path.insert(0,str(REPO/'src'))


def verify_provenance(start,proof,destination):
    engineering=start['engineering']
    require(engineering['method']=='R54_V2_GATE_INTEGRATION' and engineering['status']=='ENGINEERING_READY',
            'Explicit v2 gate schema required')
    require(engineering['original_v1_status']=='ENGINEERING_NO_GO' and engineering['v1_engineering_gate_executed'] is False,
            'Original v1 must not be relabeled or claimed as executed')
    require(engineering['qualification_native_status']==proof['qualification_status'],'Wrong qualification status')
    # A byte-identical qualification can be re-audited at a different storage path.
    saved={k:v for k,v in engineering['qualification'].items() if k!='original_evidence_path'}
    actual={k:v for k,v in proof.items() if k!='original_evidence_path'}
    require(saved==actual,'Scientific start contains stale/substituted qualification provenance')
    require(engineering['scientific_reuse_descriptor']==reuse_descriptor(),'Scientific execution body changed')
    commit=start['commit']
    require(commit==engineering['scientific_source_commit'],'Execution source attribution mismatch')
    for filename,digest in [('experiments/r5_4/frozen/LOCK.json',proof['v1_lock_sha256']),
                            ('engineering/r5_4_v2/LOCK.json',proof['v2_lock_sha256']),
                            ('integration/r5_4_gate_v2/LOCK.json',proof['integration_lock_sha256'])]:
        stored=subprocess.check_output(['git','show',f'{commit}:{filename}'],cwd=REPO)
        require(hashlib.sha256(stored).hexdigest()==digest,'Recorded execution commit incompatible with reviewed locks')
    approvals=engineering['authorizations']
    at_start=parse_time(approvals['authorized_at_utc'])
    for key,purpose in [('gate_adoption',ADOPTION),('scientific_training',TRAINING)]:
        validate_record(approvals[key],purpose,proof,commit,at_start,destination)
    require(approvals['gate_adoption']['decision_id']!=approvals['scientific_training']['decision_id'],'Decisions not distinct')
    return dict(status='VERIFIED_V2_SCIENTIFIC_GATE_PROVENANCE',qualification_receipt_sha256=proof['receipt_sha256'],
                qualification_final_receipt_sha256=proof['final_receipt_sha256'],source_commit=commit,
                authorization_time_utc=approvals['authorized_at_utc'],original_v1_status='ENGINEERING_NO_GO',
                limitation='Local human authorization records are auditable attestations, not signed identity credentials')


def audit(destination,qualification):
    destination=Path(destination)
    require((destination/'COMPLETE.json').is_file(),'No completed scientific campaign; no scientific audit entered')
    proof=verify_qualification(qualification)
    provenance=verify_provenance(load(destination/'start.json'),proof,destination)
    from mindscape.r5_4.audit import audit as frozen_audit
    metrics=frozen_audit(destination,replay_mps=False)
    return dict(gate_provenance=provenance,frozen_scientific_audit=metrics,neural_replay_executed=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--qualification',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(audit(args.campaign,args.qualification),indent=2,sort_keys=True,allow_nan=False))
