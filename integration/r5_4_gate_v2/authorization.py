"""Two separate future local human-approval attestations; engineering START never suffices."""
from datetime import datetime,timezone,timedelta
from pathlib import Path
from evidence import GateError,require,load,no_symlink_path,sha

ADOPTION='R54_V2_GATE_ADOPTION'
TRAINING='R54_SCIENTIFIC_TRAINING'
COMMON={'purpose','approved','approved_by','decision_id','issued_at_utc','expires_at_utc','source_commit',
        'v1_lock_sha256','v2_lock_sha256','integration_lock_sha256','qualification_final_receipt_sha256','statement'}
TRAINING_EXTRA={'models','seeds','fits','updates_per_fit','total_optimizer_updates','output_directory'}


def parse_time(value):
    require(type(value) is str,'Explicit UTC authorization time required')
    time=datetime.fromisoformat(value.replace('Z','+00:00'))
    require(time.tzinfo is not None and time.utcoffset()==timedelta(0),'UTC authorization timestamp required')
    return time


def validate_record(record,purpose,proof,source_commit,now,output=None):
    require(type(record) is dict and set(record)==COMMON | (TRAINING_EXTRA if purpose==TRAINING else set()),'Wrong authorization schema/scope')
    require(record['purpose']==purpose and record['approved'] is True,'Distinct explicit future human approval required')
    require(type(record['approved_by']) is str and bool(record['approved_by'].strip()) and type(record['decision_id']) is str and bool(record['decision_id'].strip()),'Human approval provenance required')
    require(record['source_commit']==source_commit,'Stale source-commit authorization')
    for field in ('v1_lock_sha256','v2_lock_sha256','integration_lock_sha256'):
        require(record[field]==proof[field],f'Stale source approval: {field}')
    require(record['qualification_final_receipt_sha256']==proof['final_receipt_sha256'],'Substituted qualification approval')
    issued,expires=map(parse_time,[record['issued_at_utc'],record['expires_at_utc']])
    require(issued<=now<=expires and issued<expires and expires-issued<=timedelta(hours=24),'Expired/future/overlong human authorization')
    if purpose==ADOPTION:
        require(record['statement']=='I approve adoption of the reviewed R5.4 v2 gate integration.','Adoption statement required')
    else:
        require(record['statement']=='I authorize exactly six R5.4 scientific fits, separate from engineering qualification.','Scientific training statement required')
        require(record['models']==['markov','recurrent'] and record['seeds']==[11,23,47] and type(record['fits']) is int and record['fits']==6,'Scientific workload authorization mismatch')
        require(type(record['updates_per_fit']) is int and record['updates_per_fit']==800 and type(record['total_optimizer_updates']) is int and record['total_optimizer_updates']==4800,'Scientific optimizer budget authorization mismatch')
        require(record['output_directory']==str(Path(output).absolute()),'Authorization bound to a different output destination')
    return record


def authorize(adoption_path,training_path,proof,source_commit,output,now=None):
    require(adoption_path is not None and training_path is not None,'Gate adoption and scientific training approvals both required')
    paths=[no_symlink_path(p) for p in (adoption_path,training_path)]
    require(paths[0]!=paths[1] and all(p.is_file() for p in paths),'Two distinct approval files required')
    records=[load(p) for p in paths]
    require(sha(paths[0])!=sha(paths[1]),'Engineering/adoption approval cannot be reused as scientific approval')
    now=now or datetime.now(timezone.utc)
    for record,purpose in zip(records,(ADOPTION,TRAINING)):
        validate_record(record,purpose,proof,source_commit,now,output)
    require(records[0]['decision_id']!=records[1]['decision_id'],'Separate human decisions required')
    return dict(gate_adoption=records[0],scientific_training=records[1],
                gate_adoption_sha256=sha(paths[0]),scientific_authorization_sha256=sha(paths[1]),
                authorized_at_utc=now.isoformat(),authority_limit='Local human approval attestations; not signed identity/security credentials')
