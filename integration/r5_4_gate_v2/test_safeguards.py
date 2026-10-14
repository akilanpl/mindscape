"""Synthetic fail-closed fixtures and AST-only equivalence. No optimizer/model/OOD inference."""
import ast
import copy
from datetime import datetime,timezone,timedelta
import hashlib
import json
from pathlib import Path
import sys
import pytest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import evidence
import authorization
import launcher
import science_reuse


def dump(path,value):
    path.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')


def synthetic_evidence(root,**changes):
    root.mkdir()
    start=dict(approved_engineering_only=True,training_authorized=False)
    result=dict(status=evidence.READY,scratch_optimizer_steps=144,campaign_optimizer_updates=0,training_authorized=False,results=[{}]*6)
    audit=dict(status=evidence.READY,independent_audit='VERIFIED',scratch_optimizer_steps=144,campaign_optimizer_updates=0,training_authorized=False,assessments=[{}]*6)
    for name,value in [('START.json',start),('RESULT.json',result),('INDEPENDENT_AUDIT.json',audit)]:
        value.update(changes.get(name,{}));dump(root/name,value)
    for i in range(6):
        for suffix in ('.json','.progress.jsonl','.scratch-progress.jsonl','.restoration-and-gradients.pt'):
            (root/f'worker-{i}{suffix}').write_bytes(b'SYNTHETIC_NONMODEL_PLACEHOLDER')
    repair_receipts(root)
    return {p.name:evidence.sha(p) for p in root.iterdir()}


def repair_receipts(root):
    names={p.name for p in root.iterdir()}-{'RECEIPT.json','FINAL_RECEIPT.json','INDEPENDENT_AUDIT.json'}
    dump(root/'RECEIPT.json',dict(sha256={name:evidence.sha(root/name) for name in sorted(names)}))
    dump(root/'FINAL_RECEIPT.json',dict(sha256={p.name:evidence.sha(p) for p in sorted(root.iterdir()) if p.name!='FINAL_RECEIPT.json'}))


def proof():
    return dict(v1_lock_sha256='v1',v2_lock_sha256='v2',integration_lock_sha256='integration',final_receipt_sha256='qualification')


def approval(purpose,now,output):
    result=dict(purpose=purpose,approved=True,approved_by='SYNTHETIC_TEST_ONLY',decision_id=purpose,
        issued_at_utc=(now-timedelta(minutes=1)).isoformat(),expires_at_utc=(now+timedelta(minutes=10)).isoformat(),
        source_commit='reviewed',**proof(),statement='I approve adoption of the reviewed R5.4 v2 gate integration.')
    result['qualification_final_receipt_sha256']=result.pop('final_receipt_sha256')
    if purpose==authorization.TRAINING:
        result.update(statement='I authorize exactly six R5.4 scientific fits, separate from engineering qualification.',
            models=['markov','recurrent'],seeds=[11,23,47],fits=6,updates_per_fit=800,total_optimizer_updates=4800,
            output_directory=str(output.absolute()))
    return result


def test_synthetic_receipt_and_readonly_snapshot(tmp_path):
    root=tmp_path/'original';manifest=synthetic_evidence(root)
    snapshot=tmp_path/'snapshot';snapshot.mkdir()
    evidence.pinned_snapshot(root,manifest,snapshot)
    receipt=evidence.validate_receipts(snapshot)
    assert receipt['qualification_status']==evidence.READY
    assert manifest=={p.name:evidence.sha(p) for p in root.iterdir()}


def test_missing_evidence_blocks(tmp_path):
    snapshot=tmp_path/'snapshot';snapshot.mkdir()
    with pytest.raises(evidence.GateError,match='Missing'):
        evidence.pinned_snapshot(tmp_path/'missing',{},snapshot)


@pytest.mark.parametrize('name',['FINAL_RECEIPT.json','INDEPENDENT_AUDIT.json','worker-5.json'])
def test_partial_evidence_blocks(tmp_path,name):
    root=tmp_path/'original';manifest=synthetic_evidence(root)
    (root/name).unlink()
    snapshot=tmp_path/'snapshot';snapshot.mkdir()
    with pytest.raises(evidence.GateError,match='inventory'):
        evidence.pinned_snapshot(root,manifest,snapshot)


@pytest.mark.parametrize('mutation',[
    {'RESULT.json':{'status':'R54_V2_ENGINEERING_NO_GO'}},
    {'INDEPENDENT_AUDIT.json':{'independent_audit':'PARTIAL_REJECTION_ONLY'}},
    {'RESULT.json':{'scratch_optimizer_steps':143}},
    {'RESULT.json':{'campaign_optimizer_updates':1}},
    {'RESULT.json':{'training_authorized':True}},
    {'RESULT.json':{'results':[{}]*5}}])
def test_failed_incomplete_or_wrong_scope_blocks_with_repaired_hashes(tmp_path,mutation):
    root=tmp_path/'original';synthetic_evidence(root,**mutation)
    with pytest.raises(evidence.GateError):
        evidence.validate_receipts(root)


def test_substitution_and_stale_receipts_block(tmp_path):
    root=tmp_path/'original';manifest=synthetic_evidence(root)
    (root/'worker-0.json').write_bytes(b'SUBSTITUTED')
    snapshot=tmp_path/'snapshot';snapshot.mkdir()
    with pytest.raises(evidence.GateError,match='Pinned evidence hash'):
        evidence.pinned_snapshot(root,manifest,snapshot)
    with pytest.raises(evidence.GateError,match='receipt'):
        evidence.validate_receipts(root)
    repair_receipts(root)
    second=tmp_path/'second-snapshot';second.mkdir()
    with pytest.raises(evidence.GateError,match='Pinned evidence hash'):
        evidence.pinned_snapshot(root,manifest,second)


@pytest.mark.parametrize('which',['directory','parent','file'])
def test_symlinked_evidence_rejected(tmp_path,which):
    root=tmp_path/'original';manifest=synthetic_evidence(root)
    target=root
    if which=='directory':
        target=tmp_path/'alias';target.symlink_to(root,target_is_directory=True)
    elif which=='parent':
        alias=tmp_path/'alias';alias.symlink_to(tmp_path,target_is_directory=True)
        target=alias/'original'
    else:
        (root/'worker-0.json').unlink();(root/'worker-0.json').symlink_to(root/'worker-1.json')
    snapshot=tmp_path/'snapshot';snapshot.mkdir()
    with pytest.raises((evidence.GateError,OSError)):
        evidence.pinned_snapshot(target,manifest,snapshot)


def test_duplicate_and_nonfinite_json_rejected(tmp_path):
    target=tmp_path/'input.json';target.write_text('{"approved": false, "approved": true}')
    with pytest.raises(evidence.GateError,match='Duplicate'):
        evidence.load(target)
    target.write_text('{"value": NaN}')
    with pytest.raises(evidence.GateError,match='Nonfinite'):
        evidence.load(target)


def test_tampered_scientific_hash_blocks(tmp_path,monkeypatch):
    source=tmp_path/'source.py';source.write_text('modified')
    pin=dict(v1_lock_sha256='wrong',v2_lock_sha256='wrong')
    monkeypatch.setattr(evidence,'load',lambda path:pin)
    with pytest.raises(evidence.GateError,match='stale v1/v2 locks'):
        evidence.verify_sources()


def test_engineering_approval_never_authorizes_science(tmp_path):
    now=datetime(2026,10,13,12,tzinfo=timezone.utc)
    engineering=dict(approved_engineering_only=True,training_authorized=False,scratch_optimizer_steps=144)
    with pytest.raises(evidence.GateError,match='schema'):
        authorization.validate_record(engineering,authorization.TRAINING,proof(),'reviewed',now,tmp_path/'output')
    adoption=approval(authorization.ADOPTION,now,tmp_path/'output')
    with pytest.raises(evidence.GateError):
        authorization.validate_record(adoption,authorization.TRAINING,proof(),'reviewed',now,tmp_path/'output')


@pytest.mark.parametrize('change',['commit','receipt','expired','fits','updates','destination','not_approved'])
def test_stale_substituted_or_wrong_scientific_authorization(tmp_path,change):
    now=datetime(2026,10,13,12,tzinfo=timezone.utc);output=tmp_path/'output'
    record=approval(authorization.TRAINING,now,output)
    if change=='commit':record['source_commit']='stale'
    elif change=='receipt':record['qualification_final_receipt_sha256']='substitute'
    elif change=='expired':record['expires_at_utc']=(now-timedelta(seconds=1)).isoformat()
    elif change=='fits':record['fits']=7
    elif change=='updates':record['updates_per_fit']=801
    elif change=='destination':record['output_directory']='other'
    else:record['approved']=False
    with pytest.raises(evidence.GateError):
        authorization.validate_record(record,authorization.TRAINING,proof(),'reviewed',now,output)


def test_two_future_decisions_validate_only_in_dry_run(tmp_path):
    now=datetime(2026,10,13,12,tzinfo=timezone.utc);output=tmp_path/'output'
    a,b=tmp_path/'adoption.json',tmp_path/'science.json'
    dump(a,approval(authorization.ADOPTION,now,output));dump(b,approval(authorization.TRAINING,now,output))
    result=authorization.authorize(a,b,proof(),'reviewed',output,now)
    assert result['gate_adoption_sha256']!=result['scientific_authorization_sha256']
    assert not output.exists()  # synthetic authorization validation never invokes science
    with pytest.raises(evidence.GateError,match='distinct'):
        authorization.authorize(a,a,proof(),'reviewed',output,now)


def test_missing_scientific_authorization_blocks_before_any_model_import(tmp_path,monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('Scientific boundary reached')
    monkeypatch.setattr(launcher,'invoke_frozen_science',forbidden)
    monkeypatch.setattr(launcher,'verify_qualification',forbidden)
    with pytest.raises(evidence.GateError,match='approvals absent'):
        launcher.execute(tmp_path/'native',tmp_path/'engineering_approval.json',None,tmp_path/'campaign')


def test_default_invocation_review_only_and_zero_science(monkeypatch,capsys):
    monkeypatch.setattr(sys,'argv',['launcher.py'])
    monkeypatch.setattr(launcher,'verify_qualification',lambda p:{'test':'readonly synthetic proof'})
    def forbidden(*args,**kwargs):raise AssertionError('Execution forbidden in default invocation')
    monkeypatch.setattr(launcher,'execute',forbidden)
    launcher.main()
    result=json.loads(capsys.readouterr().out)
    assert result['mode']=='NONTRAINING_REVIEW_ONLY' and result['scientific_optimizer_updates']==0 and result['neural_ood_predictions']==0
    assert result['gate_adopted'] is False and result['scientific_training_authorized'] is False


def test_unverified_readiness_cannot_be_adapted():
    with pytest.raises((evidence.GateError,KeyError)):
        launcher.compatibility_readiness({'qualification_status':'ENGINEERING_NO_GO'},None,'reviewed')


def test_scientific_body_is_exact_ast_suffix_compile_only():
    source=(evidence.REPO/'src/mindscape/r5_4/campaign.py').read_text()
    original,body,boundary=science_reuse.scientific_suffix(source)
    assert boundary==6 and body[0].lineno==31
    descriptor=science_reuse.reuse_descriptor()
    assert descriptor==evidence.load(HERE/'science-equivalence.json')
    code=science_reuse.compile_scientific_body()  # compile-only, never exec or invoke
    assert code.co_name=='<module>'
    assert ast.dump(ast.Module(body=body,type_ignores=[]),include_attributes=False)==ast.dump(ast.Module(body=original.body[boundary:],type_ignores=[]),include_attributes=False)
    text=ast.unparse(ast.Module(body=body,type_ignores=[]))
    assert 'for kind in PROTOCOL' in text and 'for seed in PROTOCOL' in text
    assert text.index('torch.save')<text.index('free.extend')<text.index('PREDICTIONS_SEALED.json')<text.index('local_diagnostics')<text.index('audit(destination')
    assert 'check(tasks)' not in text and 'run(' not in text


def test_no_fallthrough_monkeypatch_optimizer_or_evaluation_in_launcher_source():
    for filename in ('launcher.py','science_reuse.py','evidence.py','authorization.py'):
        tree=ast.parse((HERE/filename).read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
                assert node.func.attr not in ('AdamW','Adam','SGD','step','predict_rows','local_diagnostics')
                if node.func.attr=='run':
                    assert isinstance(node.func.value,ast.Name) and node.func.value.id=='subprocess'
                    assert filename=='evidence.py'  # storage-only re-audit, never the scientific command
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Name):
                assert node.func.id not in ('fit','check','adapter','rollout')
            if isinstance(node,ast.Assign):
                for target in node.targets:
                    assert not (isinstance(target,ast.Attribute) and target.attr in ('check','run'))


def test_v1_inventory_and_original_no_go_intact():
    pin=evidence.load(HERE/'qualification-pin.json')
    v1=evidence.REPO/'experiments/r5_4/frozen/LOCK.json'
    assert evidence.sha(v1)==pin['v1_lock_sha256']
    for name,digest in evidence.load(v1)['sha256'].items():
        assert evidence.sha(evidence.REPO/name)==digest
    assert evidence.sha(evidence.REPO/'experiments/r5_4/engineering.json')==pin['original_engineering_sha256']
    assert evidence.load(evidence.REPO/'experiments/r5_4/engineering.json')['status']=='ENGINEERING_NO_GO'


def test_no_scientific_audit_on_incomplete_campaign(tmp_path,monkeypatch):
    import audit_future_campaign
    def forbidden(*args,**kwargs):raise AssertionError('Scientific artifact inspection forbidden')
    monkeypatch.setattr(audit_future_campaign,'verify_qualification',forbidden)
    with pytest.raises(evidence.GateError,match='No completed'):
        audit_future_campaign.audit(tmp_path/'unrun',tmp_path/'native')
