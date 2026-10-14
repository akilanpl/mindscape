import copy
import inspect
import json
import base64
import pytest
import torch
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization
from mindscape.alpha_ood.benchmark import Task,digest
from mindscape.alpha_ood.models import LocalTransition
from mindscape.alpha_ood.numerical_v2 import adapter as a
from mindscape.alpha_ood.numerical_v2 import custody as c
from mindscape.alpha_ood.numerical_v2 import evaluation as e
from mindscape.r5_2_math.verification import demonstration

# Every fixture is fabricated, tiny and training-only. No protected dataset draw.
def fixture():
    tasks=(Task(9,(('add',2),('sub',3)),'train','familiar'),Task(3,(('mul',2),),'train','familiar'))
    traces=[demonstration(a.native_task(t)) for t in tasks]
    return tasks,traces

@pytest.mark.parametrize('kind',['recurrent','feedforward'])
def test_real_loss_backward_no_update(kind):
    tasks,traces=fixture(); model=LocalTransition(kind,11)
    before={k:v.clone() for k,v in model.state_dict().items()}
    value=a.loss(model,a.package_training(tasks,traces),'cpu'); value.backward()
    assert value.isfinite() and value.item()>0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())

def test_initialization_parity():
    r=LocalTransition('recurrent',23); f=LocalTransition('feedforward',23)
    for k,v in r.state_dict().items():
        if not k.startswith('core.'):
            assert torch.equal(v,f.state_dict()[k])

def test_schedule_and_optimizer_static():
    assert a.batch_plan(11).shape==(800,32)
    assert (a.batch_plan(11)==a.batch_plan(11)).all()
    assert not (a.batch_plan(11)==a.batch_plan(23)).all()
    src=inspect.getsource(a.fit_native)
    assert src.index('require_access(')<src.index('LocalTransition(')
    assert src.count('optimizer.step()')==1 and 'for indices in plan' in src
    assert "checkpoint.open('xb')" in src
    with pytest.raises(ValueError): a.batch_plan(11,2)

def test_no_eval_teacher_inputs():
    tasks,traces=fixture()
    with pytest.raises(ValueError):
        a.package_training((Task(1,(('add',1),),'ood','operand_combinations'),),traces[:1])
    source=inspect.getsource(a.Predictor)+inspect.getsource(a.predict_task)
    assert 'verification' not in source and 'reference(' not in source

def signed(private,purpose,binding,sequence=1,decision=None):
    payload={'purpose':purpose,'binding':binding,'signer':'synthetic_test_signer',
             'approved':True,'decision_id':decision or 'synthetic-'+purpose,'sequence':sequence}
    raw=json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    return {'payload':payload,'signature':base64.b64encode(private.sign(raw)).decode()}

# Ephemeral private key generated only for fabricated test bindings, never saved
# or used for real authorization. No approval records are written to repository.
def trust():
    private=Ed25519PrivateKey.generate()
    public=private.public_key().public_bytes(serialization.Encoding.Raw,serialization.PublicFormat.Raw)
    return private,{(p,'synthetic_test_signer'):public for p in (*c.REQUIRED,'independent_eligibility','seal_before_label_release')}

def binding():
    value={k:digest('synthetic-'+k) for k in c.BINDINGS}; value['numerical_spec_hash']=digest(a.SPEC)
    return value

def test_fail_closed_signatures_and_binding():
    private,keys=trust(); b=binding()
    receipts={p:signed(private,p,b) for p in (*c.REQUIRED,'independent_eligibility')}
    access=c.authorize('fit',b,receipts,keys); c.require_access(access,'fit',b)
    wrong=copy.deepcopy(receipts); wrong['fitting_authorization']['payload']['binding']['source_hash']=digest('wrong')
    with pytest.raises(ValueError): c.authorize('fit',b,wrong,keys)
    with pytest.raises(ValueError): c.authorize('fit',b,receipts,{})
    with pytest.raises(ValueError): c.authorize('fit',b,{},keys)
    with pytest.raises(ValueError): a.fit_native(tasks=(),traces=[],kind='recurrent',seed=11,bindings=b,authorization=None,checkpoint=None)

def test_fabricated_boolean_not_signature():
    b=binding(); receipts={p:{'payload':{'purpose':p,'binding':b,'signer':'invented','approved':True,'decision_id':p,'sequence':1},'signature':'true'} for p in (*c.REQUIRED,'independent_eligibility')}
    with pytest.raises(ValueError): c.authorize('fit',b,receipts,{})

def test_seal_order_missing_substitution(tmp_path):
    task=fixture()[0][0]; memory={}; table={}
    rows=[]
    for condition in e.SEALED_CONDITIONS:
        for seed in e.SEEDS:
            rows.append({'identity':task.identity,'condition':condition,'seed':seed,'result':None,'failure':{'type':'synthetic_failure'}})
    seal=e.seal_predictions((task,),rows,binding(),tmp_path/'synthetic.json')
    assert seal['prediction_hash']==digest(rows) and seal['labels_released'] is False
    for bad in (rows[:-1],list(reversed(rows)),[{**rows[0],'identity':'wrong'},*rows[1:]]):
        with pytest.raises(ValueError): e.validate_records((task,),bad)
    with pytest.raises(FileExistsError): e.seal_predictions((task,),rows,binding(),tmp_path/'synthetic.json')

def test_prediction_feedback_replay_synthetic_only():
    task=fixture()[0][0]; model=LocalTransition('recurrent',47)
    row=a.predict_task(task,model,'cpu')
    assert row['failure'] is None
    e.replay_neural(task,row['result'])
    result=copy.deepcopy(row['result']); result['transitions'][0]['state']['incoming']=7
    with pytest.raises(ValueError): e.replay_neural(task,result)

def test_public_support_requires_full_dataset():
    with pytest.raises(ValueError): c.observable_report(fixture()[0],set())

def test_frozen_proposal_unchanged():
    protocol=json.load(open('configs/alpha_ood/protocol-v1.json'))
    assert protocol['counts']=={'train':768,'validation':96,'ood_per_family':36}
    assert protocol['updates_per_fit']==800 and protocol['batch_size']==32
    assert protocol['gates']['gain_over_each_nonprocedural_baseline_min']==.10

def test_full_synthetic_sealed_scoring_and_corruption(tmp_path):
    # Explicit fixture train tasks are not protected OOD; scorer exercise retains
    # six family denominators, never generates frozen benchmark identities.
    tasks=tuple(Task(2+i,(('add',1),),'train',family) for i,family in enumerate(e.FAMILIES))
    labels={t.identity:[t.start+1] for t in tasks}; b=binding(); b['label_commitment']=digest(labels)
    private,keys=trust(); receipts={p:signed(private,p,b) for p in (*c.REQUIRED,'independent_eligibility')}
    access=c.authorize('evaluate',b,receipts,keys)
    rows=[]
    for condition in e.SEALED_CONDITIONS:
        for seed in e.SEEDS:
            for task in tasks:
                if condition in ('recurrent','feedforward'):
                    row=a.predict_task(task,LocalTransition(condition,seed),'cpu')
                else:
                    row={'identity':task.identity,'result':{'answer':str(task.start+1),'stages':[{'answer':str(task.start+1)}],'transitions':[]},'failure':None}
                rows.append({'condition':condition,'seed':seed,**row})
    seal=e.seal_predictions(tasks,rows,b,tmp_path/'synthetic-full.json')
    release_binding={**b,'prediction_hash':seal['prediction_hash'],'ordered_identity_hash':seal['ordered_identity_hash']}
    release=signed(private,'seal_before_label_release',release_binding,sequence=2)
    result=e._score_released(tasks=tasks,seal=seal,labels=labels,release_receipt=release,access=access,binding=b,trusted_keys=keys)
    assert result['unique_tasks']==6
    assert result['metrics']['recurrent']['11']['overall']['independent_tasks']==6
    assert result['scientific_status']=='NO_GO_UNVERIFIED'
    corrupt=copy.deepcopy(seal); corrupt['rows'][0]['result']['answer']='999'
    with pytest.raises(ValueError): e._score_released(tasks=tasks,seal=corrupt,labels=labels,release_receipt=release,access=access,binding=b,trusted_keys=keys)
    with pytest.raises(ValueError): e._score_released(tasks=tasks,seal=seal,labels=labels,release_receipt=signed(private,'seal_before_label_release',release_binding,sequence=1),access=access,binding=b,trusted_keys=keys)


def test_scientific_scoring_rejects_miniature():
    with pytest.raises(ValueError): e.score_released(tasks=fixture()[0])


def test_ood_prediction_requires_separate_access_before_forward():
    task=Task(1,(('add',1),),'ood','operand_combinations')
    with pytest.raises(ValueError): a.predict_task(task,None,'cpu')


def test_boolean_prediction_vocabulary_rejected():
    task=fixture()[0][0]; result=a.predict_task(task,LocalTransition('recurrent',11),'cpu')['result']
    result['stages'][0]['sign']=True
    with pytest.raises(ValueError): e.replay_neural(task,result)

def test_source_inventory_requires_coverage_and_current_bytes(tmp_path):
    from mindscape.alpha_ood.numerical_v2 import boundary as boundary
    import hashlib
    manifest={}
    for relative in boundary.MANDATORY:
        path=tmp_path/relative; path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text('fabricated synthetic source fixture\n')
        manifest[relative]=hashlib.sha256(path.read_bytes()).hexdigest()
    b={'source_hash':digest(manifest)}
    boundary.verify_source_manifest(manifest,b,root=tmp_path)
    omitted=dict(manifest); omitted.pop(next(iter(omitted)))
    with pytest.raises(ValueError): boundary.verify_source_manifest(omitted,{'source_hash':digest(omitted)},root=tmp_path)
    path=tmp_path/next(iter(manifest)); path.write_text('changed synthetic bytes')
    with pytest.raises(ValueError): boundary.verify_source_manifest(manifest,b,root=tmp_path)

def test_source_paths_reject_traversal_and_symlinks(tmp_path):
    from mindscape.alpha_ood.numerical_v2 import boundary
    import hashlib
    manifest={}
    for relative in boundary.MANDATORY:
        path=tmp_path/relative; path.parent.mkdir(parents=True,exist_ok=True); path.write_text('synthetic')
        manifest[relative]=hashlib.sha256(path.read_bytes()).hexdigest()
    traversal={**manifest,'../escaped':digest('synthetic')}
    with pytest.raises(ValueError): boundary.verify_source_manifest(traversal,{'source_hash':digest(traversal)},root=tmp_path)
    selected=next(iter(manifest)); target=tmp_path/'synthetic-target'; target.write_text('synthetic')
    path=tmp_path/selected; path.unlink(); path.symlink_to(target)
    with pytest.raises(ValueError): boundary.verify_source_manifest(manifest,{'source_hash':digest(manifest)},root=tmp_path)

def test_runtime_policy_snapshot_mismatch_and_native_guard(monkeypatch):
    from mindscape.alpha_ood.numerical_v2 import boundary
    policy={'snapshot':boundary.runtime_snapshot(),'device':'mps','dtype':'float32','status':'PROPOSED_REQUIRES_REVIEW'}
    bad=copy.deepcopy(policy); bad['snapshot']['torch_version']='fabricated-mismatch'
    with pytest.raises(ValueError): boundary.verify_runtime_policy(bad,{'runtime_policy_hash':digest(bad)})
    called=[]
    import mindscape.v3.physics_world.accelerator as accelerator
    def synthetic_native_rejection():
        called.append(True); raise ValueError('synthetic native guard rejection')
    monkeypatch.setattr(accelerator,'require_mps',synthetic_native_rejection)
    with pytest.raises(ValueError): boundary.verify_runtime_policy(policy,{'runtime_policy_hash':digest(policy)})
    assert called==[True]

def test_protected_boundaries_static_source_runtime_order():
    assert inspect.getsource(a.fit_native).index('verify_boundary(')<inspect.getsource(a.fit_native).index('LocalTransition(')
    assert inspect.getsource(a.predict_task).index('verify_boundary(')<inspect.getsource(a.predict_task).index('Predictor(')
    assert 'verify_boundary(' in inspect.getsource(e.score_released)

def audit_fixture():
    metric={'independent_tasks':36,'exact_correct':36,'exact':1.,'full_trace_exact':1.,
            'final_answer_exact':1.,'intermediate_accuracy':1.,'intermediate_correct':36,'intermediate_denominator':36}
    overall={**metric,'independent_tasks':216,'exact_correct':216,'intermediate_correct':216,'intermediate_denominator':216}
    metrics={condition:{str(seed):{'overall':copy.deepcopy(overall),**{family:copy.deepcopy(metric) for family in e.FAMILIES}} for seed in e.SEEDS} for condition in e.CONDITIONS}
    rows=[{'identity':digest(('synthetic-audit-only',i)),'condition':condition,'seed':seed} for condition in e.SEALED_CONDITIONS for seed in e.SEEDS for i in range(216)]
    comparisons={baseline+':'+family:{'independent_tasks':216 if family=='overall' else 36,'mean_gain':.2,'paired_task_bootstrap95':[.1,.3]} for baseline in ('feedforward','memorization','transition_lookup') for family in ('overall',*e.FAMILIES)}
    return {'gates':dict.fromkeys(e.GATE_KEYS,True),'scientific_status':'GATES_PASS_PENDING_INDEPENDENT_AUDIT',
            'unique_tasks':216,'metrics':metrics,'comparisons':comparisons,'raw_rows':rows,'raw_prediction_hash':digest(rows)}

@pytest.mark.parametrize('corruption',['empty','partial','truthy','wrongstatus','wrongcount','missingcondition','missingseed','missingfamily','denominator','nonfinite','comparisons','rawhash','raworder'])
def test_audit_report_cannot_vacuously_pass(corruption):
    report=audit_fixture()
    if corruption=='empty': report['gates']={}
    elif corruption=='partial': report['gates'].pop('bootstrap')
    elif corruption=='truthy': report['gates']['bootstrap']=1
    elif corruption=='wrongstatus': report['scientific_status']='INDEPENDENT_SCIENTIFIC_PASS'
    elif corruption=='wrongcount': report['unique_tasks']=215
    elif corruption=='missingcondition': report['metrics'].pop('procedural')
    elif corruption=='missingseed': report['metrics']['recurrent'].pop('47')
    elif corruption=='missingfamily': report['metrics']['recurrent']['11'].pop('carry_chains')
    elif corruption=='denominator': report['metrics']['recurrent']['11']['carry_chains']['independent_tasks']=35
    elif corruption=='nonfinite': report['metrics']['recurrent']['11']['overall']['wilson95']=[0.,float('nan')]
    elif corruption=='comparisons': report['comparisons']={}
    elif corruption=='rawhash': report['raw_prediction_hash']=digest('corrupted')
    elif corruption=='raworder':
        report['raw_rows'].reverse(); report['raw_prediction_hash']=digest(report['raw_rows'])
    with pytest.raises(ValueError): e.validate_audit_report(report)

def test_audit_verdict_requires_complete_bool_gates_with_external_boundary(monkeypatch):
    # Synthetic schema-only diagnostic; monkeypatch cannot create a production
    # receipt, and this test does not invoke protected data/optimizer/native work.
    calls=[]
    monkeypatch.setattr(e,'verify_signed',lambda *args:calls.append('synthetic signature check'))
    import mindscape.alpha_ood.numerical_v2.boundary as boundary
    monkeypatch.setattr(boundary,'verify_boundary',lambda *args:calls.append('synthetic boundary check'))
    report=audit_fixture()
    result=e.independently_audited_verdict(report,{},None,{})
    assert result['scientific_status']=='INDEPENDENT_SCIENTIFIC_PASS' and len(calls)==2
    report['gates']['bootstrap']=False; report['scientific_status']='NO_GO_UNVERIFIED'
    result=e.independently_audited_verdict(report,{},None,{})
    assert result['scientific_status']=='INDEPENDENT_SCIENTIFIC_FAIL'


def test_corrupt_committed_labels_rejected_against_postrelease_reference(tmp_path):
    # Tiny fabricated training fixture only; no real OOD labels/data/predictions.
    tasks=tuple(Task(i+2,(('add',1),),'train',family) for i,family in enumerate(e.FAMILIES))
    labels={task.identity:[task.start+2] for task in tasks} # Deliberately wrong.
    b=binding(); b['label_commitment']=digest(labels)
    private,keys=trust(); receipts={p:signed(private,p,b) for p in (*c.REQUIRED,'independent_eligibility')}
    access=c.authorize('evaluate',b,receipts,keys)
    rows=[{'identity':task.identity,'condition':condition,'seed':seed,'result':None,'failure':{'type':'fabricated_fixture_failure'}} for condition in e.SEALED_CONDITIONS for seed in e.SEEDS for task in tasks]
    seal=e.seal_predictions(tasks,rows,b,tmp_path/'synthetic-wrong-labels.json')
    release_binding={**b,'prediction_hash':seal['prediction_hash'],'ordered_identity_hash':seal['ordered_identity_hash']}
    with pytest.raises(ValueError,match='executed procedural reference'):
        e._score_released(tasks=tasks,seal=seal,labels=labels,release_receipt=signed(private,'seal_before_label_release',release_binding,sequence=2),access=access,binding=b,trusted_keys=keys)
