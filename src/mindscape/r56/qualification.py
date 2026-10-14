"""Nontraining scientific qualification, isolated splits and proposed workload.

No optimizer construction, authorization signing, campaign execution or checkpoint
selection occurs here. A package is a proposal, never scientific permission.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import random
import time
from .curriculum import BUDGETS, SEEDS, digest, prepare_data, exposure_summary

CONTROLS = ('matched-mlp', 'matched-rnn', 'matched-gru', 'graph-concept-on',
            'graph-concept-off', 'graph-concept-shuffled', 'graph-concept-ablated',
            'graph-memory-recurrent')


def workload():
    rows = [{'seed': seed, 'budget': budget, 'control': control, 'optimizer_updates': 2000,
             'batch_size': 1, 'checkpoint_selection': 'fixed final; never held-out selection'}
            for seed in SEEDS for budget in BUDGETS for control in CONTROLS]
    assert len(rows) == 72 and sum(row['optimizer_updates'] for row in rows) == 144000
    return rows


def feature_identity(problem):
    """Digest actual encoded observations, not split names or unencoded instructions."""
    from .native_domains import pack_problem
    local = dict(problem)
    if local.get('domain') == 'math':
        local.setdefault('concept_ids', [0] if local['operation']=='add_digit' else [1])
    graph = pack_problem(local)
    return digest({name: getattr(graph, name).tolist() for name in
                   ('node_features','node_types','edges','relation_types','action','concept_ids')} | {'domain': graph.domain})


def heldout_suite(training_rows):
    """Real disjoint observations, held contexts and longer original rollouts.

    References are separately sealed and never passed to a predictor. Novel coding
    source structures still measure candidate selection, not synthesis correctness.
    """
    observed_math = {(r['problem']['operation'], r['problem']['left'], r['problem']['right'],
                      r['problem'].get('carry_in',0)) for r in training_rows if r['problem']['domain']=='math'}
    problems = []
    for operation in ('add_digit','borrow_digit'):
        for left in range(10):
            for right in range(10):
                for incoming in (0,1):
                    if (operation,left,right,incoming) not in observed_math:
                        problems.append({'axis':'unseen-digit-context','problem':{'domain':'math','operation':operation,
                            'left':left,'right':right,'incoming':incoming,'concept_ids':[0 if operation=='add_digit' else 1]}})
    for index,source in enumerate(('def f(x):\n return x * x + 3',
                                 'def f(x):\n y = x - 2\n return y * x',
                                 'def f(x):\n return (x + 1) * (x - 1)',
                                 'x = 3\ny = x + 1\nz = y * x')):
        for candidates in (['alpha','beta','gamma','delta'], ['delta','gamma','beta','alpha']):
            problems.append({'axis':'novel-coding-structure-candidate-permutation', 'problem':{
                'domain':'coding','operation':'rename','source':source,'old_name':'x',
                'requested_name':'gamma','candidates':list(candidates),'concept_ids':[3]}})
    for mass in (.25,20.,100.):
        for force in ([20.,-30.],[-50.,40.]):
            for dt in (.05,2.):
                problems.append({'axis':'ood-physics', 'problem':{'domain':'physics','operation':'force',
                    'mass':mass,'force':force,'dt':dt,'velocity':[12.,-13.],'concept_ids':[2]}})
    references = {}
    for row in problems:
        problem = row['problem']; row['identity'] = digest(problem)
        domain = problem['domain']
        if domain=='math':
            value=problem['left']+problem['right']+problem['incoming'] if problem['operation']=='add_digit' else problem['left']-problem['right']-problem['incoming']
            target={'digit':value%10,'outgoing':value//10 if problem['operation']=='add_digit' else int(value<0)}
        elif domain=='coding': target={'candidate_index':problem['candidates'].index(problem['requested_name'])}
        else: target={'delta':[f*problem['dt']/problem['mass'] for f in problem['force']]}
        references[row['identity']] = target
    compositions=[]
    for length in (2,4,8,16,32):
        for kind in ('carry-chain','alternating-digits'):
            left='9'*length if kind=='carry-chain' else ('27'*length)[:length]
            right='1' if kind=='carry-chain' else ('81'*length)[:length]
            problem={'domain':'math','operation':'add','left':left,'right':right,'concept_ids':[0]}
            identity=digest(problem); compositions.append({'axis':kind,'depth':length,'identity':identity,'problem':problem})
            references[identity]={'answer':str(int(left)+int(right))}
    return {'single_transitions':problems,'compositions':compositions}, references


def overlap_audit(data):
    """Check actual graph tensors and expose duplicate training samples."""
    encoded={split:{feature_identity(row['problem']) for row in rows} for split,rows in data.items()}
    return {'encoded_train_dev_overlap':len(encoded['train'] & encoded['dev']),
            'encoded_train_test_overlap':len(encoded['train'] & encoded['test']),
            'encoded_dev_test_overlap':len(encoded['dev'] & encoded['test']),
            'exposures':{str(b):exposure_summary(data['train'][:b]) for b in BUDGETS},
            'optimizer_sample_exposures':{str(b):update_exposures(data['train'],b) for b in BUDGETS},
            'note':'Split-name and metadata identity disjointness alone is insufficient.'}


def language_audit():
    from .language_learning import build_language_dataset, training_vocabulary
    data=build_language_dataset();vocab=training_vocabulary(data)
    result={}
    train_args={tuple(sorted(r['spans'])) for r in data['train']}
    train_text={r['text'] for r in data['train']}
    for split,rows in data.items():
        result[split]={'count':len(rows),'surface_overlap_with_train':sum(r['text'] in train_text for r in rows),
                      'unknown_token_count':sum(vocab.encode(r['text'])['unknown_tokens'] for r in rows),
                      'slot_schema_seen_in_train':sum(tuple(sorted(r['spans'])) in train_args for r in rows),
                      'interpretation':{'train':'supervised templates','paraphrase':'same operand families; wording shift',
                        'structure':'reordered template and operand composition','vocabulary':'unknown words and identifier shift'}[split]}
    return {'splits':result,'semantic_generalization_qualified':False,
            'limitations':['Only 18 training templates/examples; not open-language semantics.',
                           'Null-span totals can inflate accuracy; report present slots separately.',
                           'Identifier/operand lexical copying is not evidence of task reasoning.',
                           'No lexical fallback from held-out vocabulary is allowed.']}


def compute_probe(repetitions=3):
    """Measured CPU forward+backward diagnostics; never creates an optimizer.

    Times include representation conversion consistently. Results are engineering
    estimates, not 2,000-update campaign timing or training comparisons.
    """
    import torch
    from .neural import R56Model
    from .native_domains import pack_problem
    from .benchmarks import matched_baseline, model_accounting, controlled_graph
    torch.set_num_threads(1)
    rows=prepare_data(17)['train'][:3]
    results=[]
    for control in CONTROLS:
        with torch.random.fork_rng():
            torch.manual_seed(17)
            model=matched_baseline(control.removeprefix('matched-')) if control.startswith('matched-') else R56Model()
        before={k:v.clone() for k,v in model.state_dict().items()}; timings=[]
        per_domain={}
        for repeat in range(repetitions+1):
            for row in rows:
                model.zero_grad(set_to_none=True)
                began=time.perf_counter()
                graph=controlled_graph(pack_problem(row['problem']),control)
                output=model(graph,stateless=control!='graph-memory-recurrent')
                # Scalar diagnostic uses all heads; no task labels or optimizer.
                loss=sum(output[key].square().mean() for key in
                         ('math_digit_logits','math_carry_logits','coding_logits','physics_delta'))
                loss.backward()
                elapsed=time.perf_counter()-began
                if repeat:
                    timings.append(elapsed);per_domain.setdefault(row['problem']['domain'],[]).append(elapsed)
        assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
        results.append({'control':control,**model_accounting(model),
            'forward_backward_seconds':timings,'domain_mean_seconds':{k:sum(v)/len(v) for k,v in per_domain.items()},
            'projected_2000_diagnostic_pass_seconds':2000*sum(timings)/len(timings),
            'optimizer_updates':0,'weights_unchanged':True})
    return {'torch':torch.__version__,'threads':1,'warmup_passes':1,'repetitions':repetitions,
            'results':results,'compute_matched_claim':False,
            'warning':'Diagnostic backward, not loss-matched training timing; no optimizer state/steps. CPU latency is environment-dependent.'}


def prepare(root, destination, measure=True):
    from .learning import source_locks
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    index={};audits={}
    for seed in SEEDS:
        data=prepare_data(seed);audits[str(seed)]=overlap_audit(data)
        # Training-only JSON: held-out labels are never deserialized by trainer.
        train_name=f'train-{seed}.json'; raw=json.dumps(data['train'],sort_keys=True).encode()
        (destination/train_name).write_bytes(raw); index[str(seed)]={'training_file':train_name,'sha256':hashlib.sha256(raw).hexdigest()}
        public,targets=heldout_suite(data['train'])
        for label,value in [('problems',public),('targets',targets),('iid-dev',data['dev']),('iid-test',data['test'])]:
            raw=json.dumps(value,sort_keys=True).encode();name=f'heldout-{label}-{seed}.json'
            (destination/name).write_bytes(raw);index[str(seed)][label]={'file':name,'sha256':hashlib.sha256(raw).hexdigest()}
    manifest={'version':'r56-scientific-qualification-v2','source_locks':source_locks(root),
              'index':index,'workload':workload(),'neural_run_count':72,'total_optimizer_steps':144000,
              'heldout_policy':'separate evaluator process required before GO; never checkpoint selection; final checkpoint only',
              'optimizer':'Adam','learning_rate':.001,'optimizer_steps_per_fit':2000,'batch_size':1,
              'model_config':{'width':48,'message_steps':2},'capacity_target':44962,
              'candidate_thresholds':{'classification_target_accuracy':.90,'guided_accuracy_advantage_pp':5,
                 'paired_ci95_lower_bound_gt':0,'lower_budgets':[32,128],'physics_absolute_error_tolerance':.1,
                 'physics_mse_target':.01,'holm_alpha':.05},
              'statistics':{'bootstrap_replicates':2000,'cluster':'independent problem family; seeds aggregated',
                 'selection':'fixed final checkpoint','holm_family':'predeclare domain/control/budget comparisons before GO'},
              'scientific_decision':'NO_GO_PENDING_INDEPENDENT_QUALIFICATION',
              'authorization':'UNISSUED','independent_qualification':'UNISSUED',
              'outstanding':['concept guidance remains redundant with operation/domain observations; per-observation random control requires review','action head has no supervised selection loss',
                 'novel coding candidate selection is not transformation synthesis',
                 'memory benefit unidentifiable on independent one-step training examples',
                 'cross-domain learned transfer requires separately reviewed exposure schedule',
                 'external trust keys/qualification/authorization absent']}
    manifest['package_sha256']=digest(manifest)
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2))
    receipts={'split_audits':audits,'language':language_audit(),'compute':compute_probe() if measure else {'unrun':True},
              'shared_parameters':shared_gradient_audit() if measure else {'unrun':True},
              'optimizer_updates_executed':0}
    (destination/'qualification-receipts.json').write_text(json.dumps(receipts,indent=2))
    (destination/'workload.json').write_text(json.dumps(workload(),indent=2))
    for role in ('authorization','qualification'):
        (destination/f'{role}.UNISSUED.json').write_text(json.dumps({'role':role,'issued':False,
            'package_sha256':manifest['package_sha256'],'signature_hex':'','scope':'r56-scientific-qualification-v2'},indent=2))
    return manifest


def validate_scientific_gate(root, package, authorization, qualification, trusted_keys):
    """Validate whole proposal before exposing even training labels to fitting code."""
    from .learning import source_locks
    package=Path(package);manifest=json.loads((package/'manifest.json').read_text())
    if not isinstance(manifest,dict):raise PermissionError('Scientific proposal must be an object')
    payload=dict(manifest);claimed=payload.pop('package_sha256',None)
    if manifest.get('version')!='r56-scientific-qualification-v2' or claimed!=digest(payload):
        raise PermissionError('Scientific proposal identity mismatch')
    if manifest['source_locks']!=source_locks(root):
        raise PermissionError('Scientific source lock mismatch')
    if manifest['workload']!=workload() or manifest['neural_run_count']!=72 or manifest['total_optimizer_steps']!=144000:
        raise PermissionError('Frozen workload changed')
    # NO_GO cannot be promoted by supplying a signature or editing a flag alone:
    # an independently reviewed replacement proposal must be frozen with GO.
    if manifest.get('scientific_decision')!='GO_INDEPENDENTLY_REVIEWED':
        raise PermissionError('Scientific NO_GO: unresolved defects; no optimizer authorized')
    if not isinstance(trusted_keys,dict) or not trusted_keys:
        raise PermissionError('External trust configuration required')
    issuers=[]
    for artifact,role in ((authorization,'authorization'),(qualification,'qualification')):
        issuer=artifact.get('issuer');signed=dict(artifact);signature=signed.pop('signature_hex','')
        if artifact.get('role')!=role or artifact.get('issued') is not True or artifact.get('package_sha256')!=claimed or artifact.get('scope')!='r56-scientific-qualification-v2' or issuer not in trusted_keys:
            raise PermissionError('External scientific signatures required')
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
            Ed25519PublicKey.from_public_bytes(bytes.fromhex(trusted_keys[issuer])).verify(bytes.fromhex(signature),json.dumps(signed,sort_keys=True,separators=(',',':')).encode())
        except Exception as exc:
            raise PermissionError('Scientific signature verification failed') from exc
        issuers.append(issuer)
    if issuers[0]==issuers[1]:
        raise PermissionError('Independent qualification issuer required')
    # Hash bytes only; held-out label JSON is never decoded by the trainer.
    for entry in manifest['index'].values():
        for item in [dict(file=entry['training_file'],sha256=entry['sha256'])]+[entry[k] for k in ('problems','targets','iid-dev','iid-test')]:
            name=item['file']
            if not isinstance(name,str) or Path(name).name!=name:
                raise PermissionError('Invalid package file path')
            if hashlib.sha256((package/name).read_bytes()).hexdigest()!=item['sha256']:
                raise PermissionError('Scientific data lock mismatch')
    return manifest


def load_training_only(package, manifest, seed):
    """Call only after gate: decode one training file; never open held-out labels."""
    entry=manifest['index'][str(seed)];name=entry['training_file']
    if Path(name).name!=name:raise ValueError('Invalid training path')
    raw=(Path(package)/name).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=entry['sha256']:raise ValueError('Training file changed')
    rows=json.loads(raw)
    if not isinstance(rows,list) or len(rows)!=512 or any(r.get('split')!='train' for r in rows):
        raise ValueError('Training-only file contains unexpected rows')
    return rows


def shared_gradient_audit():
    """Prove same trainable tensors receive domain losses, not learned transfer."""
    import torch
    from .neural import R56Model
    from .native_domains import pack_problem
    with torch.random.fork_rng():
        torch.manual_seed(17);model=R56Model()
    before={k:v.clone() for k,v in model.state_dict().items()};active={}
    for row in prepare_data(17)['train'][:3]:
        model.zero_grad(set_to_none=True);out=model(pack_problem(row['problem']),stateless=True)
        domain=row['problem']['domain'];labels=row['labels']
        if domain=='math':
            loss=torch.nn.functional.cross_entropy(out['math_digit_logits'][None],torch.tensor([labels['math_digit']]))+torch.nn.functional.cross_entropy(out['math_carry_logits'][None],torch.tensor([labels['math_carry']]))
        elif domain=='coding':loss=torch.nn.functional.cross_entropy(out['coding_logits'][:len(row['problem']['candidates'])][None],torch.tensor([labels['coding_index']]))
        else:loss=torch.nn.functional.mse_loss(out['physics_delta'],torch.tensor(labels['physics_delta']))
        loss.backward()
        active[domain]=sorted(name for name,p in model.named_parameters() if p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0)
    intersection=sorted(set.intersection(*(set(v) for v in active.values())))
    assert 'message.0.weight' in intersection and 'features.weight' in intersection
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
    return {'nonzero_gradient_parameters_by_domain':active,'shared_gradient_parameter_names':intersection,
            'optimizer_updates':0,'weights_unchanged':True,'transfer_demonstrated':False}


def update_exposures(rows,budget,updates=2000):
    selected=rows[:budget];sequence=[selected[i%len(selected)] for i in range(updates)]
    summary=exposure_summary(sequence)
    summary['per_sample_update_counts']={str(i):updates//budget+int(i<updates%budget) for i in range(budget)}
    return summary


def paired_report(left_records, right_records):
    """Audit existing original outcome records; never infer correctness from prose.

    All seeds of the same independent problem family form one resampling cluster.
    Exact accuracy and physics errors must be reported separately by caller.
    """
    from .benchmarks import paired_bootstrap
    def index(rows):
        result={}
        for row in rows:
            required=('identity','seed','family','metric','metric_name','repaired')
            if any(k not in row for k in required) or row['repaired'] is not False or 'original_prediction' not in row:
                raise ValueError('Original immutable prediction/metric provenance required')
            key=(row['identity'],row['seed'])
            if key in result:raise ValueError('Duplicate prediction identity/seed')
            result[key]=row
        return result
    left,right=index(left_records),index(right_records)
    if not left or set(left)!=set(right):raise ValueError('Paired evaluation identities/seeds differ')
    groups={};metric_names=set()
    for key,a in left.items():
        b=right[key]
        if a['family']!=b['family'] or a['metric_name']!=b['metric_name']:
            raise ValueError('Paired family/metric mismatch')
        metric_names.add(a['metric_name']);groups.setdefault(a['family'],[]).append((a['metric'],b['metric']))
    if len(metric_names)!=1:raise ValueError('Do not pool classification accuracy and physical error')
    a=[sum(x for x,_ in rows)/len(rows) for rows in groups.values()]
    b=[sum(y for _,y in rows)/len(rows) for rows in groups.values()]
    return {**paired_bootstrap(a,b),'independent_family_clusters':len(groups),
            'paired_predictions':len(left),'metric_name':next(iter(metric_names)),
            'scientific_success':False,'qualification':'Requires independent result audit and predeclared multiple-comparison family'}
