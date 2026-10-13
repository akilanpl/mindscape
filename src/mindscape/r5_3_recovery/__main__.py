"""Read-only forensic recovery. Replays existing CPU scenes/actions; never calls a predictor."""
import hashlib,json,math,tarfile
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from mindscape.v3.physics_world.schema import Scene,Action,digest
from mindscape.v3.physics_world.environment import trajectory,Environment,ENGINE_VERSION,ENGINE_REVISION
from mindscape.r5_3_physics.science import metrics,selection,analyze
from .numerics import native_bounds,score_tolerance,acceptable
from .gates import independent

ROOT=Path('results/r5_3_physics/native-one-campaign')
OUT=Path('experiments/r5_3_physics/recovery_v1')
FIELDS=('states','covariance','lower','upper','contacts')
def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def inventory(root):return {str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}
def require(ok,reason):
    if not ok:raise ValueError(reason)
def save(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def compare_tree(a,b):
    if isinstance(a,dict):
        require(isinstance(b,dict) and a.keys()==b.keys(),'Independent analysis keys')
        for k in a:compare_tree(a[k],b[k])
    elif isinstance(a,list):
        require(isinstance(b,list) and len(a)==len(b),'Independent analysis list')
        for x,y in zip(a,b,strict=True):compare_tree(x,y)
    elif isinstance(a,float):require(math.isclose(a,b,rel_tol=1e-14,abs_tol=1e-14),'Independent aggregate numerical mismatch')
    else:require(type(a) is type(b) and a==b,'Independent discrete/gate mismatch')

def validate_prediction(pred,horizon):
    expected={'version','states','covariance','lower','upper','contacts','private_queries'}
    require(set(pred)==expected and pred['version']=='mindscape.physics.transition/5.3.1' and pred['private_queries']==0,'Prediction schema')
    for field,shape in [('states',(horizon,3,4)),('lower',(horizon,3,4)),('upper',(horizon,3,4)),('covariance',(horizon,12,12)),('contacts',(horizon,3))]:
        x=np.asarray(pred[field]);require(x.shape==shape and np.isfinite(x).all(),'Prediction shape/finite')
    cov=np.asarray(pred['covariance']);require(np.allclose(cov,cov.transpose(0,2,1),rtol=0,atol=1e-12),'Covariance symmetry')
    require(np.min(np.linalg.eigvalsh(cov))>=-1e-5,'Covariance PSD')
    require(np.all(np.asarray(pred['lower'])<=np.asarray(pred['upper'])),'Interval bounds')
    require(np.all((np.asarray(pred['contacts'])>=0)&(np.asarray(pred['contacts'])<=1)),'Contact probability')

def execute():
    OUT.mkdir(parents=True,exist_ok=True)
    original=inventory(ROOT);external={str(p):sha(p) for p in (ROOT.with_suffix('.preservation.json'),ROOT.with_suffix('.tar.gz'))}
    receipt=load(ROOT.with_suffix('.preservation.json'))
    require(original==receipt['files'] and sha(ROOT.with_suffix('.tar.gz'))==receipt['archive_sha256'],'Original preservation hashes')
    with tarfile.open(ROOT.with_suffix('.tar.gz')) as tar:
        archive={m.name:hashlib.sha256(tar.extractfile(m).read()).hexdigest() for m in tar.getmembers()}
    require(archive==original,'Archive restoration hashes')
    save('original-inventory.json',dict(files=original,external_files=external,archive_restore='PASS'))
    freeze=load(ROOT/'freeze.json')
    for group in freeze.values():
        for name,value in group.items():require(sha(name)==value,'Frozen source/history/checkpoint changed: '+name)
    require(load(ROOT/'summary.json')['decision']=='FAILED_CAMPAIGN' and load(ROOT/'failure.json')=={'type':'ValueError','message':'Recomputed forecast metric'},'Historical failure receipt')
    manifest=load(ROOT/'manifest.json');protocol=manifest['protocol'];scene_records=manifest['records']
    require(load('configs/r5_3_physics/scenes.json')==manifest and digest(scene_records)==manifest['scene_hash'] and digest(protocol)==manifest['protocol_hash'],'Frozen manifest')
    require(len(scene_records)==112 and len({r['identity'] for r in scene_records})==112,'Unique full scene coverage')
    scenes={}
    for r in scene_records:
        s=Scene.from_record(r['scene']);require(s.identity==r['identity'],'Canonical scene identity')
        if s.split in ('iid','ood'):scenes[s.identity]=s
    require(len(scenes)==52 and Counter((s.family,s.split) for s in scenes.values())=={('motion','iid'):12,('motion','ood'):14,('interaction','iid'):12,('interaction','ood'):14},'Scene panel denominators')
    rows={};targets={}
    for i,s in scenes.items():
        first=trajectory(s);second=trajectory(s)
        for k in ('states','observations','masks','actions','contexts'):require(np.array_equal(first[k],second[k]),'Nondeterministic target replay')
        rows[i]=first;targets[i]={k:hashlib.sha256(first[k].tobytes()).hexdigest() for k in ('states','observations','masks','actions','contexts')}
    save('target-replay.json',dict(status='PASS',engine=ENGINE_VERSION,engine_revision=ENGINE_REVISION,replays_per_existing_scene=2,target_digests=targets,new_scene_identities=0))
    # Training receipt counts are independently checked; baseline forecast is explicitly not regenerated.
    contact=spring=0
    training=[r for r in scene_records if r['scene']['split']=='train']
    for r in training:
        s=Scene.from_record(r['scene']);row=trajectory(s);contact+=int(row['states'][:,:,8].sum());spring+=bool(s.spring)
    qualification=load(ROOT/'qualification.json');q=protocol['qualification']
    require(qualification['decision']=='DATA_READY' and qualification['training_scenes']==len(training)==48 and qualification['contact_frames']==contact==175 and qualification['spring_scenes']==spring==12,'Qualification discrete receipt')
    require(math.isfinite(qualification['baseline_mse']) and qualification['baseline_mse']>=q['minimum_baseline_mse'],'Recorded qualification headroom')
    require(load(ROOT/'engineering.json')=={'status':'PASS','device':'mps','multistep_gradients':True,'internal_pair_momentum':True,'permutation_without_indexed_spring':True,'optimizer_steps':0},'Engineering receipt')
    index=load(ROOT/'predictions-index.json');plans=load(ROOT/'planning-index.json');records=load(ROOT/'metrics.json')
    expected={(i,k,s) for i in scenes for k in protocol['kinds'] for s in protocol['optimizer_seeds']}|{(i,k,0) for i in scenes for k in protocol['baseline_methods'] if k not in ('matched','v3_1_frozen')}|{(i,'v3_1_frozen',s) for i in scenes for s in (11,23)}
    require(len(index)==len(plans)==len(expected)==624 and {(r['identity'],r['method'],r['seed']) for r in index}==expected,'Forecast/planning coverage')
    lookup={(r['identity'],r['method'],r['seed'],r['horizon']):r for r in records}
    require(len(records)==len(lookup)==2496 and set(lookup)=={(*v,h) for v in expected for h in protocol['horizons']},'Complete unique metric cells')
    discrepancies=[];canonical=[];tolerance_max=0;discrete_checks=0
    for entry,plan in zip(index,plans,strict=True):
        i=entry['identity'];method=entry['method'];seed=entry['seed'];row=rows[i];scene=scenes[i]
        require(entry['file']==f'{i}-{method}-{seed}.json' and plan['file']==f'{i}-{method}-{seed}-planning.json' and plan['identity']==i,'Bound file identity')
        require(sha(ROOT/entry['file'])==entry['sha256'] and sha(ROOT/plan['file'])==plan['sha256'],'Prediction/planning indexed hash')
        pred=load(ROOT/entry['file']);savedplan=load(ROOT/plan['file']);validate_prediction(pred,64)
        require(len(savedplan['predictions'])==9 and savedplan['environment_queries_during_selection']==0,'Planning candidate count/query receipt')
        for candidate in savedplan['predictions']:validate_prediction(candidate,32)
        scores=selection(savedplan['predictions'],np.array([1.5,1.]))
        require(scores==savedplan['scores'] and int(np.argmin(scores))==savedplan['selected'],'Exact selected risk/action')
        choice=[(x,y) for x in (-2,0,2) for y in (-2,0,2)][savedplan['selected']];actions=np.zeros((32,3,4),np.float32);actions[0,0,2:]=choice
        require(np.array_equal(actions,np.asarray(savedplan['action'])) and np.asarray(savedplan['actual']).shape==(32,3,10),'Action/trace event count')
        env=Environment(scene)
        for action in row['actions'][:3]:env.step(Action.from_vector(action))
        actual=np.stack([env.step(Action.from_vector(action)) for action in actions])
        require(np.array_equal(actual,np.asarray(savedplan['actual'])),'Exact independent planning replay')
        success=bool(np.linalg.norm(actual[-1,0,:2]-[1.5,1.])<=.75)
        require(success==savedplan['success'] and type(savedplan['success']) is bool,'Exact planning outcome')
        for h in protocol['horizons']:
            originalcell=lookup[i,method,seed,h]
            require(originalcell['family']==scene.family and originalcell['split']==scene.split and originalcell['stratum']==scene.ood_condition and originalcell['success']==success and type(originalcell['success']) is bool,'Exact metric identity/outcome')
            sliced={k:np.asarray(pred[k])[:h] for k in FIELDS};truth=row['states'][4:4+h];active=row['observations'][3,:,9];radius=row['observations'][3,:,5]
            recomputed=metrics(sliced,truth,active,radius);native=dict(sliced);learned=method in ('structured','matched')
            if learned:native['lower'],native['upper']=native_bounds(sliced['lower'],sliced['upper'])
            native_metric=metrics(native,truth,active,radius)
            require(all(native_metric[k]==originalcell[k] for k in recomputed),'Metric inconsistent with native numerical path')
            if learned:
                tol=score_tolerance(truth[:,active>0,:4],sliced['lower'][:,active>0],sliced['upper'][:,active>0]);tolerance_max=max(tolerance_max,tol['atol'])
                require(acceptable(originalcell['interval_score'],recomputed['interval_score'],tol),'Interval score exceeds IEEE tolerance')
            else:tol={'atol':0,'rtol':0}
            for key,value in recomputed.items():
                require(math.isfinite(value) and math.isfinite(originalcell[key]),'Nonfinite metric')
                if key!='interval_score':require(value==originalcell[key],'Non-interval metric discrepancy')
                if value!=originalcell[key]:
                    require(key=='interval_score' and learned,'Unexpected numerical discrepancy')
                    lower=sliced['lower'][:,active>0];upper=sliced['upper'][:,active>0]
                    width32=upper.astype(np.float32)-lower.astype(np.float32)
                    expected_difference=float(np.mean(width32.astype(np.float64)-(upper-lower)))
                    difference=originalcell[key]-value
                    require(abs(difference-expected_difference)<=64*np.finfo(float).eps*max(1,abs(value)),'Difference not explained by width rounding')
                    discrepancies.append(dict(identity=i,method=method,seed=seed,horizon=h,metric=key,saved=originalcell[key],recomputed=value,absolute_difference=abs(difference),relative_difference=abs(difference)/max(abs(value),abs(originalcell[key]),1e-300),signed_difference=difference,width_rounding_difference=expected_difference,native_dtype_reproduction_exact=True,tolerance=tol,classification='NUMERICAL_PATH_FLOAT32_WIDTH_TO_FLOAT64_JSON_REPLAY'))
            canonical.append({**originalcell,**recomputed});discrete_checks+=1
    save('diagnostic.json',dict(first=discrepancies[0] if discrepancies else None,metric_cells=2496,metric_comparisons=19968,mismatches=len(discrepancies),by_metric={k:{'count':len(v),'max_abs':max(r['absolute_difference'] for r in v),'max_rel':max(r['relative_difference'] for r in v),'methods':dict(Counter(r['method'] for r in v))} for k,v in ((key,[r for r in discrepancies if r['metric']==key]) for key in sorted({r['metric'] for r in discrepancies}))},details=discrepancies,native_dtype='float32 NumPy quantiles and upper-lower subtraction',json_reload_dtype='float64 arrays; float32 values are losslessly representable',numpy_version=np.__version__,max_theoretical_absolute_tolerance=tolerance_max,relative_tolerance=64*np.finfo(float).eps))
    import torch
    fits=load(ROOT/'fits.json');require(len(fits)==6 and {(r['kind'],r['seed']) for r in fits}=={(k,s) for k in protocol['kinds'] for s in protocol['optimizer_seeds']},'Exact six-fit coverage')
    checkpoint_receipts=[]
    for fit in fits:
        path=ROOT/f"{fit['kind']}-s{fit['seed']}.pt";require(sha(path)==fit['sha256'],'Checkpoint indexed hash')
        saved=torch.load(path,map_location='cpu',weights_only=True)
        require(saved['kind']==fit['kind'] and saved['seed']==fit['seed'] and saved['updates']==fit['updates']==160 and saved['curve']==fit['curve'] and len(saved['curve'])==160 and all(math.isfinite(v) for v in saved['curve']),'Checkpoint identity/curve/update receipt')
        require(all(torch.isfinite(v).all() for v in saved['state'].values()) and sum(v.numel() for v in saved['state'].values())==fit['parameters'],'Checkpoint finite/count')
        state=saved['optimizer']['state'];params=[i for group in saved['optimizer']['param_groups'] for i in group['params']]
        require(set(params)==set(state) and len(params)==len(saved['state']),'Optimizer parameter coverage')
        require(all(v['step'].item()==160 and torch.isfinite(v['exp_avg']).all() and torch.isfinite(v['exp_avg_sq']).all() for v in state.values()),'Optimizer exact steps/finite moments')
        checkpoint_receipts.append(dict(file=path.name,sha256=sha(path),updates=160,optimizer_parameters=len(params),parameters=fit['parameters']))
    original_analysis=analyze(records,protocol);recovered,gatevalues,perseed=independent(records,protocol);compare_tree(recovered,original_analysis)
    canonical_analysis=analyze(canonical,protocol)
    require(canonical_analysis['gates']==original_analysis['gates'] and canonical_analysis['decision']==original_analysis['decision'],'Numerical path changed scientific gates')
    recovered.update(provenance='POST-HOC RECOVERY FROM FROZEN EVIDENCE',historical_decision='FAILED_CAMPAIGN',additional_neural_forwards=0,additional_optimizer_steps=0)
    require(inventory(ROOT)==original and all(sha(p)==h for p,h in external.items()),'Original evidence mutated')
    save('summary.json',recovered);save('gate-values.json',gatevalues);save('per-seed.json',perseed)
    save('audit.json',dict(status='PASS',original_files=len(original),original_bytes_unchanged=True,forecasts=624,planning_executions=624,metric_cells=2496,metric_comparisons=19968,numerical_discrepancies=len(discrepancies),native_dtype_metrics_exact=2496,independent_scenes=52,independent_ood_scenes=28,independent_iid_scenes=24,checkpoints=checkpoint_receipts,optimizer_updates=960,neural_forwards=0,new_scene_identities=0,qualification=qualification,qualification_baseline_mse='Authenticated original receipt only; no baseline forecast regenerated',limitations=['Archived forecasts are not regenerated from weights; correspondence relies on hashed native provenance.','Forecast targets were not separately archived originally; repeated deterministic Pymunk reconstruction from authenticated scene manifests is exact and recorded here.','32 uncertainty particles and14 OOD scenes per family limit tail/coverage and bootstrap precision.']))
    print(json.dumps({'decision':recovered['decision'],'gates':recovered['gates'],'first_mismatch':discrepancies[0],'discrepancies':len(discrepancies)}))

if __name__=='__main__':
    try:execute()
    except Exception as exc:
        OUT.mkdir(parents=True,exist_ok=True);save('failure.json',dict(decision='INTEGRITY_FAILURE',error=str(exc),historical_decision='FAILED_CAMPAIGN'))
        raise
