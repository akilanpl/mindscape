"""One fail-closed native campaign. Inference only receives public request copies."""
import hashlib,json,subprocess,time,tarfile
from pathlib import Path
import numpy as np
from mindscape.v3.physics_world.schema import Scene,Action,digest
from mindscape.v3.physics_world.environment import Environment,trajectory
from .public import request
from . import baselines,legacy
from .predict import forecast
from .science import metrics,selection,analyze
ROOT=Path('results/r5_3_physics/native-one-campaign')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    def convert(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        raise TypeError(type(x).__name__)
    p.write_text(json.dumps(v,default=convert,allow_nan=False)+'\n')
def load(p):return json.loads(Path(p).read_text())
def verify_freeze():
    f=load('configs/r5_3_physics/freeze.json')
    for group in f.values():
        for name,h in group.items():
            if sha(name)!=h:raise ValueError('Freeze changed: '+name)
    for name in ('protocol','scenes','freeze'):
        p=f'configs/r5_3_physics/{name}.json'
        if subprocess.check_output(['git','show','HEAD:'+p])!=Path(p).read_bytes():raise ValueError('Freeze not committed')
    return f

def main():
    ROOT.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    try:
        freeze=verify_freeze()
        import sys
        receipt=subprocess.check_output([sys.executable,'-m','unittest','discover','-s','tests/r5_3_physics','-v'],stderr=subprocess.STDOUT).decode()
        save(ROOT/'preflight-tests.json',dict(status='PASS',output=receipt,neural_execution=False))
        manifest=load('configs/r5_3_physics/scenes.json');p=manifest['protocol']
        if digest(manifest['records'])!=manifest['scene_hash'] or digest(p)!=manifest['protocol_hash']:raise ValueError('Scene manifest integrity')
        save(ROOT/'freeze.json',freeze);save(ROOT/'manifest.json',manifest)
        # Qualification never generates or inspects evaluation outcomes.
        training=[];contacts=springs=0;errors=[]
        for r in manifest['records']:
            if r['scene']['split']!='train':continue
            scene=Scene.from_record(r['scene']);row=trajectory(scene);training.append(row)
            contacts+=int(row['states'][:,:,8].sum());springs+=bool(scene.spring)
            public=request(row['observations'][:4],row['masks'][:4],row['actions'][3:35],row['contexts'][3:35])
            pred=baselines.forecast(public,'analytical_contacts');active=row['observations'][3,:,9]
            errors.append(metrics(pred,row['states'][4:36],active,row['observations'][3,:,5])['mse'])
        q=p['qualification'];ready=len(training)>=q['minimum_training_scenes'] and contacts>=q['minimum_contact_frames'] and springs>=q['minimum_spring_scenes'] and np.mean(errors)>=q['minimum_baseline_mse']
        save(ROOT/'qualification.json',dict(decision='DATA_READY' if ready else 'DATA_NO_GO',training_scenes=len(training),contact_frames=contacts,spring_scenes=springs,baseline_mse=float(np.mean(errors))))
        if not ready: save(ROOT/'summary.json',{'decision':'DATA_NO_GO'})
        else:
            from .training import fit
            from mindscape.v3.physics_world.accelerator import require_mps
            require_mps()
            # Execute differentiable multistep/gradient engineering checks before any optimizer step.
            from .engineering import check
            save(ROOT/'engineering.json',check())
            models={};fits=[]
            for seed in p['optimizer_seeds']:
                for kind in p['kinds']:
                    path=ROOT/f'{kind}-s{seed}.pt';model,receipt=fit(training,kind,seed,p,path);receipt['sha256']=sha(path);fits.append(receipt);models[kind,seed]=model
                    save(ROOT/'fits.json',fits);print(json.dumps({'fit':len(fits),'updates':receipt['updates']}),flush=True)
            old={(family,seed):legacy.load(next(name for name in freeze['legacy_checkpoints'] if f'/{family}/s{seed}/' in name)) for family in ('motion','interaction') for seed in (11,23)}
            records=[];predindex=[];planindex=[]
            for r in manifest['records']:
                scene=Scene.from_record(r['scene'])
                if scene.split not in ('iid','ood'):continue
                row=trajectory(scene);sid=scene.identity;active=row['observations'][3,:,9];radius=row['observations'][3,:,5]
                public=request(row['observations'][:4],row['masks'][:4],row['actions'][3:67],row['contexts'][3:67])
                methods=[(k,s) for k in p['kinds'] for s in p['optimizer_seeds']]+[(k,0) for k in p['baseline_methods'] if k not in ('matched','v3_1_frozen')]+[('v3_1_frozen',s) for s in (11,23)]
                for method,seed in methods:
                    def call(req):
                        if method in p['kinds']:return forecast(models[method,seed],req,seed=seed+int(sid[:6],16),particles=p['particles'])
                        if method=='v3_1_frozen':return legacy.forecast(old[scene.family,seed],req)
                        return baselines.forecast(req,method)
                    pred=call(public);pred.pop('samples',None)
                    filename=f'{sid}-{method}-{seed}.json';save(ROOT/filename,pred);predindex.append(dict(identity=sid,method=method,seed=seed,file=filename,sha256=sha(ROOT/filename)))
                    candidates=[];actions=[]
                    for x in (-2,0,2):
                        for y in (-2,0,2):
                            a=np.zeros((32,3,4),np.float32);a[0,0,2:]=[x,y];c=np.repeat(row['contexts'][3:4],32,axis=0);c[:,3]+=np.arange(32)/30
                            req=request(row['observations'][:4],row['masks'][:4],a,c);candidate=call(req);candidate.pop('samples',None);candidates.append(candidate);actions.append(a)
                    scores=selection(candidates,np.array([1.5,1.]));selected=int(np.argmin(scores));a=actions[selected]
                    env=Environment(scene)
                    for prior in row['actions'][:3]:env.step(Action.from_vector(prior))
                    actual=[]
                    for action in a:actual.append(env.step(Action.from_vector(action)))
                    success=bool(np.linalg.norm(actual[-1][0,:2]-[1.5,1.])<=.75)
                    planfile=f'{sid}-{method}-{seed}-planning.json';save(ROOT/planfile,dict(scores=scores,selected=selected,action=a,actual=actual,success=success,predictions=candidates,environment_queries_during_selection=0));planindex.append(dict(identity=sid,file=planfile,sha256=sha(ROOT/planfile)))
                    for horizon in p['horizons']:
                        sliced={k:v[:horizon] for k,v in pred.items() if k in ('states','covariance','lower','upper','contacts')}
                        records.append(dict(identity=sid,family=scene.family,split=scene.split,stratum=scene.ood_condition,method=method,seed=seed,horizon=horizon,success=success,**metrics(sliced,row['states'][4:4+horizon],active,radius)))
                print(json.dumps({'evaluated_scene':sid,'completed_scenes':len({v['identity'] for v in records})}),flush=True)
            save(ROOT/'predictions-index.json',predindex);save(ROOT/'planning-index.json',planindex);save(ROOT/'metrics.json',records);save(ROOT/'summary.json',analyze(records,p))
            from .audit import audit
            save(ROOT/'audit.json',audit(ROOT))
        verify_freeze()
    except Exception as e:
        save(ROOT/'failure.json',dict(type=type(e).__name__,message=str(e)));save(ROOT/'summary.json',dict(decision='FAILED_CAMPAIGN',failure=str(e)))
    save(ROOT/'elapsed.json',dict(seconds=time.perf_counter()-started,automatic_reruns=False))
    hashes={str(f.relative_to(ROOT)):sha(f) for f in ROOT.rglob('*') if f.is_file()}
    archive=ROOT.with_suffix('.tar.gz')
    with tarfile.open(archive,'x:gz') as tar:
        for name in hashes:tar.add(ROOT/name,arcname=name,recursive=False)
    with tarfile.open(archive) as tar:
        restored={m.name:hashlib.sha256(tar.extractfile(m).read()).hexdigest() for m in tar.getmembers()}
    if restored!=hashes:raise ValueError('Archive verification')
    save(ROOT.with_suffix('.preservation.json'),dict(files=hashes,archive_sha256=sha(archive),restore='PASS'))
    print(json.dumps({'run':str(ROOT.resolve()),'decision':load(ROOT/'summary.json')['decision']}))

if __name__=='__main__':main()
