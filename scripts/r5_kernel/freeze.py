"""Pin fixed existing assets and negative results. No data generation or model search."""
import json,hashlib
from pathlib import Path
ROOT=Path.cwd()
def load(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as stream:
        while b:=stream.read(1024*1024):h.update(b)
    return h.hexdigest()
old=load('experiments/v3_integration/native_v1/summary.json')['asset_hashes']
paths={'math_neural':'results/r5_2_math/native-one-campaign/recurrent-s11.pt','physics_neural':'results/r5_3_physics/native-one-campaign/structured-s11.pt','shared_core':'results/r5_1_learning/native-one-pilot/fits/coding-shared-s11-u120.pt'}
paths['math_neural']='results/r5_2_math/native-one-campaign/checkpoints/recurrent-s11.pt'
for name,suffix in [('qwen','model.safetensors'),('coding_evaluator','evaluator-s11.pt'),('coding_memory','memory.json'),('wasi','python.wasm'),('wasi_compiled','python.cwasm')]:paths[name]=next(p for p in old if p.endswith(suffix))
assets={name:{'path':path,'sha256':sha(path)} for name,path in paths.items()}
for name in ('qwen','coding_evaluator','coding_memory','wasi','wasi_compiled'):
    assert assets[name]['sha256']==old[paths[name]]
r51=load('results/r5_1_learning/native-one-pilot/summary.json');fit=next(f for f in r51['fits'] if f['identity']['target']=='coding' and f['identity']['condition']=='shared' and f['identity']['seed']==11)
assets['shared_core']['identity']=fit['identity'];assert assets['shared_core']['sha256']==next(c['sha256'] for c in fit['checkpoints'] if c['update']==120)
files=[]
for root in ('r5_1/screen-2cac9d0fcef076df','r5_1_learning/native-one-pilot','r5_2_math/native-one-campaign','r5_3_physics/native-one-campaign'):
    p=Path('results')/root;receipt=load(p.with_suffix('.preservation.json'))
    assert sha(p.with_suffix('.tar.gz'))==receipt['archive_sha256']
    files += [p.with_suffix('.tar.gz'),p.with_suffix('.preservation.json')]
    for file,value in receipt['files'].items():
        assert sha(p/file)==value;files.append(p/file)
for root in ('src/mindscape/coding','src/mindscape/v3','src/mindscape/v5','src/mindscape/r5_1','src/mindscape/r5_1_pilot','src/mindscape/r5_2_math','src/mindscape/r5_3_physics','experiments/r5_2_math/recovery_v1','experiments/r5_3_physics/recovery_v1'):
    files += [p for p in Path(root).rglob('*') if p.is_file() and '__pycache__' not in str(p)]
files += [Path('scripts/v3_coding/path_schema.py'),Path('experiments/v3_integration/native_v1/summary.json')]
status={'R5.1':{'decision':r51['decision'],'gates':r51['gates'],'relative_changed_mse_gains':r51['relative_changed_mse_gains'],'independent_tasks_per_domain':9,'exposure_confound':'domain-only120 target updates versus shared60 target+60 auxiliary; no matched target exposure'},'R5.2':{'decision':'NO_GO','provenance':'POST-HOC RECOVERY FROM FROZEN PREDICTIONS'},'R5.3':{'decision':'NO_GO','provenance':'POST-HOC RECOVERY FROM FROZEN EVIDENCE'},'V3_runtime':'verified engineering integration; no transfer claim'}
data={'version':'mindscape.kernel.assets/1','assets':assets,'history':{str(p):sha(p) for p in files},'scientific_status':status,'selection':{'math_default':'bounded verified symbolic/procedural operations; neural arithmetic NO_GO','physics_default':'analytical_contacts; lower R5.3 mean error than structured and matched in both families; intervals uncalibrated','coding_default':'preserved Qwen/WASI specialist; no model/transport/isolation changes','experimental_checkpoints':'fixed seed11 final checkpoints; no favorable checkpoint search','shared':'coding-target shared seed11 final120 used for infrastructure across all domain adapters; no transfer success'}}
Path('configs/r5_kernel/assets.json').write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps({'assets':len(assets),'history_pins':len(data['history']),'decisions':{k:v.get('decision') for k,v in status.items() if isinstance(v,dict)}}))
