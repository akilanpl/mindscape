"""Deterministic scenes frozen before any model or qualification measurements."""
import json,hashlib
from dataclasses import replace
from pathlib import Path
from mindscape.v3.physics_world.data import generate_scene
from mindscape.v3.physics_world.schema import digest
p=json.loads(Path('configs/r5_3_physics/protocol.json').read_text()); rows=[]
for family in ('motion','interaction'):
    for split,count in [('train',24),('development',6),('iid',12),('ood',14)]:
        seed=int(digest([p['dataset_seed'],family,split])[:8],16)
        for i in range(count):
            scene=generate_scene(family,'ood' if split=='ood' else 'train',i,seed)
            scene=replace(scene,split=split,horizon=64)
            rows.append({'identity':scene.identity,'scene':scene.record()})
manifest={'protocol':p,'protocol_hash':digest(p),'records':rows,'scene_hash':digest(rows)}
Path('configs/r5_3_physics/scenes.json').write_text(json.dumps(manifest,indent=2)+'\n')
legacy={}
for family,variant in [('motion','memory'),('interaction','concepts')]:
    for seed in (11,23):
        name=f'work/v3_1-restored-evidence/native-pilot-1791816435786858000/{family}/s{seed}/{variant}/checkpoint-step000160.pt'
        legacy[name]=hashlib.sha256(Path(name).read_bytes()).hexdigest()
pins={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path('docs/v3_physics/recovery/scientific-report.md'),*Path('experiments/v3_1_physics').rglob('*')] if f.is_file()}
paths=[*Path('src/mindscape/r5_3_physics').glob('*.py'),*Path('configs/r5_3_physics').glob('*.json'),*Path('scripts/r5_3_physics').glob('*'),*Path('tests/r5_3_physics').glob('*.py'),*Path('src/mindscape/v3/physics_world').glob('*.py'),Path('src/mindscape/v3_1/physics/model.py')]
freeze={'sources':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in paths if f.is_file() and f.name!='freeze.json'},'historical_pins':pins,'legacy_checkpoints':legacy}
Path('configs/r5_3_physics/freeze.json').write_text(json.dumps(freeze,indent=2)+'\n')
print(json.dumps({'frozen_scenes':len(rows),'training':48,'independent_evaluation':52,'development':12}))
