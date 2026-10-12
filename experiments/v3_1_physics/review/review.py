import csv,json
from pathlib import Path
from collections import defaultdict
import numpy as np

root=Path('results/v3_1_physics/native-pilot-1791816435786858000')
dataset=Path('results/v3_1_physics/dataset-pilot-v1')
manifest=json.loads((dataset/'manifest.json').read_text())
lookup={r['identity']:r for r in manifest['records']}
group=defaultdict(list);errors=defaultdict(dict);scene_rows=[]
for p in root.rglob('*.json'):
 if len(p.stem)!=64:continue
 saved=json.loads(p.read_text());t=saved['trace'];i=saved['identity']
 if 'metrics' not in t:continue
 record=lookup[t['scene_id']]
 assert record['scene']['split'].startswith('development')
 with np.load(dataset/record['file'],allow_pickle=False) as d:
  truth=d['states'][4:4+t['horizon'],:,:4]
  active=d['states'][3,:,9]>0
 prediction=np.asarray(t['predictions']['states'],np.float32)[:,active]
 error=prediction-truth[:,active]
 var=np.asarray(t['predictions']['variance'],np.float32)[:,active]
 cum=var.cumsum(0)
 original_coverage=(abs(error)<=1.96*np.sqrt(cum)).mean(0).mean(0)
 # Diagnostic only: linear kinematic covariance with independent velocity innovations.
 q=var[...,2:4]
 corrected=[]
 for k in range(len(q)):
  weights=np.arange(k+1,0,-1,dtype=np.float32)[:,None,None]
  pvar=(q[:k+1]*weights**2).sum(0)/900
  vvar=q[:k+1].sum(0)
  corrected.append(np.concatenate((pvar,vvar),-1))
 corrected=np.asarray(corrected)
 altcoverage=(abs(error)<=1.96*np.sqrt(corrected)).mean(0).mean(0)
 key=(i['family'],i['seed'],i['variant'],t['split'],t['horizon'])
 value={'mse_position':float((error[...,:2]**2).mean()),'mse_velocity':float((error[...,2:]**2).mean()),
        'original_coverage_by_coordinate':original_coverage.tolist(),
        'diagnostic_kinematic_coverage':float((abs(error)<=1.96*np.sqrt(corrected)).mean()),
        'diagnostic_coverage_by_coordinate':altcoverage.tolist(),
        'original_mean95_width':float((3.92*np.sqrt(cum)).mean()),
        'diagnostic_mean95_width':float((3.92*np.sqrt(corrected)).mean())}
 group[key].append(value)
 errors[key][t['scene_id']]=t['metrics']['normalized_rollout_mse']
 if t['horizon']==32:
  scene_rows.append({'family':i['family'],'seed':i['seed'],'variant':i['variant'],
                     'panel':t['split'],'condition':record['scene']['ood_condition'],
                     'objects':len(record['scene']['objects']),'scene':t['scene_id'],
                     'error32':t['metrics']['normalized_rollout_mse'],**{k:v for k,v in value.items() if not isinstance(v,list)}})
summary=[]
for key,rows in sorted(group.items()):
 row=dict(zip(('family','seed','variant','panel','horizon'),key))
 for field in rows[0]:row[field]=np.mean([r[field] for r in rows],axis=0).tolist()
 summary.append(row)
components=[];rng=np.random.default_rng(31012)
for family in ['motion','interaction']:
 for variant in ['no_history','concepts','memory']:
  ids=sorted(errors[family,11,'candidate','development_stress',32])
  a=np.array([np.mean([errors[family,s,variant,'development_stress',32][k] for s in [11,23]]) for k in ids])
  b=np.array([np.mean([errors[family,s,'candidate','development_stress',32][k] for s in [11,23]]) for k in ids])
  idx=rng.integers(len(ids),size=(2000,len(ids)))
  reduction=1-a[idx].mean(1)/b[idx].mean(1)
  components.append({'family':family,'variant':variant,'relative_reduction':float(1-a.mean()/b.mean()),
                     'paired_scene_bootstrap95':np.quantile(reduction,[.025,.975]).tolist()})
with Path('work/v3_1_review/scene-errors.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=list(scene_rows[0]));writer.writeheader();writer.writerows(scene_rows)
Path('work/v3_1_review/diagnostics.json').write_text(json.dumps({'coordinate_diagnostics':summary,
 'component_intervals':components,'limits':'Posthoc measurement diagnosis only; original gates and outputs unchanged. Linear covariance assumes independent innovations and ignores neural feedback Jacobian.'},indent=2))
for r in summary:
 if r['variant']=='candidate' and r['horizon']==32 and r['panel']=='development_stress':print(r)
print('components',components)
for family in ['motion','interaction']:
 print('configuration',family,[(r['seed'],r['variant'],r['error32']) for r in scene_rows if r['family']==family and r['condition']=='configuration' and r['variant'] in ['candidate','matched']])
