"""Final measured summaries/claims; no interpolation or synthetic performance."""
import argparse,json,statistics,math,shutil
from pathlib import Path
from collections import Counter,defaultdict
from mindscape.evaluation.metrics import n_star,data_efficiency_ratio
from reportlab.graphics.shapes import Drawing,String,Line,Circle,Rect
from reportlab.graphics import renderSVG
from reportlab.lib import colors
from analyze_research_study import wilson,mcnemar,bars
CONDITIONS=['answer_only','structured','trajectory','experiential']
METRICS=['accuracy','grounded_rate','goal_success_rate','trajectory_validity','inference_time']
PALETTE=['#2166ac','#7570b3','#1b9e77','#d95f02']

def plot(path,title,series,ymax=100,label='Percent'):
 d=Drawing(800,440);d.add(String(60,414,title,fontSize=15));d.add(String(60,394,'Final frozen SmolLM2 backbone; three seeds; mean ± sample SD',fontSize=10))
 for y in range(5):
  val=ymax*y/4;py=95+y*67.5;d.add(Line(60,py,740,py,strokeColor=colors.HexColor('#dddddd')));d.add(String(15,py,str(round(val,2)),fontSize=9))
 for n in [10,25,50,100,250,500,1000]:
  px=60+680*math.log(n/10)/math.log(100);d.add(String(px-8,75,str(n),fontSize=9))
 for i,(name,points) in enumerate(series):
  color=colors.HexColor(PALETTE[i%4]);last=None
  for n,v,sd in points:
   if v is None:continue
   px=60+680*math.log(n/10)/math.log(100);py=95+270*v/ymax
   if last:d.add(Line(*last,px,py,strokeColor=color,strokeWidth=2))
   radius=270*(sd or 0)/ymax;d.add(Line(px,max(95,py-radius),px,min(365,py+radius),strokeColor=color));d.add(Circle(px,py,2,fillColor=color,strokeColor=color));last=(px,py)
  d.add(String(60+i*175,49,name,fontSize=9,fillColor=color))
 d.add(String(300,25,'Training problems (log scale)',fontSize=11));d.add(String(15,380,label,fontSize=9));renderSVG.drawToFile(d,str(path))

def main():
 p=argparse.ArgumentParser();p.add_argument('results');args=p.parse_args();root=Path(args.results);runs=[json.loads(x) for x in (root/'runs.jsonl').read_text().splitlines()];plots=root/'plots';plots.mkdir(exist_ok=False)
 aggregate=[]
 for c in CONDITIONS:
  for n in [10,25,50,100,250,500,1000]:
   for split in ['test','ood_test']:
    rows=[r for r in runs if r['variant']=='full' and r['condition']==c and r['budget']==n and r['split']==split]
    if len(rows)!=3:raise ValueError('Missing primary seed')
    item=dict(condition=c,budget=n,split=split,seeds=len(rows),examples_per_seed=rows[0]['metrics']['example_count'])
    for key in METRICS+['training_time']:
     values=[r[key] if key=='training_time' else r['metrics'][key] for r in rows];values=[v for v in values if v is not None]
     item[key]={'mean':statistics.mean(values) if values else None,'std':statistics.stdev(values) if len(values)>1 else None}
    aggregate.append(item)
 (root/'aggregate.json').write_text(json.dumps(aggregate,indent=2))
 efficiency=[]
 for split in ['test','ood_test']:
  curves={c:{r['budget']:r['accuracy']['mean'] for r in aggregate if r['condition']==c and r['split']==split} for c in CONDITIONS}
  for alpha in [.8,.9,.95]:
   efficiency.append(dict(split=split,threshold=alpha,n_star={c:n_star(curves[c],alpha) for c in CONDITIONS},DER={c:data_efficiency_ratio(curves['answer_only'],curves[c],alpha) for c in ['trajectory','experiential']}))
 (root/'data_efficiency.json').write_text(json.dumps(efficiency,indent=2))
 ablations=[]
 for variant in sorted({r['variant'] for r in runs}-{'full'}):
  for split in ['test','ood_test']:
   changed=[r for r in runs if r['variant']==variant and r['split']==split];condition=changed[0]['condition']
   refs=[r for r in runs if r['variant']=='full' and r['condition']==condition and r['budget']==50 and r['split']==split]
   row=dict(variant=variant,reference=condition,split=split,seeds=len(changed),delta_data_efficiency='not_estimated_single_budget')
   for key in ['accuracy','grounded_rate','goal_success_rate']:
    row['delta_'+key]=statistics.mean(r['metrics'][key] for r in changed)-statistics.mean(r['metrics'][key] for r in refs)
   ablations.append(row)
 # No-process supervision is B-to-C regime contrast, not an isolated ablation.
 for split in ['test','ood_test']:
  b=[r for r in runs if r['variant']=='full' and r['condition']=='trajectory' and r['budget']==50 and r['split']==split];c=[r for r in runs if r['variant']=='full' and r['condition']=='experiential' and r['budget']==50 and r['split']==split]
  ablations.append(dict(variant='no_trajectory_supervision_regime_C',reference='trajectory',split=split,seeds=3,delta_data_efficiency='not_estimated_single_budget',**{'delta_'+k:statistics.mean(r['metrics'][k] for r in c)-statistics.mean(r['metrics'][k] for r in b) for k in ['accuracy','grounded_rate','goal_success_rate']}))
 (root/'ablations.json').write_text(json.dumps({'measured':ablations,'interpretation':'Goal flag removal is not removal of external scoring. B/C differs in feedback and labels. No-memory uses matched rows and updates. Matched local supervised policy isolates wrapper; decomposition remains hand defined.'},indent=2))
 intervals=[];paired=[]
 def predictions(r):return [json.loads(l) for l in (Path(r['directory'])/'predictions.jsonl').read_text().splitlines()]
 for r in runs:
  m=r['metrics'];intervals.append({**{k:r[k] for k in ['condition','variant','seed','budget','split']},'accuracy_wilson95':wilson(round(m['accuracy']*m['example_count']),m['example_count'])})
 for split in ['test','ood_test']:
  b=next(r for r in runs if r['variant']=='full' and r['condition']=='answer_only' and r['budget']==1000 and r['seed']==0 and r['split']==split);bp=predictions(b)
  for c in ['structured','trajectory','experiential']:
   r=next(r for r in runs if r['variant']=='full' and r['condition']==c and r['budget']==1000 and r['seed']==0 and r['split']==split);rp=predictions(r)
   assert [x['example_id'] for x in bp]==[x['example_id'] for x in rp]
   paired.append(dict(condition=c,split=split,seed=0,budget=1000,**mcnemar([x['correct'] for x in bp],[x['correct'] for x in rp])))
 previous=0
 for rank,i in enumerate(sorted(range(len(paired)),key=lambda i:paired[i]['p_value'])):
  previous=max(previous,min(1,(len(paired)-rank)*paired[i]['p_value']));paired[i]['holm_p_value']=previous
 (root/'statistics.json').write_text(json.dumps({'run_intervals':intervals,'paired_tests':paired,'interpretation':'Exploratory one-seed paired tests, Holm six comparisons; repeated seeds not independent example replications.'},indent=2))
 failure=[];taxonomy=Counter({k:0 for k in ['arithmetic_error','wrong_action','invalid_transition','malformed_state','trajectory_failure','goal_failure','unsupported_answer','timeout','other']})
 for r in runs:
  grouped=[]
  for item in r['difficulty_errors']:failure.append({**{k:r[k] for k in ['condition','variant','budget','seed','split']},**item})
  for row in predictions(r):
   ds=(row.get('diagnostics') or {}).get('decisions',[]);original=row['error_type']
   if any(d.get('verifier_judgment') is False for d in ds):category='wrong_action'
   elif row['correct'] and row['grounded']:category='correct'
   elif not row['correct'] and r['condition'] in ('answer_only','structured'):category='arithmetic_error'
   else:category={'malformed_trajectory':'trajectory_failure','model_error':'other','correct':'unsupported_answer'}.get(original,original)
   taxonomy[category]+=1
   if category!='correct':grouped.append(dict(example_id=row['example_id'],category=category,original_error_type=original))
  folder=Path(r['directory']);(folder/'failures.json').write_text(json.dumps(grouped,indent=2))
 (root/'difficulty_failures.json').write_text(json.dumps(failure,indent=2));(root/'failure_taxonomy.json').write_text(json.dumps(dict(taxonomy),indent=2))
 for split,key,name in [('test','accuracy','accuracy'),('ood_test','accuracy','ood_accuracy'),('test','grounded_rate','groundedness'),('test','goal_success_rate','goal_success'),('test','trajectory_validity','trajectory_validity')]:
  series=[(c,[(r['budget'],100*r[key]['mean'] if r[key]['mean'] is not None else None,100*(r[key]['std'] or 0)) for r in aggregate if r['condition']==c and r['split']==split]) for c in CONDITIONS]
  plot(plots/(name+'.svg'),split+': '+name,series)
 d=Drawing(900,380);d.add(String(40,350,'Observed data-efficiency thresholds; no interpolation',fontSize=16));y=310
 for e in efficiency:d.add(String(40,y,f'{e["split"]} alpha={e["threshold"]}: {e["n_star"]}',fontSize=10));y-=40
 renderSVG.drawToFile(d,str(plots/'data_efficiency.svg'))
 selected=[r for r in ablations if r['split']=='test'];bars(plots/'ablations.svg','IID accuracy change at n50 (percentage points)',[r['variant'] for r in selected],[100*r['delta_accuracy'] for r in selected]);bars(plots/'errors.svg','Error counts; repeated evaluation sets across runs',list(taxonomy),list(taxonomy.values()))
 for key,name in [('training_time','training_compute'),('inference_time','inference_cost')]:
  series=[(c,[(r['budget'],r[key]['mean'],r[key]['std']) for r in aggregate if r['condition']==c and r['split']=='test']) for c in CONDITIONS];top=max(v+(sd or 0) for _,pts in series for n,v,sd in pts)
  plot(plots/(name+'.svg'),name+' (cached CPU seconds; encoder work shared)',series,max(1,top*1.1),'Seconds')
 # Per-run plots represent that measured run's outcomes, not a pooled claim.
 for r in runs:
  bars(Path(r['directory'])/'plots'/'outcomes.svg','Measured run rates (%)',['accuracy','grounded','goal'],[100*r['metrics'][k] for k in ['accuracy','grounded_rate','goal_success_rate']])
 claims=[dict(claim=c,status='INCONCLUSIVE',evidence=e,limitations=l) for c,e,l in [
  ('Mindscape improves accuracy','See primary curves and matched-information control.','State/scaffold and supervision differ from answer baselines; wrapper causal benefit must exceed matched local policy.'),
  ('Mindscape improves data efficiency','Observed N* and DER only.','Unequal label/query counts; compute reported separately.'),
  ('Mindscape improves structural OOD generalization','Unseen structures evaluated separately.','One domain and hand-defined cursor.'),
  ('Mindscape improves groundedness','Verified executed evidence measured.','Answer-only protocol emits no trajectory; metric favors evidence emitters by definition.'),
  ('Mindscape improves trajectory validity','Complete verifier checks.','Baselines without trajectories have null validity; matched policy is required.'),
  ('Explicit state contributes','Retrained no-state intervention at n50.','One budget and changes representation.'),
  ('Episodic memory contributes','Matched accepted rows/order/updates no-storage control.','Tests SQLite retrieval versus in-memory replay, not all reuse.'),
  ('Dream simulation contributes','Depth2/branch3 intervention n50.','Confidence scoring, not learned world model; extra inference calls.'),
  ('Architecture transfers to another environment','Goal-directed generic integer dream demo retained.','No second learned benchmark; extension deferred.')]]
 # Conservative measured adjudication; do not claim universal causal improvements.
 for c in claims:
  if c['claim']=='Mindscape improves data efficiency':c['status']='NOT SUPPORTED' if all(isinstance(e['DER']['trajectory'],str) and isinstance(e['DER']['experiential'],str) for e in efficiency) else 'INCONCLUSIVE'
  if c['claim']=='Episodic memory contributes':c['status']='NOT SUPPORTED' if all(r['delta_accuracy']==0 for r in ablations if r['variant']=='no_episodic_memory') else 'INCONCLUSIVE'
  if c['claim']=='Dream simulation contributes':c['status']='NOT SUPPORTED' if all(r['delta_accuracy']<=0 for r in ablations if r['variant']=='dream') else 'INCONCLUSIVE'
  if c['claim']=='Architecture transfers to another environment':c['status']='NOT SUPPORTED'
  if c['claim']=='Explicit state contributes':
   vals=[r['delta_accuracy'] for r in ablations if r['variant']=='no_explicit_state'];c['status']='SUPPORTED' if vals and all(v<0 for v in vals) else 'INCONCLUSIVE'
 (root/'claims.json').write_text(json.dumps(claims,indent=2));text=['# Final measured study','',f'{len(runs)} evaluations; 84 primary training runs plus6 retrained ablations; seeds0/1/2.','', '| Condition | IID mean ± SD | OOD mean ± SD | Grounded IID | Goal IID | Valid trajectory IID |','|---|---:|---:|---:|---:|---:|']
 for c in CONDITIONS:
  a=next(r for r in aggregate if r['condition']==c and r['budget']==1000 and r['split']=='test');o=next(r for r in aggregate if r['condition']==c and r['budget']==1000 and r['split']=='ood_test')
  def fmt(v):return 'N/A' if v['mean'] is None else f'{100*v["mean"]:.3f}% ± {100*(v["std"] or 0):.3f}'
  text.append(f'|{c}|{fmt(a["accuracy"])}|{fmt(o["accuracy"])}|{fmt(a["grounded_rate"])}|{fmt(a["goal_success_rate"])}|{fmt(a["trajectory_validity"])}|')
 text+=['','## Claims','', '| Claim | Status | Evidence | Limitations |','|---|---|---|---|']+[f'|{c["claim"]}|{c["status"]}|{c["evidence"]}|{c["limitations"]}|' for c in claims]
 text+=['','The comparisons characterize conditional prediction under a manually designed multiplication scaffold. They do not establish general intelligence, discovery of arbitrary procedures, or superiority over GPT/Gemini. All inference claims are uncorrected. Raw failures and historical negative results remain available.']
 (root/'summary.md').write_text('\n'.join(text)+'\n');print(root)
if __name__=='__main__':main()
