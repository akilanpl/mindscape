"""Measured summaries, intervals, paired tests and standard standalone SVG charts."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics

from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.shapes import Drawing,String,Line
from reportlab.graphics import renderSVG
from reportlab.lib import colors

CONDITIONS=['answer_only','structured','trajectory','experiential']
PALETTE=[colors.HexColor(v) for v in ['#2166ac','#7570b3','#1b9e77','#d95f02']]


def wilson(successes,n):
    if not n:return None
    z=1.959963984540054
    p=successes/n;den=1+z*z/n
    mid=(p+z*z/(2*n))/den
    radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0,mid-radius),min(1,mid+radius)]


def mcnemar(a,b):
    left=sum(x and not y for x,y in zip(a,b));right=sum(y and not x for x,y in zip(a,b))
    n=left+right
    p=min(1,2*sum(math.comb(n,k) for k in range(min(left,right)+1))/2**n) if n else 1.0
    return {'baseline_only_correct':left,'condition_only_correct':right,'p_value':p}


def lineplot(path,title,series,ylabel='Percent',logx=True):
    drawing=Drawing(740,440)
    drawing.add(String(65,414,title,fontName='Helvetica-Bold',fontSize=15))
    drawing.add(String(65,394,'Controlled numerical claims; means of three seeds; no oracle correction.',fontSize=10))
    chart=LinePlot();chart.x=65;chart.y=95;chart.width=600;chart.height=270
    chart.data=[points for name,points in series]
    xs=[x for _,points in series for x,y in points];ys=[y for _,points in series for x,y in points]
    chart.xValueAxis.valueMin=min(xs);chart.xValueAxis.valueMax=max(xs)
    chart.xValueAxis.valueSteps=sorted(set(xs))
    chart.yValueAxis.valueMin=0
    chart.yValueAxis.valueMax=max(1,math.ceil(max(ys,default=0)/5)*5)
    chart.yValueAxis.valueSteps=[chart.yValueAxis.valueMax*i/4 for i in range(5)]
    for i,(name,points) in enumerate(series):
        color=PALETTE[i%len(PALETTE)];chart.lines[i].strokeColor=color;chart.lines[i].strokeWidth=2
        drawing.add(Line(65+i*168,62,85+i*168,62,strokeColor=color,strokeWidth=2))
        drawing.add(String(90+i*168,59,name,fontSize=9))
    drawing.add(chart)
    drawing.add(String(260,30,'Training problems' if logx else 'Evaluation runtime (seconds)',fontSize=11))
    drawing.add(String(8,240,ylabel,fontSize=9))
    drawing.add(String(65,12,'Raw per-seed values and uncertainty remain in aggregate.json / statistics.json.',fontSize=9))
    renderSVG.drawToFile(drawing,str(path))


def bars(path,title,labels,values):
    drawing=Drawing(900,480)
    chart=VerticalBarChart();chart.x=60;chart.y=115;chart.width=800;chart.height=300
    chart.data=[values];chart.categoryAxis.categoryNames=labels
    chart.categoryAxis.labels.angle=25;chart.categoryAxis.labels.fontSize=8
    low=min(values,default=0);high=max(values,default=0)
    chart.valueAxis.valueMin=min(0,math.floor(low)-1)
    chart.valueAxis.valueMax=max(1,math.ceil(high)+1)
    chart.bars[0].fillColor=PALETTE[2]
    drawing.add(chart)
    drawing.add(String(60,445,title,fontName='Helvetica-Bold',fontSize=15))
    renderSVG.drawToFile(drawing,str(path))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('results');parser.add_argument('--plots-directory',default='plots');args=parser.parse_args()
    root=Path(args.results);plots=root/args.plots_directory;plots.mkdir(exist_ok=False)
    records=[json.loads(line) for line in (root/'runs.jsonl').read_text().splitlines()]
    aggregate=json.loads((root/'aggregate.json').read_text())
    intervals=[]
    for r in records:
        m=r['metrics'];n=m['example_count'];successes=round(m['accuracy']*n)
        intervals.append({**{key:r[key] for key in ['condition','variant','budget','seed','split']},
                          'accuracy_wilson_95':wilson(successes,n)})
    paired=[]
    for split in ['test','ood_test']:
        baseline=next(r for r in records if r['variant']=='full' and r['condition']=='answer_only' and r['budget']==1000 and r['seed']==0 and r['split']==split)
        def predictions(r):return [json.loads(line) for line in (Path(r['directory'])/'predictions.jsonl').read_text().splitlines()]
        base=predictions(baseline)
        for condition in ['structured','trajectory','experiential']:
            other=next(r for r in records if r['variant']=='full' and r['condition']==condition and r['budget']==1000 and r['seed']==0 and r['split']==split)
            candidate=predictions(other)
            if [x['example_id'] for x in base]!=[x['example_id'] for x in candidate]:raise ValueError('Unpaired test sets')
            paired.append({'split':split,'condition':condition,'budget':1000,'seed':0,
                           **mcnemar([r['correct'] for r in base],[r['correct'] for r in candidate])})
    ordering=sorted(range(len(paired)),key=lambda i:paired[i]['p_value'])
    previous=0
    for rank,index in enumerate(ordering):
        adjusted=min(1,(len(paired)-rank)*paired[index]['p_value'])
        previous=max(previous,adjusted);paired[index]['holm_p_value']=previous
    (root/'statistics.json').write_text(json.dumps({'run_intervals':intervals,
        'paired_tests':paired,'interpretation':'Exploratory within-seed paired tests; no pooled repeated-seed independence or architecture-causal claim.'},indent=2))
    for split in ['test','ood_test']:
        for metric in ['accuracy','grounded_rate','goal_success_rate']:
            series=[(c,[(r['budget'],100*r[metric]['mean']) for r in aggregate if r['condition']==c and r['split']==split]) for c in CONDITIONS]
            lineplot(plots/f'{split}_{metric}.svg',f'{split}: {metric} versus training data',series)
    # Censored data-efficiency display is the measured accuracy curve, with thresholds retained in JSON.
    efficiencies=json.loads((root/'data_efficiency.json').read_text())
    drawing=Drawing(740,380)
    drawing.add(String(50,340,'Data efficiency: observed threshold crossings only',fontSize=16))
    y=300
    for row in efficiencies:
        drawing.add(String(50,y,f'{row["split"]}, alpha={row["threshold"]}: N* {row["n_star"]}',fontSize=10));y-=34
    renderSVG.drawToFile(drawing,str(plots/'data_efficiency.svg'))
    ablations=json.loads((root/'ablations.json').read_text())['measured']
    selected=[r for r in ablations if r['split']=='test']
    bars(plots/'ablation_accuracy.svg','IID ablations: change in accuracy (percentage points)',[r['variant'] for r in selected],[100*r['delta_accuracy'] for r in selected])
    errors=Counter()
    for r in records:
        if r['variant']=='full':errors.update(r['error_counts'])
    bars(plots/'error_distribution.svg','Full study outcome counts (all budgets/seeds; repeated evaluation sets)',list(errors),list(errors.values()))
    # Match curve means to the same measured mean evaluation cost.
    costseries=[]
    for c in CONDITIONS:
        points=[(r['inference_time']['mean'],100*r['accuracy']['mean']) for r in aggregate if r['condition']==c and r['split']=='test']
        costseries.append((c,sorted(points)))
    lineplot(plots/'accuracy_inference_cost.svg','IID accuracy versus evaluation runtime',costseries,ylabel='Accuracy %',logx=False)
    failures=[]
    for r in records:
        for item in r['difficulty_errors']:
            failures.append({**{key:r[key] for key in ['condition','variant','budget','seed','split']},**item})
    (root/'difficulty_failures.json').write_text(json.dumps(failures,indent=2))
    # Persist per-run diagnostic categories alongside the original evaluator taxonomy.
    taxonomy=Counter()
    for r in records:
        rows=predictions(r)
        mapped=[]
        for row in rows:
            original=row['error_type']
            decisions=(row.get('diagnostics') or {}).get('decisions',[])
            if row['grounded']:category='correct'
            elif any(d.get('verifier_judgment') is False for d in decisions):category='wrong_action'
            else:category={'invalid_action':'wrong_action','malformed_trajectory':'trajectory_failure',
                'model_error':'other','unknown':'other'}.get(original,original)
            taxonomy[category]+=1
            if category!='correct':mapped.append({'example_id':row['example_id'],'category':category,'original_error_type':original})
        (Path(r['directory'])/'failures.json').write_text(json.dumps(mapped,indent=2))
    (root/'failure_taxonomy.json').write_text(json.dumps(dict(taxonomy),indent=2))
    print(plots)


if __name__=='__main__':main()
