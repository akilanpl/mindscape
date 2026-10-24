"""Read-only historical checkpoint inference/derivative audit; never fits a model."""
import copy
import hashlib
import json
from pathlib import Path
import torch
from mindscape.r57c.ordinal_model import make_model
from mindscape.r56.learning import load_checkpoint


def run(outputs, destination):
    outputs=Path(outputs); destination=Path(destination); torch.set_num_threads(1)
    comparison=json.loads((outputs/'final-comparison/comparison.json').read_text())
    report={'optimizer_updates':0,'scope':'historical diagnosis; no R6 final-test access','rows':[],
            'historical_deltas':comparison['cuda_vs_historical_r57c_changes'],
            'numeric_equivalence':json.loads((outputs/'NUMERICAL_EQUIVALENCE_REPORT.json').read_text()) if (outputs/'NUMERICAL_EQUIVALENCE_REPORT.json').exists() else None}
    for seed in (17,29,43):
        train=json.loads((outputs/f'evaluation-data-v1/train/{seed}.json').read_text()); train=[r for r in train if r['problem']['domain']=='physics']
        for condition in ('graph-concept-on','graph-concept-off','matched-gru'):
            raw=json.loads((outputs/f'independent-evaluation/evaluation-output/{seed}-{condition}-R57c.json').read_text())
            records=[r for r in raw['records'] if r['domain']=='physics']
            scopes={name:[r for r in records if (r['problem']['mass']<1)==ood] for name,ood in [('iid',False),('low_mass_long_time_ood',True)]}
            metrics={name:{'count':len(rows),'mse':sum(r['mse'] for r in rows)/len(rows), 'mae':sum(r['mae'] for r in rows)/len(rows),'component_sign_accuracy':sum(r['sign_correct'] for r in rows)/len(rows)} for name,rows in scopes.items()}
            probes={}
            for probe in ('mass_times_two','dt_times_two','force_axis_0_plus_one','force_axis_1_plus_one'):
                rows=[r for r in raw['perturbations'] if r['probe']==probe]
                probes[probe]={'count':len(rows),'mse':sum(sum((a-b)**2 for a,b in zip(r['prediction']['physics_delta'],r['analytic_reference']['physics_delta']))/2 for r in rows)/len(rows)}
            path=outputs/f'verified-fits/{seed}-{condition}/final.json'; digest=hashlib.sha256(path.read_bytes()).hexdigest()
            model=make_model(condition,seed);load_checkpoint(model,path,digest);model.eval()
            derivatives=[]
            for mass in (.7,2.3,6.1):
                for dt in (.2,.8):
                    for force in ([1.7,-2.1],[-1.7,2.1]):
                        p={'domain':'physics','operation':'force','mass':mass,'dt':dt,'force':force,'velocity':[.3,-.7],'concept_ids':[2]}
                        def pred(p):
                            with torch.no_grad():return model(model.prepare(p))['physics_delta'].tolist()
                        base=pred(p)
                        for name in ('force_x','force_y','mass','dt'):
                            changed=copy.deepcopy(p);epsilon=.01
                            if name.startswith('force'):changed['force'][0 if name=='force_x' else 1]+=epsilon
                            else:changed[name]+=epsilon
                            numerical=[(a-b)/epsilon for a,b in zip(pred(changed),base)]
                            expected=[(changed['force'][i]*changed['dt']/changed['mass']-force[i]*dt/mass)/epsilon for i in range(2)]
                            relevant=[0] if name=='force_x' else [1] if name=='force_y' else [0,1]
                            derivatives.append({'variable':name,'measured':numerical,'reference':expected,'relevant_sign_agreement':[numerical[i]*expected[i]>0 for i in relevant]})
            prepared=model.prepare({'domain':'physics','operation':'force','mass':2.3,'dt':.8,'force':[1.7,-2.1],'velocity':[.3,-.7],'concept_ids':[2]})
            out=model(prepared)['physics_delta']; grad=torch.autograd.grad(out.square().sum(),model.physics_head[0].weight)[0]
            row={'seed':seed,'condition':condition,'checkpoint_sha256':digest,'metrics':metrics,'interventions':probes,
                 'train_ranges':{k:[min(r['problem'][k] for r in train),max(r['problem'][k] for r in train)] for k in ('mass','dt')},
                 'final_ranges':{k:[min(r['problem'][k] for r in records),max(r['problem'][k] for r in records)] for k in ('mass','dt')},
                 'derivative_sign_agreement':{v:sum(x for d in derivatives if d['variable']==v for x in d['relevant_sign_agreement'])/sum(len(d['relevant_sign_agreement']) for d in derivatives if d['variable']==v) for v in ('force_x','force_y','mass','dt')},
                 'finite_difference_grid':derivatives,'physics_head_gradient_norm':float(grad.norm()),
                 'worst_cases':[{k:r[k] for k in ('identity','problem','prediction','reference','mse','mae','sign_correct')} for r in sorted(records,key=lambda r:r['mse'],reverse=True)[:3]]}
            report['rows'].append(row)
    destination.write_text(json.dumps(report,indent=2,allow_nan=False));return report

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--outputs',required=True);parser.add_argument('--destination',required=True);args=parser.parse_args();run(args.outputs,args.destination)
