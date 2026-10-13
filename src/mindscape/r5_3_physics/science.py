"""Scene-level measurements and frozen gates; simulator-independent."""
import numpy as np
from .uncertainty import interval_score,terminal_risk

def metrics(prediction,truth,active,radius):
    x=np.asarray(prediction['states'],dtype=np.float64);y=np.asarray(truth,dtype=np.float64);sel=active>0
    e=x[:,sel]-y[:,sel,:4];lo=np.asarray(prediction['lower'])[:,sel];hi=np.asarray(prediction['upper'])[:,sel]
    cov=np.asarray(prediction['covariance'])
    if not all(np.isfinite(v).all() for v in (x,e,lo,hi,cov)):raise ValueError('Nonfinite metric inputs')
    if np.any(lo>hi) or min(np.linalg.eigvalsh(cov).ravel()) < -1e-5: raise ValueError('Invalid covariance/interval')
    return dict(mse=float(((e/np.array([1,1,3,3]))**2).mean()),position_mse=float((e[...,:2]**2).mean()),velocity_mse=float((e[...,2:]**2).mean()),coverage=float(((y[:,sel,:4]>=lo)&(y[:,sel,:4]<=hi)).mean()),interval_score=float(interval_score(y[:,sel,:4],lo,hi).mean()),contact_accuracy=float(((np.asarray(prediction['contacts'])[:,sel]>=.5)==(y[:,sel,8]>=.5)).mean()),floor_penetration=float(np.maximum(0,radius[sel][None]+.02-x[:,sel,1]).max()),max_abs_state=float(abs(x).max()))

def selection(predictions,target):
    return [terminal_risk(np.asarray(p['states'])[-1,0],np.asarray(p['covariance'])[-1,:4,:4],target) for p in predictions]

def analyze(records,protocol):
    result={};gates={};rng=np.random.default_rng(523)
    for family in ('motion','interaction'):
        familyrows=[r for r in records if r['family']==family];table={}
        for split in ('iid','ood'):
            rr=[r for r in familyrows if r['split']==split];ids=sorted({r['identity'] for r in rr});methods=sorted({r['method'] for r in rr})
            for horizon in protocol['horizons']:
                for method in methods:
                    cells=[r for r in rr if r['method']==method and r['horizon']==horizon]
                    vals={i:{k:float(np.mean([r[k] for r in cells if r['identity']==i])) for k in ('mse','velocity_mse','coverage','interval_score','contact_accuracy','floor_penetration','success')} for i in ids}
                    if any(not any(r['identity']==i for r in cells) for i in ids):raise ValueError('Missing scene cell')
                    table[f'{split}/{horizon}/{method}']={'independent_scenes':len(ids),'means':{k:float(np.mean([v[k] for v in vals.values()])) for k in next(iter(vals.values()))},'scene_values':vals}
        def means(method,split='ood',h=32):return table[f'{split}/{h}/{method}']['means']
        controls=protocol['baseline_methods'];candidate=means('structured');ids=sorted(table['ood/32/structured']['scene_values']);n=len(ids)
        ratios=[];planning=[]
        for _ in range(2000):
            sample=rng.choice(ids,n)
            def avg(method,key):return np.mean([table[f'ood/32/{method}']['scene_values'][i][key] for i in sample])
            ratios.append(avg('structured','mse')/max(1e-12,min(avg(c,'mse') for c in controls)))
            planning.append(avg('structured','success')-max(avg(c,'success') for c in controls))
        ratio_ci=np.quantile(ratios,[.025,.975]).tolist();planning_ci=np.quantile(planning,[.025,.975]).tolist();g=protocol['gate']
        checks={'ood_mse':candidate['mse']<=g['ood_mse_ratio']*min(means(c)['mse'] for c in controls) and ratio_ci[1]<1,'iid_mse':means('structured','iid')['mse']<=g['iid_mse_ratio']*min(means(c,'iid')['mse'] for c in controls),'velocity':candidate['velocity_mse']<=g['velocity_mse_ratio']*min(means(c)['velocity_mse'] for c in controls),'coverage':g['coverage_min']<=candidate['coverage']<=g['coverage_max'],'interval_score':candidate['interval_score']<=g['interval_score_ratio']*min(means(c)['interval_score'] for c in controls),'contact':candidate['contact_accuracy']>=g['contact_accuracy_min'],'floor':candidate['floor_penetration']<=g['maximum_floor_penetration'],'planning':candidate['success']-max(means(c)['success'] for c in controls)>=g['planning_gain'] and planning_ci[0]>0,'stress':means('structured',h=64)['mse']<=g['maximum_stress_mse']}
        checks['all_seed_ood_gain']=all(np.mean([r['mse'] for r in familyrows if r['method']=='structured' and r['seed']==s and r['split']=='ood' and r['horizon']==32])<np.mean([r['mse'] for r in familyrows if r['method']=='matched' and r['seed']==s and r['split']=='ood' and r['horizon']==32]) for s in protocol['optimizer_seeds'])
        result[family]={'cells':table,'ood_ratio_ci95':ratio_ci,'planning_difference_ci95':planning_ci,'strata':{stratum:{method:float(np.mean([r['mse'] for r in familyrows if r['stratum']==stratum and r['method']==method and r['horizon']==32])) for method in controls+['structured']} for stratum in sorted({r['stratum'] for r in familyrows if r['split']=='ood'})}}
        gates[family]=checks
    return dict(decision='GO_TO_CONFIRMATION' if all(all(v.values()) for v in gates.values()) else 'NO_GO',gates=gates,metrics=result,independent_evaluation_scenes=len({r['identity'] for r in records}),repeated_seeds_not_independent=True)
