"""Independent scene aggregation/bootstrap and unchanged scientific gate implementation."""
import numpy as np

KEYS=('mse','velocity_mse','coverage','interval_score','contact_accuracy','floor_penetration','success')
def independent(records,protocol):
    rng=np.random.default_rng(523);families={};checks={};perseed=[];gatevalues={}
    for family in ('motion','interaction'):
        rr=[r for r in records if r['family']==family];cells={}
        for split in ('iid','ood'):
            ids=sorted({r['identity'] for r in rr if r['split']==split})
            for h in protocol['horizons']:
                for method in sorted({r['method'] for r in rr}):
                    scenevalues={}
                    for i in ids:
                        group=[r for r in rr if r['identity']==i and r['method']==method and r['horizon']==h]
                        scenevalues[i]={k:float(np.mean([r[k] for r in group])) for k in KEYS}
                    cells[f'{split}/{h}/{method}']={'independent_scenes':len(ids),'means':{k:float(np.mean([scenevalues[i][k] for i in ids])) for k in KEYS},'scene_values':scenevalues}
        for split in ('iid','ood'):
            for method in sorted({r['method'] for r in rr}):
                for seed in sorted({r['seed'] for r in rr if r['method']==method}):
                    for horizon in protocol['horizons']:
                        group=[r for r in rr if r['split']==split and r['method']==method and r['seed']==seed and r['horizon']==horizon]
                        perseed.append(dict(family=family,split=split,method=method,seed=seed,horizon=horizon,independent_scenes=len(group),successes=sum(r['success'] for r in group),**{k:float(np.mean([r[k] for r in group])) for k in KEYS}))
        mean=lambda method,split='ood',h=32:cells[f'{split}/{h}/{method}']['means']
        controls=protocol['baseline_methods'];ids=sorted(cells['ood/32/structured']['scene_values']);ratios=[];differences=[]
        for _ in range(2000):
            sampled=rng.choice(ids,len(ids))
            avg=lambda method,key:float(np.mean([cells[f'ood/32/{method}']['scene_values'][i][key] for i in sampled]))
            ratios.append(avg('structured','mse')/max(1e-12,min(avg(c,'mse') for c in controls)))
            differences.append(avg('structured','success')-max(avg(c,'success') for c in controls))
        ratio_ci=np.quantile(ratios,[.025,.975]).tolist();difference_ci=np.quantile(differences,[.025,.975]).tolist()
        g=protocol['gate'];v=mean('structured');best=lambda key,split='ood':min(mean(c,split)[key] for c in controls)
        values={'ood_mse_ratio':v['mse']/best('mse'),'iid_mse_ratio':mean('structured','iid')['mse']/best('mse','iid'),'velocity_mse_ratio':v['velocity_mse']/best('velocity_mse'),'coverage':v['coverage'],'interval_score_ratio':v['interval_score']/best('interval_score'),'contact_accuracy':v['contact_accuracy'],'floor_penetration':v['floor_penetration'],'planning_gain':v['success']-max(mean(c)['success'] for c in controls),'stress_mse':mean('structured',h=64)['mse'],'ood_ratio_ci95':ratio_ci,'planning_difference_ci95':difference_ci}
        c={'ood_mse':v['mse']<=g['ood_mse_ratio']*best('mse') and ratio_ci[1]<1,'iid_mse':mean('structured','iid')['mse']<=g['iid_mse_ratio']*best('mse','iid'),'velocity':v['velocity_mse']<=g['velocity_mse_ratio']*best('velocity_mse'),'coverage':g['coverage_min']<=v['coverage']<=g['coverage_max'],'interval_score':v['interval_score']<=g['interval_score_ratio']*best('interval_score'),'contact':v['contact_accuracy']>=g['contact_accuracy_min'],'floor':v['floor_penetration']<=g['maximum_floor_penetration'],'planning':values['planning_gain']>=g['planning_gain'] and difference_ci[0]>0,'stress':values['stress_mse']<=g['maximum_stress_mse']}
        seed_deltas={str(seed):float(np.mean([r['mse'] for r in rr if r['split']=='ood' and r['horizon']==32 and r['method']=='structured' and r['seed']==seed])-np.mean([r['mse'] for r in rr if r['split']=='ood' and r['horizon']==32 and r['method']=='matched' and r['seed']==seed])) for seed in protocol['optimizer_seeds']}
        c['all_seed_ood_gain']=all(v<0 for v in seed_deltas.values());values['seed_structured_minus_matched_mse']=seed_deltas
        checks[family]=c;gatevalues[family]=values
        strata={s:{method:float(np.mean([r['mse'] for r in rr if r['stratum']==s and r['method']==method and r['horizon']==32])) for method in controls+['structured']} for s in sorted({r['stratum'] for r in rr if r['split']=='ood'})}
        families[family]={'cells':cells,'ood_ratio_ci95':ratio_ci,'planning_difference_ci95':difference_ci,'strata':strata}
    return {'decision':'GO_TO_CONFIRMATION' if all(all(c.values()) for c in checks.values()) else 'NO_GO','gates':checks,'metrics':families,'independent_evaluation_scenes':len({r['identity'] for r in records}),'repeated_seeds_not_independent':True},gatevalues,perseed
