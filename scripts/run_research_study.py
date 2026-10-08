"""Frozen controlled numerical-claim study; all conditions use the shared runner."""
import argparse
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
import statistics

from mindscape.core.config import load_config
from mindscape.data.backends import stable_hash
from mindscape.data.generation import load_dataset
from mindscape.evaluation.metrics import n_star, data_efficiency_ratio
from mindscape.evaluation.runner import run
from mindscape.models.study import StudyModel
from mindscape.training.claims import CONDITIONS, train_claims


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--config',default='configs/experiments/research_study.toml')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    config=load_config(args.config)
    splits,manifest=load_dataset(args.dataset)
    protocol={'config':config,'dataset_hash':stable_hash(manifest),'manifest':manifest,
        'primary':'unconstrained numerical claims; no inference correctness feedback',
        'selection':'fixed optimizer steps across conditions; no test tuning',
        'fairness':{'same_training_problem_ids':True,'same_evaluation_sets':True,
            'same_backbone_and_head_capacity':True,'same_evaluator':True,
            'no_corrected_arithmetic_features':True,'minimum_candidates':100,
            'no_future_observed_state_features':True,'no_test_training':True,
            'B_receives_trajectory_labels':True,'C_receives_only_own_actions_plus_scalar_feedback':True,
            'C_feedback_query_budget_per_problem':config['attempts_per_problem'],
            'remaining_differences':['state decomposition is manually specified for Mindscape',
                'training label/query counts differ intentionally by regime',
                'dream adds model calls; compute is measured, not equalized',
                'only trajectory emitters can satisfy groundedness definition']}}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    records=[]
    def evaluate(model,metadata,variant='full'):
        for split in ['test','ood_test']:
            folder,metrics=run(model,args.dataset,output/'evaluations',split,metadata['regime'],metadata['config']['budget'],
                notes='Controlled claim study: no correction, no singleton mask, same neural capacity')
            record={'condition':model.condition,'variant':variant,'seed':metadata['config']['seed'],
                'budget':metadata['config']['budget'],'split':split,'directory':str(folder),'metrics':metrics,
                'training_time':metadata['training_time'],'parameter_count':model.parameter_count,
                'training_rows':metadata['training_rows'],'experience':metadata.get('experience')}
            rows=[json.loads(line) for line in (folder/'predictions.jsonl').read_text().splitlines()]
            candidates=[d['candidate_action_count'] for row in rows for d in (row.get('diagnostics') or {}).get('decisions',[])]
            if candidates and min(candidates)<=1:raise ValueError('Invalid singleton primary benchmark')
            record['candidate_count_min']=min(candidates) if candidates else None
            record['candidate_count_max']=max(candidates) if candidates else None
            record['error_counts']=dict(Counter(row['error_type'] for row in rows))
            byid={e.example_id:e for e in splits[split]}
            grouped=defaultdict(list)
            for row in rows:
                e=byid[row['example_id']]
                key=(e.metadata['structural_category'],e.metadata['difficulty']['carry_count'])
                grouped[key].append(row)
            record['difficulty_errors']=[{'structure':key[0],'carry_count':key[1],'count':len(values),
                'accuracy':sum(r['correct'] for r in values)/len(values),
                'errors':dict(Counter(r['error_type'] for r in values))} for key,values in sorted(grouped.items())]
            # Every run receives the same frozen fairness record and a plot directory.
            (folder/'fairness.json').write_text(json.dumps(protocol['fairness'],indent=2))
            (folder/'plots').mkdir()
            records.append(record)
            with (output/'runs.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
    for budget in config['budgets']:
        for seed in config['seeds']:
            for condition in CONDITIONS:
                cfg={key:value for key,value in config.items() if key not in ('budgets','seeds','thresholds')}
                cfg.update(condition=condition,budget=budget,seed=seed)
                path=output/f'{condition}_n{budget}_s{seed}'
                model,metadata=train_claims(splits,manifest,cfg,path)
                model=StudyModel.load(path/'checkpoint')
                evaluate(model,metadata)
                print(f'full {condition} n={budget} seed={seed} rows={metadata["training_rows"]}',flush=True)
    # Three-seed component intervention matrix at the common 50-example budget.
    for seed in config['seeds']:
        budget=50 if 50 in config['budgets'] else min(config['budgets'])
        basepath=output/f'trajectory_n{budget}_s{seed}'
        base=StudyModel.load(basepath/'checkpoint');meta=base.training_metadata
        for variant,options in [('no_dream',{'dream':False}),('constrained',{'constrained':True}),
                                ('no_environment_interaction',{'no_interaction':True}),
                                ('serialization_distractor',{'perturb':True})]:
            model=copy.copy(base)
            for key,value in options.items():setattr(model,key,value)
            evaluate(model,meta,variant)
        cfg={key:value for key,value in config.items() if key not in ('budgets','seeds','thresholds')}
        cfg.update(condition='trajectory',budget=budget,seed=seed,no_state=True)
        model,meta=train_claims(splits,manifest,cfg,output/f'no_state_n{budget}_s{seed}')
        evaluate(model,meta,'no_state')
        cfg.update(condition='experiential',no_state=False,no_memory=True)
        model,meta=train_claims(splits,manifest,cfg,output/f'online_no_replay_n{budget}_s{seed}')
        evaluate(model,meta,'no_episodic_replay')
    aggregate=[]
    for condition in CONDITIONS:
        for budget in config['budgets']:
            for split in ['test','ood_test']:
                rows=[r for r in records if r['condition']==condition and r['budget']==budget and r['split']==split and r['variant']=='full']
                summary={'condition':condition,'budget':budget,'split':split,'seeds':len(rows)}
                for metric in ['accuracy','grounded_rate','goal_success_rate','unsupported_rate','trajectory_validity','inference_time']:
                    values=[r['metrics'][metric] for r in rows if r['metrics'][metric] is not None]
                    summary[metric]={'mean':statistics.mean(values) if values else None,
                        'std':statistics.stdev(values) if len(values)>1 else None}
                aggregate.append(summary)
    efficiency=[]
    for split in ['test','ood_test']:
        curves={c:{r['budget']:r['accuracy']['mean'] for r in aggregate if r['condition']==c and r['split']==split} for c in CONDITIONS}
        for alpha in config['thresholds']:
            ns={c:n_star(curves[c],alpha) for c in CONDITIONS}
            ns={key:'not_reached' if value=='not reached' else value for key,value in ns.items()}
            ders={c:data_efficiency_ratio(curves['answer_only'],curves[c],alpha) for c in ['trajectory','experiential']}
            ders={key:'not_reached' if value=='not reached' else value for key,value in ders.items()}
            efficiency.append({'split':split,'threshold':alpha,'n_star':ns,'DER':ders})
    ablations=[]
    for variant in sorted({r['variant'] for r in records}-{'full'}):
        condition='experiential' if variant=='no_episodic_replay' else 'trajectory'
        for split in ['test','ood_test']:
            changed=[r for r in records if r['variant']==variant and r['split']==split]
            original=[r for r in records if r['variant']=='full' and r['condition']==condition and r['budget']==changed[0]['budget'] and r['split']==split]
            result={'variant':variant,'reference':condition,'split':split,'delta_data_efficiency':'not_estimated_single_budget'}
            for metric in ['accuracy','grounded_rate','goal_success_rate']:
                result['delta_'+metric]=statistics.mean(r['metrics'][metric] for r in changed)-statistics.mean(r['metrics'][metric] for r in original)
            ablations.append(result)
    (output/'aggregate.json').write_text(json.dumps(aggregate,indent=2))
    (output/'data_efficiency.json').write_text(json.dumps(efficiency,indent=2))
    (output/'ablations.json').write_text(json.dumps({'measured':ablations,'not_applicable':{
        'relation_representation':'Single multiplication relation is constant; removing it is a bias-feature intervention, not graph reasoning.',
        'goal_evaluator':'External goal verification defines evaluation and cannot be removed fairly; no model-visible goal oracle exists.',
        'trajectory_supervision':'Regime C removes trajectory labels by design; B versus C is a training-regime contrast, not an isolated component ablation.'}},indent=2))
    print(output,flush=True)


if __name__=='__main__':main()
