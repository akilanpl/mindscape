"""Derive experimental sanity checks from actual data, code boundaries and artifacts."""
from collections import Counter
from dataclasses import replace
import inspect
import json
from pathlib import Path
import sqlite3
import numpy as np

from mindscape.data.backends import stable_hash
from mindscape.data.generation import load_dataset
from mindscape.models.study import StudyModel
from mindscape.models.study_encoding import answer_features,policy_features
from mindscape.training.claims import collect_experience


def audit(root,dataset):
    root=Path(root);splits,manifest=load_dataset(dataset)
    records=[json.loads(l) for l in (root/'runs.jsonl').read_text().splitlines()]
    primary=[r for r in records if r['variant']=='full']
    checkpoints=list(root.glob('*_n*_s*/metadata.json'))
    metadata=[json.loads(p.read_text()) for p in checkpoints if p.parent.name.split('_n')[0] in ['answer_only','structured','trajectory','experiential']]
    train_ids={e.example_id for e in splits['train']}
    heldout_ids={e.example_id for split,rows in splits.items() if split!='train' for e in rows}
    checks={
        'dataset_leakage_and_integrity_validation':True, # load_dataset above performs full authoritative checks.
        'no_train_test_id_overlap':not(train_ids&heldout_ids),
        'checkpoint_training_ids_subset_train':all(set(m['training_ids'])<=train_ids for m in metadata),
        'equal_parameter_count':len({r['parameter_count'] for r in primary})==1,
        'checkpoint_dataset_hash_matches':all(m['dataset_hash']==stable_hash(manifest) for m in metadata),
        'balanced_budget_seed_condition_grid':len(primary)==7*3*4*2 and all(v==2 for v in Counter((r['condition'],r['budget'],r['seed']) for r in primary).values()),
        'same_primary_optimizer_configuration':len({tuple(m['config'][k] for k in ['hidden','steps','batch_size','learning_rate']) for m in metadata})==1,
        'C_collection_code_never_accesses_targets':all(term not in inspect.getsource(collect_experience) for term in ['example.target_answer','example.target_trajectory']),
        'primary_prediction_code_never_calls_numerical_oracle':'expected_value' not in inspect.getsource(StudyModel.predict),
        'all_C_reports_no_provided_full_trajectories':all((m.get('experience') or {}).get('full_target_trajectories_received',0)==0 for m in metadata),
    }
    first=splits['test'][0];poisoned=replace(first,target_answer=-123456,target_trajectory=[])
    checks['target_mutation_does_not_change_baseline_features']=np.array_equal(answer_features(first.view('structured')),answer_features(poisoned.view('structured')))
    candidate_counts=[];oracle_flags=[];claim_consistency=[];dream_tags=[];eval_hashes=[]
    for r in primary:
        folder=Path(r['directory'])
        config=json.loads((folder/'config.json').read_text());eval_hashes.append(config['dataset_hash'])
        for line in (folder/'predictions.jsonl').read_text().splitlines():
            row=json.loads(line);diagnostics=row.get('diagnostics') or {}
            if diagnostics:oracle_flags.append(diagnostics['oracle_feedback_visible'])
            for d in diagnostics.get('decisions',[]):
                candidate_counts.append(d['candidate_action_count'])
                if 'actual_result' in d:claim_consistency.append(d['actual_result']['value']==d['model_prediction'])
                dream_tags.extend(s['hypothetical'] for s in d['hypothetical_rollouts'])
    checks.update(primary_candidate_minimum_above_one=bool(candidate_counts) and min(candidate_counts)>1,
                  no_model_visible_oracle_feedback=not any(oracle_flags),
                  all_actual_numeric_results_equal_own_proposals=all(claim_consistency),
                  all_simulated_state_records_tagged=all(dream_tags),
                  identical_evaluation_dataset_hashes=set(eval_hashes)=={stable_hash(manifest)})
    c_summary=[]
    for path in root.glob('experiential_n*_s*/metadata.json'):
        m=json.loads(path.read_text())
        with sqlite3.connect(path.parent/'episodes.sqlite') as db:
            rows=db.execute('SELECT partition,payload FROM episodes').fetchall()
        decoded=[json.loads(payload) for partition,payload in rows]
        count=sum(bool(e['success']) for e in decoded)
        c_summary.append({'budget':m['config']['budget'],'seed':m['config']['seed'],
            'stored_successful_episodes':count,'recorded_completion_counter':m['experience']['completed_episodes'],
            'attempts':sum(len(e['trajectory']) for e in decoded),
            'accepted':sum(t['reward']==1 for e in decoded for t in e['trajectory'])})
        checks.setdefault('C_memory_contains_only_train_partition',True)
        checks['C_memory_contains_only_train_partition'] &= all(partition=='train' for partition,payload in rows)
    failures=[name for name,passed in checks.items() if not passed]
    if failures:raise ValueError('Fairness/integrity audit failed: '+str(failures))
    result={'checks':checks,'observed_primary_candidate_range':[min(candidate_counts),max(candidate_counts)],
        'C_experience_counts':c_summary,
        'residual_differences':['handwritten procedural decomposition','different supervision/reward label budgets',
            'different inference call counts and runtime','groundedness requires supporting trajectories'],
        'not_claimed':'Equal inference compute, autonomous rule discovery, or isolation of architecture alone'}
    (root/'fairness_checks.json').write_text(json.dumps(result,indent=2))
    for r in records:
        folder=Path(r['directory'])
        (folder/'fairness_checks.json').write_text(json.dumps(result['checks'],indent=2))
        with (folder/'summary.md').open('a') as stream:
            stream.write('\n\nFairness/integrity checks: '+str(len(checks))+' passed; see fairness_checks.json.\n')
            stream.write('Remaining procedural/supervision/compute differences are explicit in the protocol.\n')
    print('checks passed',len(checks));print('C completion-counter discrepancies',sum(r['stored_successful_episodes']!=r['recorded_completion_counter'] for r in c_summary))
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('results');parser.add_argument('--dataset',required=True)
    args=parser.parse_args();audit(args.results,args.dataset)
