"""Report only observed evidence; retain incomplete cells and hardware strata explicitly."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from mindscape.coding.statistics import der, paired_delta, threshold, wilson_interval

BASE = Path('results/coding')
OUT = BASE / 'emergency_analysis_v1'
OUT.mkdir(parents=True, exist_ok=True)


def load(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def rows(name):
    path = BASE / name / 'rows.jsonl'
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def main():
    learning = rows('completion_gradient_v1')
    locked = rows('completion_lockbox_eval_v1')
    audit = rows('completion_trace_audit_v1')
    profile = load(BASE/'emergency_mps_v1/profile.json', {})
    cells = defaultdict(list)
    for r in learning:
        device = r.get('neural_device', 'cpu')
        precision = r.get('model_precision', 'float32')
        cells[(r['condition'], r['split'], r['seed'], r['budget'], device, precision)].append(r)
    cell_summary = []
    curves = defaultdict(dict)
    for key, values in sorted(cells.items()):
        c, split, seed, budget, device, precision = key
        n = len(values)
        successes = sum(r['success'] for r in values)
        cell_summary.append({'condition': c, 'split': split, 'seed': seed, 'budget': budget,
                                 'device': device, 'precision': precision, 'episodes': n, 'planned': 20,
                                 'complete': n == 20, 'successes': successes, 'accuracy': successes/n})
        if n == 20:
            curves[(c, split, seed, device, precision)][budget] = successes/n
    thresholds = [{'condition':k[0], 'split':k[1], 'seed':k[2], 'device':k[3], 'precision':k[4],
                   'observed_budget_thresholds':{str(q):threshold(v,q) for q in (.8,.9,.95)}}
                  for k,v in sorted(curves.items())]
    aggregates = defaultdict(list)
    for cell in cell_summary:
        if cell['complete']:
            aggregates[(cell['condition'],cell['split'],cell['budget'],cell['device'],cell['precision'])].append(cell)
    complete_seed_summaries = [{"condition":k[0],"split":k[1],"budget":k[2],"device":k[3],"precision":k[4],
        "seeds":[r['seed'] for r in v],"mean_accuracy":float(np.mean([r['accuracy'] for r in v])),
        "seed_sd":float(np.std([r['accuracy'] for r in v],ddof=1)) if len(v)>1 else None,
        "planned_seed_count":3,"complete_three_seed_cell":len(v)==3} for k,v in sorted(aggregates.items())]
    descriptive_der = []
    for k, curve in sorted(curves.items()):
        baseline = curves.get(("model_only",k[1],k[2],k[3],k[4]))
        if baseline is not None:
            descriptive_der.append({"condition":k[0],"split":k[1],"seed":k[2],"device":k[3],"precision":k[4],
                "ratios":{str(q):der(threshold(baseline,q),threshold(curve,q)) for q in (.8,.9,.95)},
                "interpretation":"Descriptive observed-budget ratios; not architecture-only causal evidence"})
    protocol_cells = defaultdict(list)
    for r in learning:
        protocol_cells[(r['condition'],r['split'],r['seed'],r['budget'])].append(r)
    protocol_curves = defaultdict(dict)
    protocol_summary = []
    for k, values in sorted(protocol_cells.items()):
        complete = len(values)==20
        accuracy = sum(r['success'] for r in values)/len(values)
        protocol_summary.append({'condition':k[0],'split':k[1],'seed':k[2],'budget':k[3],
            'episodes':len(values),'complete':complete,'accuracy':accuracy,
            'hardware_precision_counts':dict(Counter(
                r.get('neural_device','cpu')+'/'+r.get('model_precision','float32') for r in values))})
        if complete:
            protocol_curves[k[:3]][k[3]]=accuracy
    protocol_thresholds = [{'condition':k[0],'split':k[1],'seed':k[2],
        'all_four_budgets_complete':len(v)==4,
        'observed_budget_thresholds':{str(q):threshold(v,q) for q in (.8,.9,.95)},
        'interpretation':'Observed protocol budgets; hardware/precision may differ across cells; not global N*'}
        for k,v in sorted(protocol_curves.items()) if len(v)==4]
    protocol_der = []
    for k,v in sorted(protocol_curves.items()):
        baseline = protocol_curves.get(('model_only',k[1],k[2]),{})
        if k[0]!='model_only' and len(v)==len(baseline)==4:
            protocol_der.append({'condition':k[0],'split':k[1],'seed':k[2],
                'ratios':{str(q):der(threshold(baseline,q),threshold(v,q)) for q in (.8,.9,.95)},
                'interpretation':'Descriptive mixed-hardware protocol ratio; no isolated architectural effect'})
    protocol_aggregates=defaultdict(list)
    for cell in protocol_summary:
        if cell['complete']:
            protocol_aggregates[(cell['condition'],cell['split'],cell['budget'])].append(cell)
    protocol_seed_summary=[{'condition':k[0],'split':k[1],'budget':k[2],
        'seeds':[r['seed'] for r in v],'complete_three_seed_cell':len(v)==3,
        'mean_accuracy':float(np.mean([r['accuracy'] for r in v])),
        'seed_sd':float(np.std([r['accuracy'] for r in v],ddof=1)) if len(v)>1 else None,
        'interpretation':'Protocol-level summary; arithmetic/hardware composition is explicitly recorded per cell'}
        for k,v in sorted(protocol_aggregates.items())]
    lock_summary = []
    paired = []
    for c in ('model_only','structured','mindscape_b','mindscape_c'):
        for split in ('test','ood_test'):
            group = [r for r in locked if r['condition']==c and r['split']==split]
            if not group:
                continue
            n=len(group); correct=sum(r['success'] for r in group)
            transitions=[t for r in group for t in r['trajectory']['transitions']]
            lock_summary.append({'condition': c,'split': split,'episodes': n,'planned': 50,'complete': n==50,
                'nonempty_trajectories':sum(bool(r['trajectory']['transitions']) for r in group),
                'proposal_errors':sum(bool(a.get('parse_error')) for r in group for a in r['attempts']),
                'proposal_error_reasons':dict(Counter(a['parse_error'] for r in group for a in r['attempts'] if a.get('parse_error'))),
                'unparseable_proposals':sum(bool(a.get('parse_error')) and a.get('action') is None for r in group for a in r['attempts']),
                'successes': correct,'accuracy': correct/n,'wilson_ci95': wilson_interval(correct,n),
                'hidden_test_cases_passed':sum(r['terminal']['passed'] for r in group),
                'hidden_test_cases_total':sum(r['terminal']['total'] for r in group),
                'grounded_transition_rate': sum(t['valid'] for t in transitions)/len(transitions) if transitions else None,
                'trajectory_validity': sum(all(t['valid'] for t in r['trajectory']['transitions']) for r in group)/n,
                'mean_model_calls': sum(r['model_calls'] for r in group)/n,
                'mean_environment_calls': len(transitions)/n,
                'mean_generated_tokens': sum(a['latency'].get('tokens_out',0) for r in group for a in r['attempts'])/n})
    for split in ('test','ood_test'):
        baseline={r['task_id']:float(r['success']) for r in locked if r['condition']=='model_only' and r['split']==split}
        for c in ('structured','mindscape_b','mindscape_c'):
            other={r['task_id']:float(r['success']) for r in locked if r['condition']==c and r['split']==split}
            if len(baseline)==50 and set(baseline)==set(other):
                paired.append({'condition': c,'split': split,'delta': paired_delta(baseline,other)})
    replay = {'episodes':len(audit),'planned':400,
              'final_outcome_agreement':sum(r['matches_reported_goal'] for r in audit)/len(audit) if audit else None,
              'structured_evidence_agreement':sum(r['structured_evidence_matches'] for r in audit)/len(audit) if audit else None}
    for kind in ('state','tool_result'):
        denominator=sum(r.get(kind+'_comparisons',0) for r in audit)
        replay[kind+'_agreement']=sum(r.get(kind+'_matches',0) for r in audit)/denominator if denominator else None
        replay[kind+'_comparisons']=denominator
    latency = rows('latency_probe_v1')
    latency_source = 'Dedicated cache-disabled stage benchmark'
    if not latency:
        latency = rows('emergency_mps_v1') if (BASE/'emergency_mps_v1/rows.jsonl').exists() else []
        smoke_path = BASE/'emergency_mps_v1/smoke_rows.jsonl'
        if not latency and smoke_path.exists():
            latency = [json.loads(x) for x in smoke_path.read_text().splitlines()]
            latency_source = 'Reused actual cache-disabled representative MPS smoke benchmark; no additional neural run'
            for r in latency:
                r['stages'] = {
                    'model_generation':{'seconds':sum(a['latency']['generation'] for a in r['attempts'])},
                    'environment_execution':{'seconds':sum(t['result']['duration'] for t in r['trajectory']['transitions'])},
                    'verification':{'seconds':r['terminal']['execution']['duration']}}
    latency_summary=[]
    for c in ('model_only','structured','mindscape_b','mindscape_c'):
        group=[r for r in latency if r['condition']==c]
        if not group:
            continue
        fields={'total':[r['wall_seconds'] for r in group],
                'ttft':[a['latency']['ttft'] for r in group for a in r['attempts']],
                'response':[a['latency']['generation'] for r in group for a in r['attempts']]}
        for stage in ('model_generation','environment_execution','verification','test_execution'):
            fields[stage]=[r['stages'].get(stage,{}).get('seconds',0) for r in group]
        latency_summary.append({'condition': c,'episodes': len(group),'planned': 4 if latency_source.startswith('Dedicated') else 20,'measurement_source':latency_source,
            'device_precision_counts':dict(Counter(a['latency']['device']+'/'+a['latency']['precision']
                for r in group for a in r['attempts'])),
            'actual_batch_sizes':sorted({a['latency'].get('actual_batch_size',1) for r in group for a in r['attempts']}),
            'peak_process_rss_bytes':max((r.get('peak_rss_bytes',0) for r in group),default=0) or None,
            'peak_mps_live_bytes':max((r.get('mps_allocated_bytes',0) for r in group),default=0) or None,
            'peak_mps_driver_bytes':max((r.get('mps_driver_bytes',0) for r in group),default=0) or None,
            'sequential_episodes_per_hour':3600*len(group)/sum(r['wall_seconds'] for r in group)
                if latency_source.startswith('Dedicated') else None,
            'percentiles': {k:{'p50':float(np.percentile(v,50)),'p95':float(np.percentile(v,95)),'n':len(v)} for k,v in fields.items() if v},
            'caution': 'Descriptive sample; stage timers overlap; TTFT is per call, other fields per episode. Batched model durations are shared across requests; environment covers timed transitions, verification is measured sandbox execution; no state-construction timer in reused smoke evidence'})
    training_records = [{"path":str(p),**{k:v for k,v in load(p).items() if k!='task_ids'}}
                        for pattern in ('gradient_v1/seed_*/training.json','completion_gradient_v1/*/seed_*/training.json')
                        for p in sorted(BASE.glob(pattern))]
    tests=load(Path('experiments/coding_completion_v2/test_results.json'),{})
    fairness=load(BASE/'completion_audits_v1/fairness_leakage.json',{})
    claims={k:'INCONCLUSIVE' for k in ('high bounded-vertical capability','OOD improvement','data efficiency','grounded execution','trajectory validity','goal success','recovery','external benchmark competitiveness','memory benefit','dream benefit')}
    claims.update({'GPT reference exceedance':'NOT SUPPORTED','Gemini reference exceedance':'NOT SUPPORTED'})
    c_final=[r for r in lock_summary if r['condition']=='mindscape_c']
    if len(c_final)==2 and all(r['complete'] for r in c_final):
        claims['high bounded-vertical capability']='SUPPORTED' if all(r['accuracy']>=.9 for r in c_final) else 'NOT SUPPORTED'
        claims['goal success']='SUPPORTED' if all(r['accuracy']>=.9 for r in c_final) else 'NOT SUPPORTED'
        claims['trajectory validity']='SUPPORTED' if all(r['trajectory_validity']==1 for r in c_final) else 'NOT SUPPORTED'
    ood=next((p for p in paired if p['condition']=='mindscape_c' and p['split']=='ood_test'),None)
    if ood:
        ci=ood['delta']['ci95']; claims['OOD improvement']='SUPPORTED' if ci[0]>0 else ('NOT SUPPORTED' if ci[1]<=0 else 'INCONCLUSIVE')
    if audit and len(audit)==len(locked) and any(r['condition']=='mindscape_c' and r['complete'] for r in lock_summary):
        claims['grounded execution']='SUPPORTED' if replay['structured_evidence_agreement']==1 else 'NOT SUPPORTED'
    result={'learning_curve': {'completed': len(learning),'planned': 1920,'not_run': 1920-len(learning),'status': 'complete' if len(learning)==1920 else 'partial','cells': cell_summary,'complete_seed_summaries':complete_seed_summaries,'thresholds': thresholds,
        'protocol_cells':protocol_summary,'protocol_observed_thresholds':protocol_thresholds,
        'protocol_complete_seed_summaries':protocol_seed_summary,
        'DER': {'descriptive_matched_stratum_ratios':descriptive_der,
                'descriptive_protocol_ratios':protocol_der,'value':None,'status':'undefined',
                'reason':'No architecture-only estimate: hardware/precision and supervision differ; unreached thresholds are censored'},
        'zero_budget': 'not run; observed thresholds are not global minimum sample counts'},
        'lockbox': {'episodes': len(locked),'planned': 400,'not_run':400-len(locked),'scope':load(BASE/'completion_lockbox_eval_v1/scope.json'),'summary': lock_summary,'paired_differences': paired},
        'phase_memory_release':load(BASE/'emergency_mps_v1/phase_memory_release.json'),'mps_inference_verified':bool(locked) and all(a['latency'].get('device')=='mps' for r in locked for a in r['attempts']),'profile': profile,'native_device': load(BASE/'emergency_mps_v1/native_device.json'),'replay': replay,'latency': latency_summary,'latency_scope':load(BASE/'latency_probe_v1/scope.json',{'status':'Dedicated stage benchmark not run; reused real fresh representative smoke timings','source':latency_source}),
        'tests': tests,'audit_count': fairness.get('check_count'),'fairness': fairness,'claims': claims,
        'humaneval': {name:load(BASE/name/'summary.json') for name in ('humaneval_05b_v1','humaneval_15b_v1')},
        'SQL': 'Environment implemented; model study deferred','recovery': 'not run','ablations': 'memory/dream model ablations not run',
        'training_records':training_records,'parameters': 1543714304,'trainable_lora_parameters': 1089536,
        'neural_failure': load(BASE/'emergency_mps_v1/failure.json'),'deadline_utc': '2026-10-08 15:30:00 UTC'}
    result['continuation']=load(BASE/'research_continuation_v1/launch.json')
    continuation_complete=load(BASE/'research_continuation_v1/complete.json')
    if continuation_complete and result['continuation']:
        elapsed=continuation_complete['time_unix']-result['continuation']['started_unix']
        result['continuation_compute']={
            'elapsed_seconds_including_load_tools_and_sequential_latency':elapsed,
            'fresh_completed_episodes':104+798+16,
            'mixed_stage_episodes_per_hour':(104+798+16)*3600/elapsed,
            'interpretation':'Inclusive resumed workload, not isolated batched inference throughput'}
    resource_path=BASE/'research_continuation_v1/resources.jsonl'
    if resource_path.exists():
        resources=[json.loads(s) for s in resource_path.read_text().splitlines()]
        result['continuation_sampled_resources']={
            'samples':len(resources),'actual_batch_sizes':sorted({r['actual_batch_size'] for r in resources}),
            'sampled_peak_mps_live_bytes':max(r['mps_live_bytes'] for r in resources),
            'sampled_peak_mps_driver_bytes':max(r['mps_driver_bytes'] for r in resources),
            'peak_process_rss_bytes':max(r['peak_rss_bytes'] for r in resources),
            'interpretation':'Post-batch allocator samples; transient allocator peak not instrumented'}
    result['historical_emergency_deadline_utc']=result.pop('deadline_utc')
    result['historical_emergency_failure']=result.pop('neural_failure')
    result['continuation_failure']=load(BASE/'research_continuation_v1/failure.json')
    if result['continuation_failure'] and result['continuation_failure'].get('type')=='BlockingIOError':
        result['rejected_duplicate_launch']=result.pop('continuation_failure')
        result['continuation_failure']=None
    result['configuration']=load(BASE/'emergency_mps_v1/locked_protocol.json')
    result['configuration_sha256']=hashlib.sha256(
        (BASE/'emergency_mps_v1/locked_protocol.json').read_bytes()).hexdigest()
    result['operational_answer_metrics']={
        'definition':'Successful terminal program AND independent full-trajectory replay agreement; no semantic hallucination detector',
        'denominator':len(locked),
        'grounded_successful_answers':sum(r['success'] and any(
            a['key']==[r['condition'],r['task_id']] and a['structured_evidence_matches'] and a['matches_reported_goal']
            for a in audit) for r in locked),
        'terminal_incorrect_answers':sum(not r['success'] for r in locked),
        'semantic_hallucination_rate':None,
        'semantic_hallucination_status':'Not measured by this program-repair protocol'}
    answer_metrics=result['operational_answer_metrics']
    answer_metrics['grounded_successful_answer_rate']=(answer_metrics['grounded_successful_answers']/len(locked)) if locked else None
    answer_metrics['unsupported_by_terminal_tests_rate']=(answer_metrics['terminal_incorrect_answers']/len(locked)) if locked else None
    result['hardware_runtime']=load(Path('experiments/coding_completion_v2/host.json'),{})
    result['hardware_runtime']['mps_available_in_tool_sandbox']=result['hardware_runtime'].pop('mps_available',None)
    result['hardware_runtime']['native_mps_inference_verified']=result['mps_inference_verified']
    (OUT/'summary.json').write_text(json.dumps(result,indent=2))
    narrative='# Mindscape bounded MPS completion evidence\n\n'
    narrative+=f"Learning curve: **{len(learning)}/1,920 episodes**, {1920-len(learning)} not run. Lockbox: **{len(locked)}/400 condition-task episodes** (100 independent tasks). Preserved CPU work remains intact.\n\n"
    narrative+='All metrics below derive from saved episodes. Missing cells are not zero scores. CPU float32 and MPS reduced-precision strata are kept separate. Supervision and interaction budgets differ across conditions; architecture-only causality is not established.\n\n'
    narrative+='|Condition|Split|Completed/planned|Success|95% Wilson interval|\n|---|---|---:|---:|---|\n'
    for r in lock_summary:
        narrative+=f"|{r['condition']}|{r['split']}|{r['episodes']}/50|{r['accuracy']:.1%}|{r['wilson_ci95']}|\n"
    narrative+='\nObserved complete-cell sample thresholds and all partial cells are in `results/coding/emergency_analysis_v1/summary.json`. DER is undefined for a causal architecture comparison. No zero-budget control was run.\n\n'
    narrative+='Task-paired confidence intervals are descriptive and unadjusted for multiple comparisons. Transition validity is conditional on recorded actions; malformed model proposals are reported separately.\n\n'
    narrative+='Operational answer metrics: `'+json.dumps(answer_metrics)+'`. Failed terminal tests are unsupported program answers under this oracle; they are not a measured semantic hallucination rate.\n\n'
    narrative+='Original frozen configuration SHA-256: `'+result['configuration_sha256']+'`. Seeds: 11/23/37 for learning; seed 11 and budget 100 for lockbox. Exact checkpoint/source hashes are in the frozen protocol; host/runtime and training compute are in the machine-readable summary.\n\n'
    narrative+='## Actual acceleration measurements\n\n```json\n'+json.dumps({k:v for k,v in profile.items() if k not in ('comparisons','batch_trials','precision_trials')},indent=2)+'\n```\n\n'
    narrative+='One phase-boundary memory release: `'+json.dumps(result['phase_memory_release'])+'`.\n\n'
    narrative+='CPU reference episodes were previously measured with uncached inference; they were not rerun. MPS sec/episode is amortized throughput, distinct from per-episode latency while waiting for batches. Device tensors and model placement are checked; unsupported MPS operations fail with fallback disabled. Native allocation reported by the user is verified, but inference use is only established by native run outputs.\n\n'
    narrative+='## Replay and latency\n\n```json\n'+json.dumps({'replay':replay,'latency':latency_summary},indent=2)+'\n```\n\n'
    narrative+='## Preserved public benchmark\n\n0.5B: 94/164 (57.32%); 1.5B: 98/164 (59.76%). Greedy one-sample full-module generation, 512-token cap, CPython 3.14.7 WASI. Foundation pretraining exposure is unknown. These local results do not exceed historical GPT-4 67.0% or Gemini Ultra 74.4%, and protocols differ. See `docs/historical_references.md` for pinned primary sources.\n\n'
    narrative+=('Dedicated latency: '+str(len(latency))+'/16 fresh sequential cases; actual batch sizes and memory are reported above. Sequential response latency is distinct from frozen batch-16 evaluation throughput.\n\n'
                if latency_source.startswith('Dedicated') and latency else
                'Dedicated post-lockbox latency is not complete. Any reused smoke measurements are labeled explicitly.\n\n')
    narrative+=f"Complete suite: {tests}. Local fairness/leakage assertions: {fairness.get('check_count')}, {fairness.get('local_leakage_status')}. Contamination-free status remains unknown. All 56 retained adapters validated. SQL environment implemented; SQL model study deferred. Recovery and memory/dream ablations not run; their claims remain inconclusive.\n"
    historical = Path('docs/final_results.md').read_text().split('\n<!-- emergency-mps-results -->')[0]
    Path('docs/final_results.md').write_text(historical+'\n<!-- emergency-mps-results -->\n\n'+narrative)
    Path('docs/final_metrics.md').write_text('# Actual completion metrics\n\nMachine-readable full metrics: `results/coding/emergency_analysis_v1/summary.json`.\n\n'+narrative)
    Path('docs/final_claims.md').write_text('# Claims and evidence\n\n|Claim|Status|\n|---|---|\n'+''.join(f'|{k}|{v}|\n' for k,v in claims.items())+'\nCapability/goal claims use a descriptive 90% threshold (reporting criterion, not a preregistered hypothesis test) on both locked splits. OOD improvement requires a positive task-paired 95% CI against A. Grounded execution requires all locked independent replay evidence to match. Grounding is established only for saved re-executed episodes. No architecture-only causal claim is supported.\n')
    c_count = sum(r['condition']=='mindscape_c' for r in locked)
    Path('docs/final_limitations.md').write_text(f'# Completion limitations\n\nFlagship C locked episodes: {c_count}/100. Lockbox: {len(locked)}/400. Learning: {len(learning)}/1920. Dedicated latency: {len(rows("latency_probe_v1"))}/16. Counts describe actual saved evidence; completion requires every mandatory scope. The historical emergency package preserves its cutoff and failures. The authorized continuation removes that operational cutoff without changing the frozen scientific protocol.\n\nMixed CPU float32/MPS reduced precision; missing zero control; unmatched supervision and inference interaction budgets; synthetic bounded tasks; unknown foundation pretraining exposure; no learned world model; no recovery or component ablation results; tiny descriptive latency sample; public protocol mismatch; SQL model study deferred. Unmeasured results stay inconclusive. Hardware-only speedup is not isolated from batching and precision.\n')
    Path('docs/reproducibility.md').write_text('# Reproduce preserved evidence\n\nUse the pinned foundation snapshots and WASI runtime described in `docs/coding/reproducibility.md`. The emergency native entry point is `scripts/coding/emergency_mps.py`; its original deadline is fixed and it intentionally refuses late neural reruns. Saved per-episode files checkpoint completed keys; completed CPU episodes are never deleted. Explicit historical CPU reproduction uses `MINDSCAPE_DEVICE=cpu MINDSCAPE_PRECISION=float32`; accelerated evaluation requires MPS and rejects silent fallback. Filesystem, tokenizer serialization, test execution, checkpoint CPU backups and Git remain CPU utilities. Neural weights, forward/generation and LoRA tensors use MPS. Batch only independent requests across episodes; decisions within each episode remain sequential.\n\nRecompute reports with `work/final-venv/bin/python scripts/coding/emergency_report.py`. Restore frozen evidence into an empty directory with `scripts/coding/restore_completion.py --snapshot results/final/coding_research_v2 --destination PATH`; use the actual snapshot name if the mandatory locked study remained partial. The restore utility verifies every manifest hash before copying. Foundation weights are external pinned downloads, not bundled; adapters, protocol, actual datasets, source, runtime and results are bundled.\n')
    with Path('docs/reproducibility.md').open('a') as stream:
        stream.write('\nThe authorized full continuation is `scripts/coding/complete_research.py`; it removes the superseded operational cutoff, verifies the original protocol/source/checkpoint hashes, holds a single-runner kernel lock and resumes exact missing keys. See `docs/research_continuation.md`. The original emergency deadline and failure records are historical.\n')
    print('Reported actual learning/locked episodes',len(learning),len(locked))


if __name__=='__main__':
    main()
