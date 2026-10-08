"""Strict publication gate: exact frozen evidence keys, replay and measured latency."""
import hashlib
import json
from pathlib import Path

BASE = Path('results/coding')
CONDITIONS = ('model_only','structured','mindscape_b','mindscape_c')


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def exact(actual, expected, label):
    if len(actual)!=len(set(actual)) or set(actual)!=expected:
        raise RuntimeError(f'{label}: duplicate, missing or unexpected keys')


def verify():
    task_path=BASE/'completion_lockbox_v1/tasks.json'
    tasks=json.loads(task_path.read_text())
    dataset=json.loads((BASE/'final_dataset_v1/dataset.json').read_text())
    locked_path=BASE/'completion_lockbox_eval_v1/rows.jsonl'
    locked=rows(locked_path)
    locked_keys={(c,t['task_id']) for c in CONDITIONS for t in tasks}
    exact([(r['condition'],r['task_id']) for r in locked],locked_keys,'Locked evidence')
    learning_path=BASE/'completion_gradient_v1/rows.jsonl'
    learning=rows(learning_path)
    learning_keys={(c,s,n,split,t['task_id']) for c in CONDITIONS for s in (11,23,37)
                   for n in (10,25,50,100) for split in ('test','ood_test') for t in dataset[split]}
    exact([tuple(r['key']) for r in learning],learning_keys,'Learning evidence')
    original_prefix=b''.join(learning_path.read_bytes().splitlines(keepends=True)[:1122])
    if hashlib.sha256(original_prefix).hexdigest()!='bcf9a9c0f0a21e8c00a30661db48bcca9b45ff11175172a3a8457c7e5f64f49a':
        raise RuntimeError('Preserved CPU evidence changed')
    audit_path=BASE/'completion_trace_audit_v1/rows.jsonl'
    audit=rows(audit_path)
    exact([tuple(r['key']) for r in audit],locked_keys,'Independent replay')
    if not all(r['structured_evidence_matches'] and r['matches_reported_goal'] for r in audit):
        raise RuntimeError('Independent replay disagreement')
    # Bind legacy successful replay to the byte-identical original episodes.
    preservation=json.loads(Path('experiments/coding_completion_v2/locked_preservation.json').read_text())
    current={(r['condition'],r['task_id']):r for r in locked}
    hashes={k:hashlib.sha256(json.dumps(r,sort_keys=True).encode()).hexdigest() for k,r in current.items()}
    if not all(hashes[tuple(r['key'])]==r['canonical_sha256'] for r in preservation['rows']):
        raise RuntimeError('Previously audited locked evidence changed')
    prefix=b''.join(audit_path.read_bytes().splitlines(keepends=True)[:296])
    if hashlib.sha256(prefix).hexdigest()!=preservation['original_audit_prefix_sha256']:
        raise RuntimeError('Preserved independent replay changed')
    for path,digest in preservation['checkpoint_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Preserved checkpoint changed: '+path)
    old_keys={tuple(r['key']) for r in preservation['rows']}
    if not all(tuple(r['key']) in old_keys or r.get('canonical_episode_sha256')==hashes[tuple(r['key'])] for r in audit):
        raise RuntimeError('New replay is not bound to the actual saved episode')
    latency_path=BASE/'latency_probe_v1/rows.jsonl'
    latency=rows(latency_path)
    reps={dataset[split][i]['task_id'] for i in range(2) for split in ('test','ood_test')}
    exact([(r['condition'],r['task_id']) for r in latency],{(c,t) for c in CONDITIONS for t in reps},'Dedicated latency')
    if not all(r['gpu_used'] and not r['cache_enabled'] and r['stages']
               and all(a['latency']['device']=='mps' and not a['latency'].get('cache_hit',False)
                       for a in r['attempts']) for r in latency):
        raise RuntimeError('Dedicated latency device/cache evidence invalid')
    protocol_path=BASE/'emergency_mps_v1/locked_protocol.json'
    protocol=json.loads(protocol_path.read_text())
    if hashlib.sha256(task_path.read_bytes()).hexdigest()!=protocol['lockbox_sha256']:
        raise RuntimeError('Lockbox changed')
    for path,digest in protocol['source_hashes'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Frozen scientific source changed: '+path)
    tests=json.loads(Path('experiments/coding_completion_v2/test_results.json').read_text())
    if not tests.get('tests') or any(tests.get(k,1) for k in ('failures','errors','skipped')):
        raise RuntimeError('Complete test suite did not pass')
    fairness=json.loads((BASE/'completion_audits_v1/fairness_leakage.json').read_text())
    if fairness.get('local_leakage_status')!='PASS':
        raise RuntimeError('Local fairness audit failed')
    summary=json.loads((BASE/'emergency_analysis_v1/summary.json').read_text())
    if summary['lockbox']['episodes']!=400 or summary['learning_curve']['completed']!=1920 or summary['replay']['episodes']!=400:
        raise RuntimeError('Analysis is stale')
    if sum(r['episodes'] for r in summary['latency'])!=16 or not all(
            r['measurement_source']=='Dedicated cache-disabled stage benchmark' for r in summary['latency']):
        raise RuntimeError('Analysis does not use dedicated latency')
    for cell in summary['lockbox']['summary']:
        values=[r for r in locked if r['condition']==cell['condition'] and r['split']==cell['split']]
        if cell['episodes']!=50 or cell['successes']!=sum(r['success'] for r in values):
            raise RuntimeError('Analysis score mismatch')
    for name in ('docs/final_metrics.md','docs/final_results.md'):
        text=Path(name).read_text()
        if '400/400' not in text or '1920/1,920' not in text:
            raise RuntimeError('Report counts do not agree: '+name)
    return {'lockbox':400,'learning':1920,'dedicated_latency':16,'independent_replay':400,
            'preserved_cpu_rows':1122,'preserved_adapter_files':len(preservation['checkpoint_sha256']),'tests':tests,
            'configuration_sha256':hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
            'evidence_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (locked_path,learning_path,audit_path,latency_path)},
            'all_mandatory_evidence_passed':True}


if __name__=='__main__':
    receipt=verify()
    Path('experiments/coding_completion_v2/research_evidence_verified.json').write_text(json.dumps(receipt,indent=2))
    print('PASS: 400 locked / 1920 learning / 400 replay / 16 dedicated latency')
