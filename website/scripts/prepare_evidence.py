"""Export a bounded public view of verified research; never alter raw evidence."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'website/assets'
PIN = 'c530faa20aee2339679a5f2cfe9bea28509d4fb1'
GH = 'https://github.com/akilanpl/mindscape'


def load(path):
    return json.loads((ROOT/path).read_text())


def rows(path):
    return [json.loads(x) for x in (ROOT/path).read_text().splitlines()]


v1 = load('results/coding/emergency_analysis_v1/summary.json')
old = load('experiments/research_v2/cloud_runs/37946027443/mechanism_study/statistics.json')
new = load('experiments/research_v2/gap_closure/statistics.json')
manifest = load('experiments/research_v2/release_manifest.json')
coverage = {n: len(rows(f'results/coding/{folder}/rows.jsonl')) for n,folder in
            [('learning','completion_gradient_v1'),('lockbox','completion_lockbox_eval_v1'),('audits','completion_trace_audit_v1'),('latency','latency_probe_v1')]}
assert coverage == {'learning':1920,'lockbox':400,'audits':400,'latency':16}
assert old['completed']==1920 and old['executed_unsuccessful']==734
assert new['completed']==288 and new['unsuccessful']==85
assert manifest['post_hoc_conservative95_interval']==[.666228580390205,1.0]
source_paths=['experiments/research_v2/release_manifest.json','experiments/research_v2/gap_closure/statistics.json','docs/final_metrics.md','docs/research_v2/gap_closure.md']
sources=[{'path':p,'url':f'{GH}/blob/{PIN}/{p}','sha256':hashlib.sha256((ROOT/p).read_bytes()).hexdigest()} for p in source_paths]
evidence={'researchCommit':PIN,'v1Commit':'2862f5ae54443b231abda4255bdd6c652eca2343','v1Tag':'mindscape-research-complete-v1','coverage':coverage,
 'v2':{'originalCases':old['completed'],'originalFailures':old['executed_unsuccessful'],'newCases':new['completed'],'newFailures':new['unsuccessful'],'effect':manifest['descriptive_paired_effect'],'interval':manifest['post_hoc_conservative95_interval'],'holmSignificant':manifest['primary_holm_significant_tests'],'codingIndexed':2320},
 'lockbox':[{k:r[k] for k in ('condition','split','successes','episodes','accuracy','wilson_ci95')} for r in v1['lockbox']['summary']],
 'latency':[{'condition':r['condition'],'n':r['episodes'],'p50':r['percentiles']['total']['p50'],'p95':r['percentiles']['total']['p95'],'rss':r['peak_process_rss_bytes'],'mpsLive':r['peak_mps_live_bytes']} for r in v1['latency']],
 'controls':new['cells'],'effects':new['effects'],'sources':sources,
 'historical':[{'name':'GPT historical reference','status':'Not yet measured','detail':'Exact release, access conditions and authorized budget remain unspecified.'},{'name':'Gemini historical reference','status':'Not yet measured','detail':'No matched experiment has been executed.'}],
 'archive':{'url':f'{GH}/raw/{PIN}/research_checkpoint/research_v2_gap_closure_2026-10-09/evidence.tar.gz','bytes':manifest['archive_bytes'],'sha256':manifest['archive_sha256']}}
OUT.mkdir(exist_ok=True)
(OUT/'evidence.json').write_text(json.dumps(evidence,separators=(',',':'))+'\n')
traces=[]
for name in ('coding_probe_v2','coding_probe_c'):
    t=load(f'results/research_v2/gap_closure/{name}/trace.json')
    events=[]
    for e in t['events']:
        if e['kind'] in ('model_request','model_response','deterministic_component','deterministic_environment_transition'):
            events.append(e)
        elif e['kind']=='actual_forward_output' and (not events or events[-1]['kind']!='actual_forward_output'):
            events.append(e)
        elif e['kind']=='actual_token_input' and len(e['observed']['token_ids'][0])>1:
            events.append(e)
    traces.append({'label':'Model-only diagnostic' if name=='coding_probe_v2' else 'Mindscape C diagnostic',
      'taskId':t['task_id'],'model':t['model_revision'],'backend':t['backend'],'precision':t['precision'],'success':t['result']['success'],
      'calls':t['result']['model_calls'],'eventCount':len(t['events']),'elapsed':t['elapsed_seconds_including_load'],'rss':t['peak_process_rss_bytes'],
      'events':events,'output':t['result']['repository'],'failures':[a['parse_error'] for a in t['result']['attempts']],
      'sourceSha256':hashlib.sha256((ROOT/f'results/research_v2/gap_closure/{name}/trace.json').read_bytes()).hexdigest()})
public=json.dumps(traces,separators=(',',':'))
assert '/Users/' not in public and 'checkpoint_sha256' not in public
(OUT/'traces.json').write_text(public+'\n')
print('Verified public evidence export: eight lockbox cells,16latency cases, two actual trace views')
