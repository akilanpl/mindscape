"""Freeze verified evidence, explicitly distinguishing mandatory completion from partial scope."""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--partial',action='store_true',help='Preserve an explicitly incomplete emergency package; no research-v2 release claim')
a=parser.parse_args()
base=Path('results/coding')
summary=json.loads((base/'emergency_analysis_v1/summary.json').read_text())
tests=summary['tests']
if tests.get('failures',1) or tests.get('errors',1):
    raise RuntimeError('Passing complete tests required')
primary=[r for r in summary['lockbox']['summary'] if r['condition'] in ('model_only','mindscape_c')]
complete=(len(primary)==4 and all(r['complete'] for r in primary)
          and summary['replay']['episodes']==summary['lockbox']['episodes']
          and summary['replay']['final_outcome_agreement']==1
          and summary['replay']['structured_evidence_agreement']==1
          and any(r['condition']=='mindscape_c' and r['episodes']>=2 for r in summary['latency']))
if not complete and not a.partial:
    raise RuntimeError('Mandatory locked/replay/fresh-latency evidence incomplete; use explicit partial package only')
name='coding_emergency_partial_v1' if a.partial else 'coding_research_v2'
root=Path('results/final')/name
staging=root.parent/('.'+name+'_staging')
if root.exists() or staging.exists():
    raise RuntimeError('Immutable snapshot or interrupted staging already exists; inspect before proceeding')
sources={
    'dataset':base/'final_dataset_v1','dataset_validation':base/'final_dataset_validation_v1',
    'lockbox':base/'completion_lockbox_v1','lockbox_validation':base/'completion_lockbox_validation_v1',
    'patch_training':base/'gradient_v1','gradient_study':base/'completion_gradient_v1',
    'actual_collection':base/'completion_collection_v1','actual_tuples':base/'teacher_traces_v1',
    'teacher_memory':base/'final_retrieval_v1','final_evaluation':base/'completion_lockbox_eval_v1',
    'fresh_latency':base/'latency_probe_v1','trace_audit':base/'completion_trace_audit_v1',
    'analysis':base/'emergency_analysis_v1','acceleration':base/'emergency_mps_v1',
    'audit':base/'completion_audits_v1','public_benchmark_input':Path('work/coding/humaneval'),
    'humaneval_05b':base/'humaneval_05b_v1','humaneval_15b':base/'humaneval_15b_v1',
    'configs':Path('configs'),'scripts':Path('scripts'),'docs':Path('docs'),
    'experiments':Path('experiments'),'tests':Path('tests'),'wasi_runtime':Path('work/coding/runtime')}
if Path('demo/coding').exists():
    sources['demo']=Path('demo/coding')
sources={label:path for label,path in sources.items() if path.exists()}
staging.mkdir(parents=True)
for label,path in sources.items():
    shutil.copytree(path,staging/label,ignore=shutil.ignore_patterns('__pycache__','.DS_Store'))
shutil.copytree('src/mindscape',staging/'source/mindscape',ignore=shutil.ignore_patterns('__pycache__'))
for path in ('pyproject.toml','README.md'):
    shutil.copy2(path,staging/path)
files={str(p.relative_to(staging)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(staging.rglob('*')) if p.is_file()}
manifest={'release':'emergency-partial-no-final-tag' if a.partial else 'mindscape-research-v2',
          'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
          'files':files,'source_paths':{label:str(path) for label,path in sources.items()},
          'learning_curve_scope':summary['learning_curve']['status'],
          'completed_learning_episodes':summary['learning_curve']['completed'],
          'skipped_learning_episodes':summary['learning_curve']['not_run'],
          'mandatory_evidence_complete':complete,
          'four_condition_lockbox_complete':summary['lockbox']['episodes']==400,
          'locked_completed_episodes':summary['lockbox']['episodes'],
          'locked_not_run_episodes':400-summary['lockbox']['episodes'],
          'latency_completed_episodes':sum(r['episodes'] for r in summary['latency']),
          'scope':'Timed emergency completion: mandatory C and A on all 100 locked tasks; supplementary control/latency scopes explicitly disclosed',
          'foundation_models':{'Qwen/Qwen2.5-Coder-0.5B-Instruct':'ea3f2471cf1b1f0db85067f1ef93848e38e88c25',
                               'Qwen/Qwen2.5-Coder-1.5B-Instruct':'2e1fd397ee46e1388853d2af2c993145b0f1098a'},
          'foundation_weights':'External exact pinned downloads; not included in this evidence snapshot',
          'historical_release':'mindscape-final-study-v1 retained','paid_services_usd':0}
(staging/'MANIFEST.json').write_text(json.dumps(manifest,indent=2))
staging.rename(root)
print('Frozen',len(files),'files at',root)
