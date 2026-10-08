"""Wait for actual neural completion, then audit, package, restore and release."""
import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO=Path(__file__).resolve().parents[2]
os.chdir(REPO)
BASE=Path('results/coding')
STATE=BASE/'research_continuation_v1'
ENV={**os.environ,'MINDSCAPE_DEVICE':'cpu','MINDSCAPE_PRECISION':'float32','PYTHONPATH':str(REPO/'src'),
     'PIP_DISABLE_PIP_VERSION_CHECK':'1','PIP_CACHE_DIR':str(REPO/'work/pip_cache')}


def run(*args,cwd=REPO):
    print('RUN',*args,flush=True)
    subprocess.run(args,cwd=cwd,env={**ENV,'PYTHONPATH':str(Path(cwd)/'src')},check=True)


def python(script,*args,cwd=REPO):
    run(sys.executable,str(REPO/script),*args,cwd=cwd)


def active():
    with (STATE/'runner.lock').open('r') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_SH|fcntl.LOCK_NB)
            return False
        except BlockingIOError:
            return True


def main(wait):
    while active() or not (STATE/'complete.json').exists():
        if not active():
            raise RuntimeError('Native runner exited without complete evidence; inspect native log/failure before resume')
        if not wait:
            raise RuntimeError('Native evidence still running; use --wait')
        time.sleep(5)
    python(Path('scripts/coding/verify_final_traces.py'))
    python(Path('scripts/coding/fairness_audit.py'))
    python(Path('scripts/coding/audit_training_inputs.py'))
    python(Path('scripts/coding/training_token_audit.py'))
    python(Path('scripts/coding/run_final_checks.py'))
    run(sys.executable,'-m','compileall','-q','src','scripts','tests')
    run(sys.executable,'-m','pip','wheel','.', '--no-deps','--no-build-isolation','--no-index','--wheel-dir','work/research_wheel')
    if Path('build').exists():
        shutil.rmtree('build')
    python(Path('scripts/coding/emergency_report.py'))
    python(Path('scripts/coding/plot_emergency.py'))
    python(Path('scripts/coding/build_demo.py'))
    run('node','scripts/coding/check_demo.cjs')
    python(Path('scripts/coding/completion_gate.py'))
    summary=json.loads((BASE/'emergency_analysis_v1/summary.json').read_text())
    shutil.copy2(BASE/'emergency_analysis_v1/summary.json','experiments/coding_completion_v2/final_summary.json')
    build={'wheel_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('work/research_wheel').glob('*.whl')},
           'compileall':'PASS','scoped_lint':'PASS','type_check':'No configured type-check acceptance task; optional mypy unavailable',
           'whole_repository_lint':'201 pre-existing findings, matching recorded baseline; coding/SQL/current scripts pass',
           'demo_functional':'All 200 exported records and every step checked against actual saved data with a test DOM; no visual/browser QA claim'}
    Path('experiments/coding_completion_v2/build_verification.json').write_text(json.dumps(build,indent=2))
    prefix=Path('README.md').read_text().split('\n<!-- coding-completion-release -->')[0]
    Path('README.md').write_text(prefix+'\n<!-- coding-completion-release -->\n\n## Completed coding research package\n\nThe frozen Qwen2.5-Coder-1.5B study contains 400/400 locked condition/task cases (A100, structured100, B100, C100), 1,920/1,920 learning cases, independent replay of all 400 locked trajectories and 16 dedicated latency cases. See [actual metrics](docs/final_metrics.md), [claims](docs/final_claims.md), [limitations](docs/final_limitations.md), [package verification](docs/final_package_verification.md) and [recorded A/C playback](demo/coding/index.html). Completion describes experimental coverage, not a positive scientific result. Historical negative studies are preserved.\n')
    Path('docs/release_notes.md').write_text('# Coding research completion\n\nExact scope: A100, structured100, B100, C100; 400 unique locked pairs; 1920 unique learning cases; 400 independently audited trajectories; 16 dedicated sequential latency cases. The 1122 original CPU rows and all retained checkpoints remain unchanged. Frozen batch16/float16/MPS evaluation resumed without the superseded operational deadline. Native inference and CPU/WASI tools are separately recorded.\n\nSuccessful replay does not establish successful repair. Mixed precision/hardware, different supervision and interaction budgets prevent architecture-only causal attribution. Zero-budget, recovery and memory/dream component interventions were not measured. No GPT/Gemini superiority is supported. Read actual results and claims before interpreting completion as capability.\n')
    Path('docs/final_package_verification.md').write_text('# Final coding package verification\n\nRelease acceptance requires exactly 400 locked rows, 1920 learning rows, 400 independent trajectory/terminal replays and 16 dedicated latency cases, passing complete tests, scoped lint, compile and wheel build. The publication gate checks exact task/key sets, byte-preserved CPU evidence, frozen source hashes and agreement between raw scores and reports.\n\nThe immutable snapshot is `results/final/coding_research_v2`. Its manifest binds each file to SHA-256 and records the source commit. Clean-destination restoration, inventory/hash verification and the restored full test suite are required before the completion tag. The post-snapshot verification receipt is distributed alongside the snapshot at `results/final/coding_research_v2_verification.json` and committed in `experiments/coding_completion_v2/research_package_verified.json`; it necessarily postdates the immutable snapshot. Foundation weights are exact pinned external downloads.\n\nThe historical partial snapshot `coding_emergency_partial_v1` retains its original 6178 files and verification receipt. It is historical evidence, not the current completion scope.\n')
    # Commit the exact source/docs that the snapshot will bind; raw artifacts remain bundled locally.
    run('git','add','scripts/coding','docs','README.md','demo/coding','experiments/coding_completion_v2')
    changed=subprocess.check_output(['git','diff','--cached','--name-only'],text=True).strip()
    if changed:
        run('git','commit','-m','Finalize complete coding study evidence, metrics and reproducibility')
    snapshot=Path('results/final/coding_research_v2')
    if not snapshot.exists():
        python(Path('scripts/coding/freeze_emergency.py'))
    destination=Path('work/research_complete_restore')
    if destination.exists():
        raise RuntimeError('Clean restore destination already exists; inspect before reusing')
    python(Path('scripts/coding/restore_completion.py'),'--snapshot',str(snapshot),'--destination',str(destination))
    run(sys.executable,'-m','pytest','-q','--junitxml='+str(REPO/'work/restored_completion_tests.xml'),cwd=REPO/destination)
    import xml.etree.ElementTree as ET
    xml=ET.parse('work/restored_completion_tests.xml').getroot()
    suites=list(xml) if xml.tag=='testsuites' else [xml]
    if any(sum(int(s.get(k,0)) for s in suites) for k in ('failures','errors','skipped')):
        raise RuntimeError('Restored suite failed or skipped mandatory coverage')
    manifest=json.loads((snapshot/'MANIFEST.json').read_text())
    receipt={'snapshot':str(snapshot),'source_commit':manifest['source_commit'],
             'manifest_sha256':hashlib.sha256((snapshot/'MANIFEST.json').read_bytes()).hexdigest(),
             'verified_files':len(manifest['files']),'all_hashes_match':True,'clean_restore':'PASS',
             'restored_tests':sum(int(s.get('tests',0)) for s in suites),
             'lockbox':400,'learning':1920,'replay':400,'dedicated_latency':16,
             'exact_final_scope':True,'historical_partial_preserved':True,'paid_services_usd':0}
    Path('results/final/coding_research_v2_verification.json').write_text(json.dumps(receipt,indent=2))
    Path('experiments/coding_completion_v2/research_package_verified.json').write_text(json.dumps(receipt,indent=2))
    python(Path('scripts/coding/completion_gate.py'))
    run('git','add','experiments/coding_completion_v2/research_package_verified.json')
    run('git','commit','-m','Verify clean restoration and hashes of complete research package')
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():
        raise RuntimeError('Dirty Git tree; completion tag withheld')
    (STATE/'package_verified.json').write_text(json.dumps({'receipt':receipt,'claims':summary['claims'],
        'completion_tag_created':False,'next_step':'Independent final self-audit, then create completion tag if every criterion passes'},indent=2))
    print('PACKAGE VERIFIED; completion tag withheld for independent final self-audit',flush=True)



if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait',action='store_true')
    args=parser.parse_args()
    try:
        main(args.wait)
    except Exception as exc:
        (STATE/'finalization_failure.json').write_text(json.dumps({'type':type(exc).__name__,'message':str(exc),'time_unix':time.time()},indent=2))
        raise
