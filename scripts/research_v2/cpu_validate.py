"""CPU-only saved-evidence validation; never launches benchmark inference."""
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    evidence, output = args.evidence_root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_path = REPO/'configs/research_v2/cpu_validation.json'
    config = json.loads(config_path.read_text())
    raw_paths = [evidence/'results/coding'/name/'rows.jsonl' for name in (
        'completion_lockbox_eval_v1', 'completion_gradient_v1',
        'completion_trace_audit_v1', 'latency_probe_v1')]
    pins = {str(p.relative_to(evidence)): digest(p) for p in raw_paths}
    packages = dict(sorted((d.metadata['Name'].lower(), d.version)
                           for d in importlib.metadata.distributions() if d.metadata['Name']))
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    identity = {'commit': commit, 'script_sha256': digest(Path(__file__)),
                'config_sha256': digest(config_path), 'raw_sha256': pins,
                'packages': packages, 'python': platform.python_version()}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    env = {**os.environ, 'PYTHONPATH': str(REPO/'src'), 'MINDSCAPE_DEVICE': 'cpu',
           'MINDSCAPE_PRECISION': 'float32', 'MPLBACKEND': 'Agg'}
    with (output/'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prior = output/'environment.json'
        if prior.exists() and json.loads(prior.read_text())['fingerprint'] != fingerprint:
            raise RuntimeError('Output namespace belongs to different inputs/runtime; use a new namespace')
        import torch
        if torch.cuda.is_available():
            raise RuntimeError('CPU validation does not authorize GPU computation')
        if not prior.exists():
            write(prior, {'fingerprint': fingerprint, **identity, 'platform': platform.platform(),
                         'host': platform.node(), 'cpu_count': os.cpu_count(),
                         'backend': 'cpu', 'torch': torch.__version__, 'started_unix': time.time(),
                         'scope': 'Saved-evidence analysis/audit/tests; tiny learned unit-test model only'})
        def step(name, action):
            receipt = output/(name+'.json')
            if receipt.exists():
                saved = json.loads(receipt.read_text())
                if saved['fingerprint'] != fingerprint or not all(
                    digest(output/p) == h for p, h in saved['output_sha256'].items()):
                    raise RuntimeError('Resume receipt/output mismatch: '+name)
                print('RESUME verified', name, flush=True)
                return
            started = time.time()
            result, files = action()
            write(receipt, {'fingerprint': fingerprint, 'exit_code': 0, 'result': result,
                            'elapsed_seconds': time.time()-started,
                            'output_sha256': {str(p.relative_to(output)): digest(p) for p in files}})
            print('PASS', name, flush=True)
        def command(name, argv):
            logfile = output/(name+'.log')
            with logfile.open('w') as stream:
                result = subprocess.run(argv, cwd=evidence, env=env, stdout=stream,
                                        stderr=subprocess.STDOUT, check=False)
            if result.returncode:
                raise RuntimeError(f'{name} exit {result.returncode}; see {logfile}')
            return {'command': argv, 'exit_code': result.returncode}, [logfile]
        def integrity():
            old = Path.cwd()
            sys.path.insert(0, str(REPO/'scripts/coding'))
            os.chdir(evidence)
            try:
                from completion_gate import verify
                result = verify()
            finally:
                os.chdir(old)
            for k, n in config['required_counts'].items():
                assert result[k if k != 'replay' else 'independent_replay'] == n
            return result, []
        step('integrity', integrity)
        def statistics():
            rows = [json.loads(s) for s in raw_paths[1].read_text().splitlines()]
            cells = {}
            for row in rows:
                if type(row['success']) is not bool:
                    raise RuntimeError('Invalid success outcome')
                key = tuple(row[k] for k in ('condition', 'seed', 'budget', 'split'))
                cells.setdefault(key, []).append(row)
            summary = json.loads((evidence/'experiments/coding_completion_v2/final_summary.json').read_text())
            for cell in summary['learning_curve']['protocol_cells']:
                values = cells[tuple(cell[k] for k in ('condition', 'seed', 'budget', 'split'))]
                assert len(values) == cell['episodes'] == 20
                assert sum(r['success'] for r in values)/20 == cell['accuracy']
            path = output/'baseline_statistics.json'
            write(path, {'cells_verified': len(cells), 'episodes': len(rows),
                         'successes': sum(r['success'] for r in rows),
                         'claims': summary['claims'], 'scope': 'Recount of existing frozen evidence; no new inference'})
            return {'verified_cells': len(cells)}, [path]
        step('statistics', statistics)
        def tests():
            xml = output/'tests.xml'
            result, files = command('tests', [sys.executable, '-m', 'pytest', '-q',
                                              str(REPO/'tests'), '--junitxml='+str(xml)])
            tree = ET.parse(xml).getroot()
            suites = list(tree) if tree.tag == 'testsuites' else [tree]
            counts = {k: sum(int(s.get(k, 0)) for s in suites)
                      for k in ('tests', 'failures', 'errors', 'skipped')}
            if not counts['tests'] or any(counts[k] for k in ('failures', 'errors', 'skipped')):
                raise RuntimeError('Mandatory tests failed or skipped')
            return {**result, **counts}, files+[xml]
        step('tests', tests)
        step('fairness', lambda: command('fairness', [sys.executable, str(REPO/'scripts/coding/fairness_audit.py')]))
        step('frontend', lambda: command('frontend', ['node', str(REPO/'scripts/coding/check_demo.cjs')]))
        step('lint', lambda: command('lint', [sys.executable, '-m', 'ruff', 'check',
                                              str(REPO/'scripts/research_v2')]))
        if {str(p.relative_to(evidence)): digest(p) for p in raw_paths} != pins:
            raise RuntimeError('Frozen evidence mutated during validation')
        write(output/'complete.json', {'fingerprint': fingerprint, 'status': 'PASS',
                                      'completed_unix': time.time(), 'frozen_raw_unchanged': True,
                                      'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss *
                                      (1024 if platform.system() == 'Linux' else 1),
                                      'steps': ['integrity', 'statistics', 'tests', 'fairness', 'frontend', 'lint']})


if __name__ == '__main__':
    main()
