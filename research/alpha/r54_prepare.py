"""Read-only Alpha readiness: re-audit existing evidence; never invoke science."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]


def prepare(qualification):
    import torch

    # Separate process uses the frozen review launcher; no execution flag is exposed.
    command = [sys.executable, str(REPO / 'integration/r5_4_gate_v2/launcher.py'),
               '--qualification', str(Path(qualification).absolute())]
    process = subprocess.run(command, cwd=REPO, capture_output=True, text=True, check=True, timeout=60)
    review = json.loads(process.stdout)
    if (review['mode'] != 'NONTRAINING_REVIEW_ONLY'
            or review['scientific_optimizer_updates'] != 0 or review['neural_ood_predictions'] != 0):
        raise ValueError('Read-only review boundary violated')
    proof = review['proof']
    protocol_path = REPO / 'experiments/r5_4/frozen/protocol.json'
    protocol = json.loads(protocol_path.read_text())
    runtime = {'torch': str(torch.__version__), 'machine': platform.machine(),
               'platform': platform.platform(), 'mps_available': torch.backends.mps.is_available()}
    compatible = all(runtime[key] == proof['qualified_runtime'][key] for key in runtime)
    return {'status': 'UNRUN', 'engineering_status': proof['qualification_status'],
            'integration_status': review['status'], 'independent_engineering_audit': proof['independent_audit'],
            'gate_adopted': False, 'scientific_training_authorized': False,
            'disposable_engineering_steps': proof['disposable_steps'], 'scientific_updates': 0,
            'neural_ood_predictions': 0, 'observed_runtime': runtime, 'qualified_runtime_match': compatible,
            'execution_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
            'protocol_sha256': hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
            'source_locks': {key: proof[key] for key in ('v1_lock_sha256', 'v2_lock_sha256', 'integration_lock_sha256')},
            'qualification_final_receipt_sha256': proof['final_receipt_sha256'],
            'science_reuse': review['science_reuse'],
            'workload': {key: protocol[key] for key in ('fits', 'models', 'seeds', 'updates_per_fit',
                                                      'batch_size', 'max_fit_seconds', 'counts', 'gates_per_seed_per_model')},
            'maximum_qualification_projection_seconds': max(a['projected_fit_seconds'] for a in proof['qualification_assessments']),
            'maximum_qualification_rss_bytes': max(a['process_peak_rss_bytes'] for a in proof['qualification_assessments']),
            'authorization_decisions_required': ['Explicit adoption of reviewed AST boundary and legacy status mapping',
                                                 'Separate explicit six-fit scientific training authorization'],
            'limitations': ['No scientific fit or neural OOD prediction exists.',
                            'Runtime identity and local authorization are not cryptographic hardware/human attestations.',
                            'An 800-update fit runtime and live AST execution equivalence remain untested.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qualification', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.qualification), indent=2, sort_keys=True))
