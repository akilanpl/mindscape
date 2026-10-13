"""Code/data freeze and read-only hash validation."""
import json
import subprocess
from pathlib import Path
from .data import ROOT, canonical, digest

PROTOCOL = {
    'version': 'R5.4-v1', 'seeds': [11, 23, 47], 'models': {'markov': 2188, 'recurrent': 2124},
    'fits': 6, 'updates_per_fit': 800, 'batch_size': 32, 'max_fit_seconds': 900,
    'optimizer': {'kind': 'AdamW', 'lr': 0.003, 'weight_decay': 0.0001, 'clip_norm': 1.0},
    'data_order': 'Python Random(seed); permutation every 32 updates; identical policy for both models',
    'checkpoint': 'final update 800 only; no selection, retries, early stopping or OOD tuning',
    'objective': 'updates 1-400 TF CE(digit)+CE(carry); 401-800 equal TF and own-carry local CE',
    'gradient': 'BPTT through GRU hidden; detached argmax carry; no derivative through carry choice',
    'counts': {'train': 1024, 'development': 64, 'ood': 192, 'ood_per_family': 48, 'context_probes': 200},
    'family_order': ['length_known', 'long_carry_known', 'mixed_known', 'unseen_short'],
    'memory': 'exact training-context map; miss -> (digit 0, carry 0); no answer retrieval',
    'whole_problem_memory': 'commutation-normalized training identity map; miss -> answer 0',
    'zero_weight': 'same two architectures, all tensors zero, argmax first class',
    'gates_per_seed_per_model': {
        'ood_exact_min': 0.95, 'ood_wilson95_lower_min': 0.90,
        'each_known_family_exact_min': 0.95, 'long_carry_exact_min': 0.95,
        'unseen_short_exact_min': 0.90, 'unseen_short_wilson95_lower_min': 0.78,
        'held_context_probe_accuracy_min': 0.90, 'ood_gain_over_memory_min': 0.15,
        'unseen_gain_over_memory_min': 0.75, 'paired_p_max': 0.05 / 12},
    'paired_tests': 'one-sided exact McNemar vs local memory, overall and unseen family; Bonferroni 12',
    'intervals': 'Wilson95 per seed; task denominator 192/48; repeated seeds never independent tasks',
    'success': 'all gates in all three seeds for a model; recurrence benefit exploratory paired comparison',
    'device': 'native MPS float32, no downloads, no CPU fallback; qualification backward only',
    'qualification': 'known families have no unseen contexts; memory overall <=75%; held exposures=0',
    'inference_boundary': 'Operands only; learned adapter has no reference import or label argument',
    'evaluation_order': 'all free predictions and checkpoints sealed before outcomes read; TF diagnostic follows',
    'terminal_readiness': 'ENGINEERING_NO_GO until MPS backward and bounded resource qualification pass',
}


def paths():
    return sorted([*Path('src/mindscape/r5_4').glob('*.py'), *Path('tests/r5_4').glob('*.py'),
                   Path('docs/r5_4/protocol.md'), ROOT / 'protocol.json',
                   *[ROOT / f'{s}.json' for s in ('train', 'development', 'ood', 'outcomes')]])


def make_lock():
    (ROOT / 'protocol.json').write_text(canonical(PROTOCOL))
    lock = dict(version='R5.4-v1', base_commit=subprocess.check_output(
        ['git', 'merge-base', 'HEAD', 'origin/development/r5.3-robust-physics'], text=True).strip(),
        sha256={str(p): digest(p) for p in paths()})
    (ROOT / 'LOCK.json').write_text(canonical(lock))
    return lock


def verify_lock():
    lock = json.loads((ROOT / 'LOCK.json').read_text())
    if set(lock['sha256']) != set(map(str, paths())):
        raise ValueError('Incomplete frozen source inventory')
    for path, expected in lock['sha256'].items():
        if digest(path) != expected:
            raise ValueError(f'Freeze mismatch: {path}')
    if json.loads((ROOT / 'protocol.json').read_text()) != PROTOCOL:
        raise ValueError('Protocol mismatch')
    # Frozen files must already be committed; qualification cannot silently refreeze a campaign.
    changed = subprocess.check_output(['git', 'status', '--porcelain', '--', *lock['sha256']], text=True)
    if changed:
        raise ValueError('Frozen inputs not committed cleanly')
    return lock
