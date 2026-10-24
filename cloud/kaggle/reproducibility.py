"""Content-addressed, allowlisted packaging; never executes model fitting."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

SEEDS = [17, 29, 43]
CONDITIONS = ['graph-concept-on', 'graph-concept-off', 'matched-gru']
FROZEN_COMMIT = 'd54ff8fe64f30c53b9662ddba90691338aad05f6'
PROTOCOL = dict(optimizer='Adam', learning_rate=0.001, batch_size=1,
                optimizer_updates=2000, seeds=SEEDS, conditions=CONDITIONS)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)

def snapshot(root):
    root = Path(root)
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*'))
            if p.is_file() and not p.is_symlink() and '.git' not in p.parts
            and '__pycache__' not in p.parts}

def prepare_training(repo, data_root, out):
    repo, data_root, out = map(Path, (repo, data_root, out))
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != FROZEN_COMMIT:
        raise ValueError('Expected actual local frozen R5.7c d54ff8f source')
    freeze = json.loads((data_root.parent / 'candidate-freeze.json').read_text())
    for rel, digest in freeze['source_hashes'].items():
        if sha(repo / rel) != digest:
            raise ValueError('Frozen source differs: ' + rel)
    data_manifest = json.loads((data_root / 'manifest.json').read_text())
    for seed in SEEDS:
        p = data_root / f'train/{seed}.json'
        if sha(p) != data_manifest['files'][f'train/{seed}.json']:
            raise ValueError('Frozen training bytes differ')
        rows = json.loads(p.read_text())
        if len(rows) != 128 or any(r['seed'] != seed or r['split'] != 'train' for r in rows):
            raise ValueError('Frozen training protocol differs')
    out.mkdir(parents=True, exist_ok=False)
    files = {}
    for path in sorted((repo / 'src').rglob('*.py')):
        if path.is_symlink():
            raise ValueError('Symlink source refused')
        rel = path.relative_to(repo)
        target = out / 'source' / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        files[str(rel)] = sha(target)
    # Only executable cloud code, no evaluator data or local reports/credentials.
    infra = {}
    for path in sorted((repo / 'cloud/kaggle').rglob('*.py')):
        if 'tests' in path.parts or path.is_symlink():
            continue
        rel = path.relative_to(repo)
        if 'evaluation' in path.parts:
            continue
        target = out / 'source' / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        infra[str(rel)] = sha(target)
    (out / 'train').mkdir()
    hashes = {}
    for seed in SEEDS:
        target = out / f'train/{seed}.json'
        shutil.copyfile(data_root / f'train/{seed}.json', target)
        hashes[str(seed)] = sha(target)
    manifest = dict(schema='mindscape-cloud-training-v1', source_commit=commit,
        source_files_sha256=files, infrastructure_files_sha256=infra,
        training_files_sha256=hashes, protocol=PROTOCOL, total_optimizer_updates=18000,
        checkpoint_policy='final-update-2000-only', device='cuda:0',
        evaluation_data_included=False, automatic_resume=False,
        freeze_sha256=sha(data_root.parent / 'candidate-freeze.json'))
    write_new(out / 'manifest.json', manifest)
    return manifest

def verify_export(run):
    run = Path(run)
    result = json.loads((run / 'result.json').read_text())
    if result.get('optimizer_updates') != 2000 or result.get('optimizer_steps') != 2000:
        raise ValueError('Incorrect update budget')
    if result.get('status') != 'COMPLETED_EXPLORATORY' or result.get('device') != 'cuda:0':
        raise ValueError('Completed CUDA fit receipt required')
    if result.get('seed') not in SEEDS or result.get('condition') not in CONDITIONS:
        raise ValueError('Unknown fit identity')
    if result.get('source_commit') != FROZEN_COMMIT:
        raise ValueError('Source commit mismatch')
    if result.get('step_post_hook_count') != 2000:
        raise ValueError('Missing optimizer hook receipt')
    if result['adam_steps_by_parameter'] != result['gradient_exposures_by_parameter']:
        raise ValueError('Adam exposure mismatch')
    for file, key in [('initial.json', 'initial_sha256'), ('final.json', 'checkpoint_sha256'),
                      ('loss.jsonl', 'loss_sha256')]:
        if sha(run / file) != result[key]:
            raise ValueError('Artifact changed: ' + file)
    losses = [json.loads(line) for line in (run / 'loss.jsonl').read_text().splitlines()]
    if [entry['update'] for entry in losses] != list(range(1, 2001)):
        raise ValueError('Incomplete loss ledger')
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(dest='command', required=True)
    prep = subs.add_parser('prepare')
    for flag in ('repo', 'data-root', 'output'):
        prep.add_argument('--' + flag, required=True, type=Path)
    verify = subs.add_parser('verify-export')
    verify.add_argument('run', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare_training(args.repo, args.data_root, args.output)
    else:
        print(json.dumps(verify_export(args.run), indent=2))
