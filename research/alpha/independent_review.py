"""Read-only independent byte preservation review. Never imports scientific modules."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

BASELINE = '6c558caa3c3a1a09ec15e286a408cbb6d5076b03'


def git(root, *arguments):
    return subprocess.check_output(['git', '-C', str(root), *arguments])


def review(root, baseline=BASELINE):
    root = Path(root).resolve()
    paths = git(root, 'ls-tree', '-rz', '--name-only', baseline).decode().split('\0')
    changed = []
    inventory = {}
    for relative in filter(None, paths):
        before = git(root, 'show', f'{baseline}:{relative}')
        path = root / relative
        if path.is_symlink() or not path.is_file() or path.read_bytes() != before:
            changed.append(relative)
        inventory[relative] = hashlib.sha256(before).hexdigest()
    sources = {}
    for pattern in ('research/alpha/*.py', 'src/mindscape/alpha_*/*.py',
                    'configs/alpha_ood/*.json', 'docs/alpha/*'):
        for path in root.glob(pattern):
            if path.is_file():
                sources[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {'baseline': baseline, 'head': git(root, 'rev-parse', 'HEAD').decode().strip(),
            'baseline_files_checked': len(inventory), 'changed_baseline_files': changed,
            'baseline_inventory_sha256': hashlib.sha256(json.dumps(inventory, sort_keys=True).encode()).hexdigest(),
            'submission_file_sha256': sources, 'scientific_status': 'UNRUN',
            'limitation': 'Local byte preservation and source review; no custody, execution or outcome attestation.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('repository', type=Path)
    args = parser.parse_args()
    print(json.dumps(review(args.repository), indent=2, sort_keys=True))
