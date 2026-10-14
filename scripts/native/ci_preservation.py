"""Additive native-release audit; does not edit or replace historical checks."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess

BASELINE = '3cefed2a2cc4e6a5867602822fcd35bc8d19bc96'
FROZEN_V3 = '33c5b5657b2d19741b7b0ac84388d4f7997f9582'
PR27_BASE = '8c141d8f1b237251d8fa9e5b4ccd87efbf3b3c96'
# Existing product surfaces are mutable in this correction. All other baseline
# blobs, including original R5.6 research sources, are protected byte-for-byte.
PRODUCT_PREFIXES = ('src/mindscape/mind1/', 'mind1_web/', 'docs/mind1/', 'tests_mind1/')
PRODUCT_FILES = ('.env.mind1.example', 'requirements-mind1.txt')


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], stderr=subprocess.PIPE)


def blob_id(path, mode):
    """Git SHA-1 blob ID, streaming file bytes without writing Git objects."""
    if mode == '120000':
        raw = os.fsencode(os.readlink(path))
        return hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
    size = path.stat().st_size
    digest = hashlib.sha1(b'blob '+str(size).encode()+b'\0')
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''): digest.update(chunk)
    return digest.hexdigest()


def inventory(root, baseline=BASELINE):
    entries = {}
    for item in git(root, 'ls-tree', '-r', '-z', baseline).split(b'\0'):
        if not item: continue
        meta, name = item.split(b'\t',1)
        mode, kind, digest = meta.decode().split()
        if kind != 'blob': raise ValueError('Unsupported historical non-blob entry')
        entries[name.decode()] = (mode, digest)
    return entries


def verify(root, baseline=BASELINE):
    root = Path(root).resolve()
    protected = {name: value for name,value in inventory(root,baseline).items()
                 if not name.startswith(PRODUCT_PREFIXES) and name not in PRODUCT_FILES}
    changed = []
    for name,(mode,expected) in protected.items():
        path = root/name
        try:
            if path.is_symlink() != (mode == '120000') or blob_id(path,mode) != expected:
                changed.append(name)
            elif mode != '120000' and bool(path.stat().st_mode & stat.S_IXUSR) != (mode == '100755'):
                changed.append(name)
        except (OSError, ValueError): changed.append(name)
    tags_path = root/'configs/v3/preserved_tags.json'
    tag_changes = []
    expected_tags = json.loads(tags_path.read_text()) if tags_path.is_file() else {}
    for name, expected_ref in expected_tags.items():
        try: actual_ref = git(root,'rev-parse','refs/tags/'+name).decode().strip()
        except subprocess.CalledProcessError: actual_ref = None
        if actual_ref != expected_ref: tag_changes.append(name)
    return {'baseline': baseline, 'protected_baseline_files': len(protected),
            'protected_baseline_unchanged': not changed and not tag_changes,
            'declared_historical_tags': len(expected_tags), 'changed_or_missing_tags': tag_changes, 'changed_or_missing': sorted(changed),
            'mutable_existing_product_prefixes': list(PRODUCT_PREFIXES),
            'mutable_existing_product_files': list(PRODUCT_FILES),
            'scope': 'Every baseline tracked file outside listed product prefixes and files; new additive files are not frozen baseline blobs.',
            'historical_checks_replaced': False}


def readme_audit(root):
    refs = {'frozen_v3': FROZEN_V3, 'pr27_base': PR27_BASE, 'pr27_head': BASELINE}
    hashes = {name: git(root,'rev-parse',ref+':README.md').decode().strip() for name,ref in refs.items()}
    return {'references': refs, 'readme_git_blob_ids': hashes,
            'preexisting_frozen_readme_divergence': hashes['frozen_v3'] != hashes['pr27_base'] == hashes['pr27_head']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = {'native_preservation': verify(args.root), 'historical_readme_audit': readme_audit(args.root)}
    raw = json.dumps(result,indent=2)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(raw)
    print(raw,end='')
    if not result['native_preservation']['protected_baseline_unchanged']: raise SystemExit(1)


if __name__ == '__main__': main()
