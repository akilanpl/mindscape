"""Version a bounded cloud evidence directory and verify clean restoration."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--restore', type=Path, required=True)
    parser.add_argument('--source-commit')
    parser.add_argument('--verify-existing', action='store_true')
    args = parser.parse_args()
    if args.verify_existing:
        manifest = json.loads((args.archive/'MANIFEST.json').read_text())
        packed = args.archive/'evidence.tar.gz'
        if digest(packed) != manifest['archive_sha256'] or args.restore.exists():
            raise RuntimeError('Archive changed or restore path not clean')
        args.restore.mkdir(parents=True)
        with tarfile.open(packed, 'r:gz') as archive:
            archive.extractall(args.restore, filter='data')
        restored = {str(p.relative_to(args.restore)): digest(p) for p in args.restore.rglob('*') if p.is_file()}
        if restored != manifest['files']:
            raise RuntimeError('Restored inventory/hash mismatch')
        print('PASS existing versioned archive clean restoration:', len(restored), 'files')
        return
    if args.source is None or args.source_commit is None:
        raise RuntimeError('Creation requires source and source commit')
    if args.archive.exists() or args.restore.exists():
        raise RuntimeError('Immutable archive or clean restore path already exists')
    files = {str(p.relative_to(args.source)): digest(p) for p in sorted(args.source.rglob('*')) if p.is_file()}
    args.archive.mkdir(parents=True)
    packed = args.archive/'evidence.tar.gz'
    with tarfile.open(packed, 'w:gz') as archive:
        for name in files:
            archive.add(args.source/name, arcname=name, recursive=False)
    if packed.stat().st_size >= 95*1024*1024:
        raise RuntimeError('Use multipart versioning for an artifact exceeding Git file limit')
    args.restore.mkdir(parents=True)
    with tarfile.open(packed, 'r:gz') as archive:
        archive.extractall(args.restore, filter='data')
    restored = {str(p.relative_to(args.restore)): digest(p) for p in args.restore.rglob('*') if p.is_file()}
    if files != restored:
        raise RuntimeError('Clean restore inventory/hash mismatch')
    manifest = {'scope': 'Bounded numeric mechanism evidence, not full Research v2 completion',
                'source_commit': args.source_commit, 'files': files,
                'archive_sha256': digest(packed), 'archive_bytes': packed.stat().st_size,
                'clean_restore': 'PASS', 'restored_files': len(restored)}
    (args.archive/'MANIFEST.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('PASS archive and clean restoration:', len(files), 'files;', packed.stat().st_size, 'bytes')


if __name__ == '__main__':
    main()
