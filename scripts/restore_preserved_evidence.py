"""Verify a preservation release and optionally restore into a new directory."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

def safe_path(name):
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or not p.parts:
        raise ValueError('Unsafe path')
    return p

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--destination', type=Path)
    args = parser.parse_args()
    h = hashlib.sha256()
    with args.archive.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    if h.hexdigest() != args.sha256:
        raise ValueError('Archive SHA-256 mismatch')
    with tarfile.open(args.archive, 'r:gz') as archive:
        members = archive.getmembers()
        names = [m.name for m in members]
        if len(names) != len(set(names)) or any(not m.isfile() for m in members):
            raise ValueError('Duplicate or non-regular archive members')
        manifest = json.load(archive.extractfile('MANIFEST.json'))
        expected = manifest['objects']
        if set(names) != {'MANIFEST.json'} | {'objects/' + sha for sha in expected}:
            raise ValueError('Incomplete or unexpected archive inventory')
        for sha, entry in expected.items():
            safe_path(entry['origin'])
            member = archive.getmember('objects/' + sha)
            if member.size != entry['size']:
                raise ValueError('Object size mismatch')
            h = hashlib.sha256()
            with archive.extractfile(member) as stream:
                for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                    h.update(chunk)
            if h.hexdigest() != sha:
                raise ValueError('Object SHA-256 mismatch')
        # No output files are created until the entire archive has passed.
        if args.destination:
            args.destination.mkdir(parents=True, exist_ok=False)
            for sha, entry in expected.items():
                path = args.destination / 'origins' / str(safe_path(entry['origin']))
                path.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile('objects/' + sha) as source, path.open('xb') as target:
                    for chunk in iter(lambda: source.read(4 * 1024 * 1024), b''):
                        target.write(chunk)
    print(json.dumps({'status': 'PASS', 'objects': len(expected),
                      'restored': args.destination is not None}))

if __name__ == '__main__':
    main()
