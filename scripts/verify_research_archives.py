"""Read-only verification of archive parts and every streamed archived file."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

class Parts(io.RawIOBase):
    def __init__(self, paths):
        self.paths = iter(paths)
        self.stream = None
    def readable(self):
        return True
    def readinto(self, buffer):
        while True:
            if self.stream is None:
                path = next(self.paths, None)
                if path is None:
                    return 0
                self.stream = path.open('rb')
            n = self.stream.readinto(buffer)
            if n:
                return n
            self.stream.close()
            self.stream = None
    def close(self):
        if self.stream:
            self.stream.close()
        super().close()

def verify(root):
    if not (root / 'ARCHIVE.json').exists():
        manifest = json.loads((root / 'MANIFEST.json').read_text())
        path = root / 'evidence.tar.gz'
        if path.is_symlink():
            raise ValueError('Archive symlink refused')
        if 'archive_sha256' in manifest and digest(path) != manifest['archive_sha256']:
            raise ValueError('Archive hash mismatch')
        if 'archive_bytes' in manifest and path.stat().st_size != manifest['archive_bytes']:
            raise ValueError('Archive size mismatch')
        expected = manifest['files']
        seen = set()
        with tarfile.open(path, mode='r|gz') as archive:
            for member in archive:
                name = str(PurePosixPath(member.name))
                if name in seen or PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts:
                    raise ValueError('Unsafe/duplicate archive member')
                if member.isdir():
                    continue
                if not member.isfile():
                    raise ValueError('Archive links/special members refused')
                seen.add(name)
                h = hashlib.sha256()
                with archive.extractfile(member) as stream:
                    for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                        h.update(chunk)
                if h.hexdigest() != expected.get(name):
                    raise ValueError('Archived file mismatch: ' + name)
        if seen != set(expected):
            raise ValueError('Archive inventory incomplete')
        return {'archive': str(root), 'files': len(expected), 'status': 'PASS'}
    index = json.loads((root / 'ARCHIVE.json').read_text())
    manifest_path = root / 'SNAPSHOT_MANIFEST.json'
    if digest(manifest_path) != index['snapshot_manifest_sha256']:
        raise ValueError('Snapshot manifest hash mismatch')
    expected = json.loads(manifest_path.read_text())['files']
    paths = []
    for entry in index['parts']:
        name = PurePosixPath(entry['path'])
        if name.is_absolute() or '..' in name.parts:
            raise ValueError('Unsafe part path')
        path = root / str(name)
        if path.is_symlink() or path.stat().st_size != entry['size'] or digest(path) != entry['sha256']:
            raise ValueError('Archive part mismatch: ' + str(name))
        paths.append(path)
    seen = set()
    with Parts(paths) as raw, io.BufferedReader(raw) as joined:
        with tarfile.open(fileobj=joined, mode='r|gz') as archive:
            for member in archive:
                name = str(PurePosixPath(member.name))
                if name in seen or PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts:
                    raise ValueError('Unsafe/duplicate archive member')
                if not member.isfile():
                    if member.isdir():
                        continue
                    raise ValueError('Links/special members are not accepted')
                seen.add(name)
                h = hashlib.sha256()
                with archive.extractfile(member) as stream:
                    for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                        h.update(chunk)
                wanted = index['snapshot_manifest_sha256'] if name == 'MANIFEST.json' else expected.get(name)
                if wanted is None or h.hexdigest() != wanted:
                    raise ValueError('Archived file mismatch: ' + name)
    if seen != set(expected) | {'MANIFEST.json'}:
        raise ValueError('Archive inventory is incomplete')
    return {'archive': str(root), 'parts': len(paths), 'files': len(expected), 'status': 'PASS'}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('roots', nargs='+', type=Path)
    args = parser.parse_args()
    for root in args.roots:
        print(json.dumps(verify(root)), flush=True)
