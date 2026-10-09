"""Stream-verify the complete export, restoring only its immutable v1 package."""
import argparse
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path, PurePosixPath


class PartStream(io.RawIOBase):
    def __init__(self, paths):
        self.paths = iter(paths)
        self.current = None

    def readable(self):
        return True

    def readinto(self, buffer):
        while True:
            if self.current is None:
                try:
                    self.current = next(self.paths).open('rb')
                except StopIteration:
                    return 0
            count = self.current.readinto(buffer)
            if count:
                return count
            self.current.close()
            self.current = None

    def close(self):
        if self.current is not None:
            self.current.close()
        super().close()


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    archive, destination = args.archive.resolve(), args.destination.resolve()
    snapshot = destination.with_name(destination.name+'_snapshot')
    if snapshot.exists() or destination.exists():
        raise RuntimeError('Clean destinations required; no overwrite')
    index = json.loads((archive/'ARCHIVE.json').read_text())
    if digest(archive/'SNAPSHOT_MANIFEST.json') != index['snapshot_manifest_sha256']:
        raise RuntimeError('Export manifest mismatch')
    manifest = json.loads((archive/'SNAPSHOT_MANIFEST.json').read_text())
    paths = []
    for entry in index['parts']:
        name = PurePosixPath(entry['path'])
        if name.is_absolute() or '..' in name.parts or len(name.parts) != 1:
            raise RuntimeError('Unsafe archive part')
        path = archive/entry['path']
        if path.stat().st_size != entry['size'] or digest(path) != entry['sha256']:
            raise RuntimeError('Archive part mismatch')
        paths.append(path)
    snapshot.mkdir(parents=True)
    prefix = 'all_results/final/coding_research_v2/'
    seen = set()
    with io.BufferedReader(PartStream(paths)) as stream, tarfile.open(fileobj=stream, mode='r|gz') as packed:
        for member in packed:
            name = PurePosixPath(member.name)
            if not member.isfile() or name.is_absolute() or '..' in name.parts or member.name in seen:
                raise RuntimeError('Unsafe or duplicate export member')
            expected = (index['snapshot_manifest_sha256'] if member.name == 'MANIFEST.json'
                        else manifest['files'].get(member.name))
            if expected is None:
                raise RuntimeError('Unexpected export member')
            seen.add(member.name)
            target = snapshot/member.name[len(prefix):] if member.name.startswith(prefix) else None
            output = None
            if target is not None:
                target.parent.mkdir(parents=True, exist_ok=True)
                output = target.open('wb')
            h = hashlib.sha256()
            try:
                with packed.extractfile(member) as source:
                    for chunk in iter(lambda: source.read(1024*1024), b''):
                        h.update(chunk)
                        if output is not None:
                            output.write(chunk)
            finally:
                if output is not None:
                    output.close()
            if h.hexdigest() != expected:
                raise RuntimeError('Export file hash mismatch: '+member.name)
    if seen != set(manifest['files']) | {'MANIFEST.json'}:
        raise RuntimeError('Export inventory mismatch')
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'coding'))
    from restore_completion import restore
    restore(snapshot, destination)
    from prepare_cpu_runtime import prepare
    pin = json.loads((destination/'configs/coding/runtime_v1.json').read_text())
    runtime_receipt = prepare(destination/'work/coding/runtime', pin['python_wasm_sha256'])
    (destination/'CPU_RUNTIME_PREPARATION.json').write_text(json.dumps(runtime_receipt, indent=2)+'\n')
    print('PASS all', len(manifest['files']), 'export hashes; selected immutable package clean-restored')


if __name__ == '__main__':
    main()
