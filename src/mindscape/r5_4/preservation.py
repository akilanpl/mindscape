"""Authenticated archive creation and read-only verification, never extraction."""
import hashlib
import io
import json
import tarfile
from pathlib import Path, PurePosixPath
from .data import ROOT, canonical
from .freeze import paths, verify_lock


def preserve(campaign, archive):
    from .audit import audit
    audit(campaign)  # incomplete/failed evidence cannot receive a verified receipt
    verify_lock()
    source = [*paths(), ROOT / 'LOCK.json']
    members = {f'source/{p}': p.read_bytes() for p in source}
    for path in sorted(Path(campaign).iterdir()):
        if not path.is_file() or path.is_symlink():
            raise ValueError('Unexpected evidence entry')
        members[f'campaign/{path.name}'] = path.read_bytes()
    receipt = {'schema': 'R5.4-preservation-v1', 'sha256': {
        name: hashlib.sha256(content).hexdigest() for name, content in members.items()}}
    members['PRESERVATION.json'] = canonical(receipt).encode()
    archive = Path(archive)
    # Exclusive output creation preserves previous archives.
    with archive.open('xb') as stream, tarfile.open(fileobj=stream, mode='w:gz') as bundle:
        for name, content in sorted(members.items()):
            info = tarfile.TarInfo(name)
            info.size, info.mtime = len(content), 0
            bundle.addfile(info, io.BytesIO(content))
    return verify_archive(archive)


def verify_archive(archive):
    content = {}
    with tarfile.open(archive, 'r:gz') as bundle:
        for member in bundle:
            path = PurePosixPath(member.name)
            if not member.isfile() or path.is_absolute() or '..' in path.parts or member.name in content:
                raise ValueError('Unsafe/duplicate archive member')
            content[member.name] = bundle.extractfile(member).read()
    receipt = json.loads(content.pop('PRESERVATION.json'))
    if receipt['schema'] != 'R5.4-preservation-v1' or set(content) != set(receipt['sha256']):
        raise ValueError('Incomplete archive receipt')
    if any(hashlib.sha256(value).hexdigest() != receipt['sha256'][name] for name, value in content.items()):
        raise ValueError('Archive hash mismatch')
    return dict(status='VERIFIED', members=len(content),
                archive_sha256=hashlib.sha256(Path(archive).read_bytes()).hexdigest())
