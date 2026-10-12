"""Hash-bound atomic receipts and exclusive resumable run lock."""

import fcntl
import hashlib
import json
import os
from pathlib import Path

from .schema import digest


def atomic_json(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as stream:
        json.dump(row, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1048576), b""):
            h.update(part)
    return h.hexdigest()


def source_hash():
    return digest({p.name: file_sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))})


class RunLock:
    def __init__(self, root, identity):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.handle = (self.root / "runner.lock").open("a")
        fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = self.root / "identity.json"
        if path.exists() and json.loads(path.read_text()) != identity:
            raise ValueError("Resume identity changed")
        if not path.exists():
            atomic_json(path, identity)

    def close(self):
        fcntl.flock(self.handle, fcntl.LOCK_UN)
        self.handle.close()
