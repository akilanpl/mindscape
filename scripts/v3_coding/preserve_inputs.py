"""Hash immutable external model, adapter, old lockbox and WASI files without executing them."""

import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def pins(roots):
    return {
        str(p.resolve()): sha(p)
        for root in roots
        for p in sorted(Path(root).rglob("*"))
        if p.is_file()
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--roots", nargs="+")
    p.add_argument("--output", required=True)
    p.add_argument("--verify", action="store_true")
    a = p.parse_args()
    out = Path(a.output)
    if a.verify:
        saved = json.loads(out.read_text())
        bad = [name for name, value in saved["files"].items() if sha(Path(name)) != value]
        if bad:
            raise ValueError("Protected input changed: " + str(bad))
        print(json.dumps({"verified_files": len(saved["files"]), "immutable": True}))
    else:
        if out.exists():
            raise ValueError("Receipt exists")
        files = pins(a.roots)
        out.write_text(
            json.dumps({"files": files, "policy": "read-only historical inputs"}, indent=2)
        )
        print(json.dumps({"pinned_files": len(files)}))
