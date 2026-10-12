"""Verify all pre-V3 tracked blobs and tag refs against the declared untouched base."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

BASE = "33c5b5657b2d19741b7b0ac84388d4f7997f9582"


def git(*args):
    return subprocess.check_output(["git", *args])


def verify():
    tree = git("ls-tree", "-r", "-z", BASE).split(b"\0")
    protected = {}
    for entry in tree:
        if not entry:
            continue
        metadata, rawpath = entry.split(b"\t", 1)
        _, kind, expected = metadata.split()
        if kind != b"blob":
            raise ValueError("Unexpected non-blob historical entry")
        path = Path(rawpath.decode())
        actual = git("hash-object", "--", str(path)).strip()
        if actual != expected:
            raise ValueError(f"Historical file changed: {path}")
        protected[str(path)] = expected.decode()
    tags = {
        name: git("rev-parse", f"refs/tags/{name}").decode().strip()
        for name in git("tag", "--list").decode().splitlines()
    }
    expected_tags = json.loads(Path("configs/v3/preserved_tags.json").read_text())
    if tags != expected_tags:
        raise ValueError("Release tags changed")
    return {
        "base_commit": BASE,
        "verified_pre_v3_files": len(protected),
        "tree_sha256": hashlib.sha256(json.dumps(protected, sort_keys=True).encode()).hexdigest(),
        "tags": tags,
        "historical_files_unchanged": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    receipt = verify()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))
