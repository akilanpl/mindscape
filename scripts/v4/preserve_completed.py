"""Archive the original run and frozen sources, then verify clean restoration by SHA-256."""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(root):
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*")) if p.is_file()}


def preserve(run, destination, restore):
    original = inventory(run)
    source_manifest = json.loads((run / "manifest.json").read_text())
    sources = source_manifest["sources"]
    for name, expected in sources.items():
        if sha(Path(name)) != expected:
            raise ValueError("Frozen source mismatch: " + name)
    destination.mkdir(parents=True, exist_ok=False)
    restore.mkdir(parents=True, exist_ok=False)
    for archive_name, files, base, subdir in (
        ("evidence.tar.gz", original, run, "run"),
        ("source.tar.gz", sources, Path("."), "source"),
    ):
        path = destination / archive_name
        with tarfile.open(path, "w:gz") as tar:
            for name in files:
                tar.add(base / name, arcname=name, recursive=False)
        restored = restore / subdir
        restored.mkdir()
        with tarfile.open(path, "r:gz") as tar:
            tar.extractall(restored, filter="data")
        if inventory(restored) != files:
            raise ValueError("Restoration mismatch: " + archive_name)
    if inventory(run) != original:
        raise ValueError("Original run changed")
    result = {
        "scope": "Complete original Mindscape V4 native pilot and its frozen executable sources",
        "run": str(run),
        "original_decision": "NO_GO",
        "original_files": original,
        "sources": sources,
        "archives": {
            name: sha(destination / name) for name in ("evidence.tar.gz", "source.tar.gz")
        },
        "clean_restore": "PASS",
        "original_files_unchanged": len(original),
        "source_files": len(sources),
    }
    (destination / "MANIFEST.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {"clean_restore": "PASS", "run_files": len(original), "source_files": len(sources)}
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--restore", type=Path, required=True)
    args = parser.parse_args()
    preserve(args.run, args.destination, args.restore)
