"""Verify a frozen package and restore its repository layout into an empty directory."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path, PurePosixPath


def inside(root, relative):
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("Unsafe manifest path")
    return root.joinpath(*path.parts)


def restore(snapshot, destination):
    snapshot = snapshot.resolve()
    destination = destination.resolve()
    if destination == snapshot or snapshot in destination.parents:
        raise ValueError("Restore destination must be outside the frozen snapshot")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Restore destination must be empty; existing work is never overwritten")
    manifest = json.loads((snapshot / "MANIFEST.json").read_text())
    present = {
        str(path.relative_to(snapshot)) for path in snapshot.rglob("*")
        if path.is_file() and path != snapshot / "MANIFEST.json"
    }
    if present != set(manifest["files"]):
        raise ValueError("Frozen file inventory differs from manifest")
    for relative, expected in manifest["files"].items():
        path = inside(snapshot, relative)
        if (path.is_symlink() or not path.resolve().is_relative_to(snapshot)
                or hashlib.sha256(path.read_bytes()).hexdigest() != expected):
            raise ValueError("Frozen hash mismatch: " + relative)
    destinations = {
        label: inside(destination, original)
        for label, original in manifest["source_paths"].items()
    }
    destination.mkdir(parents=True, exist_ok=True)
    for label, target in destinations.items():
        source = inside(snapshot, label)
        if source.is_symlink():
            raise ValueError("Symlink source refused")
        shutil.copytree(source, target)
    shutil.copytree(snapshot / "source/mindscape", destination / "src/mindscape")
    for name in ("README.md", "pyproject.toml"):
        shutil.copy2(snapshot / name, destination / name)
    shutil.copy2(snapshot / "MANIFEST.json", destination / "FROZEN_MANIFEST.json")
    print("Verified and restored", len(manifest["files"]), "frozen files to", destination)
    print("Foundation weights require the pinned bootstrap; see docs/coding/reproducibility.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", default="results/final/coding_research_v2")
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    restore(Path(args.snapshot), Path(args.destination))
