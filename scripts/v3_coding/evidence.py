"""Deterministic multipart evidence export and bounded path-safe restoration."""

import argparse
import fcntl
import gzip
import hashlib
import json
import tarfile
import tempfile
from contextlib import ExitStack
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def pack(root, destination):
    root = Path(root)
    if (root / ".lock").exists():
        raise ValueError("Cannot export a running experiment")
    # Retain every existing run lock throughout the export, blocking new invocations.
    with ExitStack() as stack:
        for path in sorted(root.rglob("runner.lock")):
            handle = stack.enter_context(path.open("r"))
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError("Cannot export an active run") from error
        return _pack(root, destination)


def _pack(root, destination):
    root, destination = Path(root), Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    if (root / ".lock").exists():
        raise ValueError("Cannot export a running experiment")
    files = {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*")) if p.is_file()}
    with tempfile.TemporaryDirectory(prefix="v3-coding-pack-") as tmp:
        archive = Path(tmp, "evidence.tar.gz")
        with (
            archive.open("wb") as raw,
            gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as zipped,
            tarfile.open(fileobj=zipped, mode="w|") as tar,
        ):
            for name in files:
                path = root / name
                if path.is_symlink():
                    raise ValueError("Evidence symlink forbidden")
                info = tar.gettarinfo(str(path), arcname=name)
                info.uid = info.gid = info.mtime = 0
                info.uname = info.gname = ""
                with path.open("rb") as f:
                    tar.addfile(info, f)
        parts = []
        with archive.open("rb") as f:
            index = 0
            while chunk := f.read(32 * 1024 * 1024):
                path = destination / f"evidence.part{index:03d}"
                path.write_bytes(chunk)
                parts.append({"name": path.name, "size": len(chunk), "sha256": sha(path)})
                index += 1
        manifest = {
            "schema": "mindscape.v3.coding.evidence/1",
            "archive_sha256": sha(archive),
            "files": files,
            "parts": parts,
            "protocol_commit": "38a8494",
            "lockbox": "historical lockbox untouched; pretrained weights and historical adapters not exported",
        }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {
        "files": len(files),
        "parts": len(parts),
        "compressed_bytes": sum(p["size"] for p in parts),
    }


def restore(package, destination):
    package, destination = Path(package), Path(destination)
    manifest = json.loads((package / "manifest.json").read_text())
    if manifest["schema"] != "mindscape.v3.coding.evidence/1":
        raise ValueError("Unknown archive schema")
    if destination.exists():
        raise ValueError("Destination must be new")
    if len(manifest["files"]) > 5000:
        raise ValueError("Member budget exceeded")
    with tempfile.TemporaryDirectory(prefix="v3-coding-restore-") as tmp:
        archive = Path(tmp, "evidence.tar.gz")
        total = 0
        with archive.open("wb") as joined:
            for part in manifest["parts"]:
                name = Path(part["name"])
                if name.name != part["name"]:
                    raise ValueError("Unsafe part name")
                path = package / name
                if (
                    path.is_symlink()
                    or sha(path) != part["sha256"]
                    or path.stat().st_size != part["size"]
                ):
                    raise ValueError("Corrupt archive part")
                total += part["size"]
                if total > 512 * 1024 * 1024:
                    raise ValueError("Compressed byte budget exceeded")
                with path.open("rb") as f:
                    for chunk in iter(lambda: f.read(1024 * 1024), b""):
                        joined.write(chunk)
        if sha(archive) != manifest["archive_sha256"]:
            raise ValueError("Joined archive mismatch")
        seen = set()
        expanded = 0
        # Verify every member first; no writes until the entire archive passes.
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar:
                name = Path(member.name)
                if (
                    not member.isfile()
                    or name.is_absolute()
                    or ".." in name.parts
                    or name.as_posix() != member.name
                    or member.name in seen
                    or member.size > 128 * 1024 * 1024
                ):
                    raise ValueError("Unsafe member")
                expanded += member.size
                if expanded > 1024 * 1024 * 1024:
                    raise ValueError("Expanded byte budget exceeded")
                content = tar.extractfile(member)
                h = hashlib.sha256()
                for chunk in iter(lambda content=content: content.read(1024 * 1024), b""):
                    h.update(chunk)
                if h.hexdigest() != manifest["files"].get(member.name):
                    raise ValueError("Member checksum mismatch")
                seen.add(member.name)
        if seen != set(manifest["files"]):
            raise ValueError("Incomplete member set")
        destination.mkdir(parents=True, exist_ok=False)
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar:
                target = destination / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as f:
                    content = tar.extractfile(member)
                    for chunk in iter(lambda content=content: content.read(1024 * 1024), b""):
                        f.write(chunk)
    return {"verified_files": len(seen), "compressed_bytes": total, "expanded_bytes": expanded}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)
    a = sub.add_parser("pack")
    a.add_argument("--root", required=True)
    a.add_argument("--destination", required=True)
    a = sub.add_parser("restore")
    a.add_argument("--package", required=True)
    a.add_argument("--destination", required=True)
    args = p.parse_args()
    result = (
        pack(args.root, args.destination)
        if args.mode == "pack"
        else restore(args.package, args.destination)
    )
    print(json.dumps(result))
