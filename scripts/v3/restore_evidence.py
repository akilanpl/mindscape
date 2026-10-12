"""Checksum-verified, path-safe restoration and independent replay of V3 evidence."""

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path

from mindscape.v3.domains import DOMAINS
from mindscape.v3.kernel import replay
from mindscape.v3.schema import digest


def restore(archive, manifest_path, destination):
    manifest = json.loads(Path(manifest_path).read_text())
    payload = Path(archive).read_bytes()
    if hashlib.sha256(payload).hexdigest() != manifest["archive_sha256"]:
        raise ValueError("Archive checksum mismatch")
    files = {}
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as tar:
        for member in tar:
            name = Path(member.name)
            if (
                not member.isfile()
                or name.is_absolute()
                or ".." in name.parts
                or member.name in files
                or member.size > 20_000_000
            ):
                raise ValueError("Unsafe evidence member")
            if len(files) >= 1000:
                raise ValueError("Archive member budget exceeded")
            content = tar.extractfile(member).read()
            if hashlib.sha256(content).hexdigest() != manifest["files"].get(member.name):
                raise ValueError("Evidence member checksum mismatch")
            files[member.name] = content
    if set(files) != set(manifest["files"]):
        raise ValueError("Incomplete evidence")
    identity = json.loads(files["cpu-smoke/identity.json"])
    summary = json.loads(files["cpu-smoke/summary.json"])
    if digest(identity) != summary["identity_hash"]:
        raise ValueError("Run identity mismatch")
    keys = set()
    for row in summary["rows"]:
        if row["key"] in keys:
            raise ValueError("Duplicate completed key")
        keys.add(row["key"])
        saved = json.loads(files[f"cpu-smoke/traces/{row['key']}.json"])
        trace = saved["trace"]
        if digest(trace) != saved["sha256"] or saved["sha256"] != row["trace_hash"]:
            raise ValueError("Raw/summary hash mismatch")
        judgment = replay(trace, DOMAINS[row["domain"]])
        if judgment.success != row["success"]:
            raise ValueError("Raw/summary judgment mismatch")
    if len(keys) != summary["completed"]:
        raise ValueError("Execution count mismatch")
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=False)
    for name, content in files.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    return {
        "verified_files": len(files),
        "replayed_executions": len(keys),
        "neural_forward_calls": 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", default="experiments/v3/cpu-smoke.tar.gz")
    parser.add_argument("--manifest", default="experiments/v3/evidence_manifest.json")
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    print(json.dumps(restore(args.archive, args.manifest, args.destination)))
