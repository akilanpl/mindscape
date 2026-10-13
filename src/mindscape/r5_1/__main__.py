"""One fail-closed native engineering+qualification run; never training."""

import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

from mindscape.v3.schema import digest

from .qualification import save, screen


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    manifest_path = Path("configs/r5_1/tasks.json")
    freeze_path = Path("configs/r5_1/source-freeze.json")
    manifest = json.loads(manifest_path.read_text())
    freeze = json.loads(freeze_path.read_text())
    if (
        digest(manifest["tasks"]) != manifest["task_hash"]
        or digest(manifest["protocol"]) != manifest["protocol_hash"]
    ):
        raise ValueError("Task/protocol freeze")
    for path in (manifest_path, freeze_path, Path("configs/r5_1/protocol.json")):
        if subprocess.check_output(["git", "show", "HEAD:" + str(path)]) != path.read_bytes():
            raise ValueError("Freeze files must be committed before execution")
    for name, expected in freeze["sources"].items():
        if sha(name) != expected:
            raise ValueError("Frozen source changed: " + name)
    # Deterministic attempt lock prevents repeat screens. No resume or alternate output option.
    root = Path("results/r5_1") / ("screen-" + sha(manifest_path)[:16])
    root.mkdir(parents=True, exist_ok=False)
    save(root / "manifest.json", manifest)
    save(root / "source-freeze.json", freeze)
    save(
        root / "commit.json",
        {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()},
    )
    try:
        from .engineering import run

        engineering = run(manifest)
        save(root / "engineering.json", engineering)
        summary = screen(manifest, root)
    except (RuntimeError, ValueError, OSError, KeyError, TypeError) as exc:
        summary = {
            "decision": "ENGINEERING_NO_GO",
            "failure": {"type": type(exc).__name__, "message": str(exc)},
            "neural_training": False,
        }
        save(root / "summary.json", summary)
    files = {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*")) if p.is_file()}
    archive = root.with_suffix(".tar.gz")
    with tarfile.open(archive, "x:gz") as tar:
        for name in files:
            tar.add(root / name, arcname=name, recursive=False)
    with tarfile.open(archive) as tar:
        restored = {
            m.name: hashlib.sha256(tar.extractfile(m).read()).hexdigest() for m in tar.getmembers()
        }
    if files != restored:
        raise ValueError("Archive verification failure")
    save(
        root.with_suffix(".preservation.json"),
        {"files": files, "archive_sha256": sha(archive), "archive_roundtrip": "PASS"},
    )
    print(json.dumps({"run": str(root.resolve()), **summary}), flush=True)


if __name__ == "__main__":
    main()
