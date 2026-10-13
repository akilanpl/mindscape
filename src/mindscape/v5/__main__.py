"""One bounded native training launcher, conditional on committed DATA_READY receipts."""

import hashlib
import json
import subprocess
import time
from pathlib import Path

from .qualify import save


def main():
    root = Path("results/v5/native-pilot")
    if root.exists():
        raise FileExistsError("Single-attempt pilot already exists; no automatic retry")
    qualification = Path("experiments/v5/qualification_v1")
    receipt = json.loads((qualification / "qualification.json").read_text())
    if receipt["decision"] != "DATA_READY":
        raise RuntimeError("Qualification not DATA_READY; training forbidden")
    freeze = json.loads(Path("configs/v5/source-freeze.json").read_text())
    for name, expected in freeze["sources"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError("Frozen source mismatch: " + name)
    for name, expected in freeze["qualification_files"].items():
        p = qualification / name
        if hashlib.sha256(p.read_bytes()).hexdigest() != expected:
            raise ValueError("Qualification receipt mismatch")
        if subprocess.check_output(["git", "show", "HEAD:" + str(p)]) != p.read_bytes():
            raise ValueError("Qualification must be committed before training")
    manifest = json.loads((qualification / "manifest.json").read_text())
    rows = json.loads((qualification / "rows.json").read_text())
    sealed = []
    for item in json.loads((qualification / "sealed-index.json").read_text()):
        p = qualification / item["file"]
        if hashlib.sha256(p.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("Sealed composition hash changed")
        sealed.append(json.loads(p.read_text()))
    root.mkdir(parents=True, exist_ok=False)
    save(
        root / "manifest.json",
        {
            "protocol": manifest["protocol"],
            "source_freeze": freeze,
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "composition_read_for": "post-freeze evaluation only; never optimizer or model input",
        },
    )
    started = time.perf_counter()
    try:
        from mindscape.v3.physics_world.accelerator import require_mps

        require_mps()
        from .native_checks import check

        demonstration = [
            next(r["graph"] for r in rows if r["domain"] == d)
            for d in ("mathematics", "coding", "physics")
        ]
        save(root / "native-engineering-checks.json", check(demonstration))
        from .learning import run

        summary = run(rows, sealed, manifest["protocol"], root / "fits")
        summary["seconds_total"] = time.perf_counter() - started
        save(root / "summary.json", summary)
        from .replay import replay

        audit = replay(root, qualification)
        save(root / "independent-replay.json", audit)
        print(
            json.dumps(
                {
                    "run": str(root.resolve()),
                    "decision": summary["decision"],
                    "gates": summary["gates"],
                }
            ),
            flush=True,
        )
    except (RuntimeError, ValueError, OSError, KeyError, TypeError) as exc:
        save(
            root / "failure.json",
            {
                "type": type(exc).__name__,
                "message": str(exc),
                "seconds": time.perf_counter() - started,
            },
        )
        raise


if __name__ == "__main__":
    main()
