"""Native launcher with captured logs, source-bound resume and independent verification."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from mindscape.v3.physics_world.accelerator import require_mps
from mindscape.v3.physics_world.storage import atomic_json

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--resume")
    a = p.parse_args()
    require_mps()
    root = (
        Path(a.resume) if a.resume else Path(f"results/v3_1_physics/native-pilot-{time.time_ns()}")
    )
    root.mkdir(parents=True, exist_ok=bool(a.resume))
    atomic_json(
        root / f"launch-{time.time_ns()}.json",
        {
            "device": "mps",
            "precision": "float32",
            "interpreter": sys.executable,
            "fallback": os.getenv("PYTORCH_ENABLE_MPS_FALLBACK"),
            "sandbox": os.getenv("CODEX_SANDBOX"),
            "mode": "development pilot; no sealed access",
        },
    )
    for stage, cmd in [
        ("pilot", ["scripts/v3_1_physics/pilot.py", str(root)]),
        ("verify", ["scripts/v3_1_physics/verify.py", str(root)]),
    ]:
        started = time.perf_counter()
        with (root / f"{stage}-{time.time_ns()}.log").open("w") as log:
            result = subprocess.run(
                [sys.executable, *cmd], stdout=log, stderr=subprocess.STDOUT, check=False
            )
        atomic_json(
            root / f"{stage}-receipt-{time.time_ns()}.json",
            {"returncode": result.returncode, "seconds": time.perf_counter() - started},
        )
        if result.returncode:
            raise RuntimeError(f"{stage} failed; preserve {root}")
    analysis = json.loads((root / "analysis.json").read_text())
    print(
        json.dumps(
            {
                "run": str(root),
                "verified": True,
                "development_gates_passed": analysis["passed"],
                "decision": "PROPOSE VALIDATION"
                if analysis["passed"]
                else "NO-GO; investigate before scaling",
            }
        )
    )
