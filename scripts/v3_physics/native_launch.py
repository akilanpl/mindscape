"""Manual native launch orchestration; subprocesses inherit the real native boundary."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from mindscape.v3.physics_world.accelerator import require_mps
from mindscape.v3.physics_world.storage import atomic_json, file_sha, source_hash

ROOT = Path(__file__).resolve().parents[2]
PINNED_PYTHON = Path(
    "/Users/akilan/Documents/Codex/2026-10-08/files-pasted-by-the-user-mindscape/outputs/mindscape/work/final-venv/bin/python"
)


def launch(args):
    if Path(sys.executable).absolute() != PINNED_PYTHON.absolute():
        raise RuntimeError("Use pinned research interpreter")
    os.chdir(ROOT)
    require_mps()
    # Do not unset sandbox indicators: they document where this launcher was invoked.
    root = (
        Path(args.resume)
        if args.resume
        else ROOT / "results/v3_physics" / f"native-{args.mode}-{time.time_ns()}"
    )
    if args.resume:
        if not root.is_dir():
            raise ValueError("Resume directory missing")
        saved = json.loads((root / "launch.json").read_text())
        if saved["source_hash"] != source_hash() or saved["mode"] != args.mode:
            raise ValueError("Launch resume source/mode changed")
    else:
        root.mkdir(parents=True, exist_ok=False)
        atomic_json(
            root / "launch.json",
            {
                "mode": args.mode,
                "interpreter": sys.executable,
                "source_hash": source_hash(),
                "protocol_sha256": file_sha(ROOT / "configs/v3_physics/protocol.json"),
                "device": "mps",
                "precision": "float32",
                "environment_indicators": {
                    k: os.getenv(k)
                    for k in ("CODEX_SANDBOX", "CODEX_CI", "PYTORCH_ENABLE_MPS_FALLBACK")
                },
                "created_ns": time.time_ns(),
                "working_tree": str(ROOT),
                "status": "started",
            },
        )

    def execute(command, label):
        started = time.perf_counter()
        log = root / f"{label}-{time.time_ns()}.log"
        print(f"{label}: {log}", flush=True)
        with log.open("w") as stream:
            result = subprocess.run(
                [sys.executable, *command], stdout=stream, stderr=subprocess.STDOUT, check=False
            )
        atomic_json(
            root / f"{label}-receipt-{time.time_ns()}.json",
            {
                "command": [sys.executable, *command],
                "returncode": result.returncode,
                "seconds": time.perf_counter() - started,
                "log": log.name,
            },
        )
        if result.returncode:
            raise RuntimeError(f"{label} failed; inspect {log}")

    if args.mode == "pilot":
        audit = root / f"hardware-audit-{time.time_ns()}.json"
        execute(["scripts/v3_physics/hardware_audit.py", "--output", str(audit)], "hardware-audit")
        hardware = json.loads(audit.read_text())
        batch = hardware["selected_batch_size"]
        dataset = ROOT / "results/v3_physics/dataset-pilot-v1"
        execute(["scripts/v3_physics/prepare.py", "--output", str(dataset), "--pilot"], "prepare")
        execute(
            [
                "scripts/v3_physics/native_integration.py",
                "--dataset",
                str(dataset),
                "--output",
                str(root / "resume-integration"),
            ],
            "native-resume-integration",
        )
        execute(
            [
                "scripts/v3_physics/run_study.py",
                "--dataset",
                str(dataset),
                "--output",
                str(root / "study"),
                "--pilot",
                "--batch-size",
                str(batch),
            ],
            "pilot-training",
        )
        execute(
            [
                "scripts/v3_physics/analyze.py",
                "--run",
                str(root / "study"),
                "--output",
                str(root / "analysis.json"),
            ],
            "pilot-analysis",
        )
        execute(
            [
                "scripts/v3_physics/verify.py",
                "--dataset",
                str(dataset),
                "--run",
                str(root / "study"),
                "--output",
                str(root / "verification.json"),
            ],
            "pilot-verification",
        )
        if json.loads((root / "analysis.json").read_text())["primary_completed"] != 4:
            raise RuntimeError("Pilot incomplete")
        atomic_json(
            root / "pilot-pass.json",
            {
                "pilot_passed": True,
                "source_hash": source_hash(),
                "batch_size": batch,
                "device": "mps",
                "precision": "float32",
                "verification": "completed",
                "scope": "small pilot, not full acceptance",
            },
        )
    else:
        pilot = Path(args.pilot_receipt) if args.pilot_receipt else None
        if pilot is None or not pilot.is_file():
            raise ValueError("Verified pilot receipt required")
        evidence = json.loads(pilot.read_text())
        if not evidence.get("pilot_passed") or evidence["source_hash"] != source_hash():
            raise ValueError("Pilot stale or failed")
        dataset = ROOT / "results/v3_physics/dataset-full-v1"
        execute(["scripts/v3_physics/prepare.py", "--output", str(dataset)], "prepare")
        execute(
            [
                "scripts/v3_physics/run_study.py",
                "--dataset",
                str(dataset),
                "--output",
                str(root / "study"),
                "--batch-size",
                str(evidence["batch_size"]),
            ],
            "full-training-evaluation",
        )
        execute(
            [
                "scripts/v3_physics/analyze.py",
                "--run",
                str(root / "study"),
                "--output",
                str(root / "analysis.json"),
            ],
            "analysis",
        )
        execute(
            [
                "scripts/v3_physics/verify.py",
                "--dataset",
                str(dataset),
                "--run",
                str(root / "study"),
                "--output",
                str(root / "verification.json"),
            ],
            "verification",
        )
    print(
        json.dumps(
            {
                "run": str(root),
                "mode": args.mode,
                "status": "driver_finished; inspect acceptance receipts",
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("pilot", "full"), default="pilot")
    p.add_argument("--resume")
    p.add_argument("--pilot-receipt")
    launch(p.parse_args())
