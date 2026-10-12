"""Resume frozen CPU studies to coverage; never resample completed failures."""

import argparse
import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path


def count(root):
    return sum(1 for p in root.glob("*.json") if len(p.name.split("-")[0]) == 64)


def complete(args):
    evidence = Path(args.output)
    receipts = []
    for seed in (11, 23, 37):
        root = evidence / f"transport_v2_s{seed}"
        root.mkdir(parents=True, exist_ok=True)
        # Wait for a separately started invocation to release its advisory lock.
        with (root / "runner.lock").open("a") as lock:
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    time.sleep(5)
            fcntl.flock(lock, fcntl.LOCK_UN)
        stalled = 0
        for invocation in range(8):
            before = count(root)
            command = [
                sys.executable,
                "scripts/v3_coding/transport_v2_study.py",
                "--model",
                args.model,
                "--runtime",
                args.runtime,
                "--training",
                args.training,
                "--output",
                str(root),
                "--seed",
                str(seed),
                "--seconds",
                "1800",
            ]
            log = evidence / f"completion-s{seed}-{invocation}.txt"
            start = time.perf_counter()
            with log.open("w") as stream:
                result = subprocess.run(
                    command, stdout=stream, stderr=subprocess.STDOUT, check=False
                )
            after = count(root)
            receipts.append(
                {
                    "seed": seed,
                    "before": before,
                    "after": after,
                    "returncode": result.returncode,
                    "seconds": time.perf_counter() - start,
                    "command": command,
                    "log": str(log),
                }
            )
            (evidence / "completion-receipts.json").write_text(json.dumps(receipts, indent=2))
            print(json.dumps(receipts[-1]), flush=True)
            if result.returncode:
                raise RuntimeError("Study failed; inspect preserved receipt")
            if after == 144:
                # Running once with before144 also exercises the hash-bound no-op resume.
                if before == 144:
                    break
                continue
            stalled = stalled + 1 if before == after else 0
            if stalled >= 3:
                raise RuntimeError("Three invocations without coverage progress")
        else:
            raise RuntimeError("Explicit invocation resource limit reached")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--runtime", required=True)
    p.add_argument("--training", required=True)
    p.add_argument("--output", default="results/v3_coding")
    complete(p.parse_args())
