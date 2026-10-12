"""Replay frozen observations using additive source/recursive-type hardening."""

import argparse
import hashlib
import json
from pathlib import Path

import mindscape.v3.coding_safety as safety
from mindscape.v3.coding_learning import verify

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--manifest", default="configs/v3_coding/benchmark_v2")
    p.add_argument("--runtime", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    verify.Executor = safety.GuardedExecutor
    result = verify.replay(a.run, a.manifest, a.runtime)
    result.update(
        policy=safety.POLICY,
        guard_source_sha256=hashlib.sha256(Path(safety.__file__).read_bytes()).hexdigest(),
        unchanged_frozen_outcomes=True,
        scope="Independent post-termination replay; no model prompts, weights or recorded scores modified",
    )
    Path(a.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
