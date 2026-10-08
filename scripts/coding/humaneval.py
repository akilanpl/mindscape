"""Pinned full HumanEval, one greedy independent completion; no repair feedback."""

import argparse
import gzip
import hashlib
import json
import statistics
from pathlib import Path

from mindscape.coding.model import LocalCoder, source_from_response
from mindscape.coding.sandbox import WasiSandbox

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--output", default="results/coding/humaneval_v1")
a = p.parse_args()
root = Path(a.output)
root.mkdir(parents=True, exist_ok=True)
source = Path("work/coding/humaneval")
manifest = json.loads((source / "manifest.json").read_text())
payload = (source / "HumanEval.jsonl.gz").read_bytes()
assert hashlib.sha256(payload).hexdigest() == manifest["sha256"]
tasks = [json.loads(line) for line in gzip.decompress(payload).splitlines()]
coder = LocalCoder(a.model)
sandbox = WasiSandbox("work/coding/runtime", timeout=10)
sandbox.probe()
rows_path = root / "rows.jsonl"
rows = [json.loads(x) for x in rows_path.read_text().splitlines()] if rows_path.exists() else []
done = {r["task_id"] for r in rows}
protocol = {
    "dataset": manifest,
    "model_revision": coder.revision,
    "sampling": "greedy",
    "samples_per_task": 1,
    "max_new_tokens": 512,
    "test_feedback": False,
    "runtime": "CPython 3.14.7 WASI",
    "pass_k": {
        "1": "mean independent completion success",
        "k>1": "unavailable: one sample per task",
    },
    "contamination": "Pretraining exposure unknown; public benchmark cannot establish contamination-free generalization",
}
(root / "protocol.json").write_text(json.dumps(protocol, indent=2))
for t in tasks:
    if t["task_id"] in done:
        continue
    response = coder.generate(
        "Complete the specified Python function. Return the full Python module in one code block, preserving its signature. No explanation.",
        t["prompt"],
        512,
    )
    error = None
    code = ""
    try:
        code = source_from_response(response)
    except ValueError as exc:
        error = str(exc)
    nonce = "RESULT_" + hashlib.sha256((t["task_id"] + coder.revision).encode()).hexdigest()
    program = (
        "import candidate\n"
        + t["test"]
        + "\ncheck(candidate."
        + t["entry_point"]
        + ")\nprint("
        + repr(nonce)
        + ")\n"
    )
    result = sandbox.execute({"candidate.py": code}, program) if not error else None
    passed = bool(result and result.returncode == 0 and nonce in result.stdout.splitlines())
    row = {
        "task_id": t["task_id"],
        "success": passed,
        "response": response,
        "code": code,
        "parse_error": error,
        "execution": vars(result) if result else None,
        "latency": coder.timings[-1],
    }
    with rows_path.open("a") as f:
        f.write(json.dumps(row) + "\n")
    rows.append(row)
    print(t["task_id"], passed, len(rows), flush=True)
summary = {
    "protocol": protocol,
    "parameters": coder.parameter_count,
    "tasks": len(rows),
    "successes": sum(r["success"] for r in rows),
    "pass@1": sum(r["success"] for r in rows) / len(rows),
    "mean_generation_seconds": statistics.mean(r["latency"]["generation"] for r in rows),
}
(root / "summary.json").write_text(json.dumps(summary, indent=2))
print(summary, flush=True)
