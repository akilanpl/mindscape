"""Validation-only model selection; every repair executes in WASI."""

import argparse
import json
import resource
from pathlib import Path

from mindscape.coding.generator import TRAIN, task
from mindscape.coding.model import LocalCoder, source_from_response
from mindscape.coding.sandbox import WasiSandbox
from mindscape.coding.testing import run_cases

p = argparse.ArgumentParser()
p.add_argument("model")
p.add_argument("output")
a = p.parse_args()
sandbox = WasiSandbox("work/coding/runtime")
coder = LocalCoder(a.model)
rows = []
for i, category in enumerate(TRAIN):
    t = task(8000 + i, category, "validation")
    visible = t.visible()
    path = next(iter(t.repository))
    prompt = (
        "Repair this Python module. Return the complete corrected module in a Python code block.\n"
        + json.dumps(visible)
    )
    response = coder.generate(
        "You repair Python programs. Preserve the public function signature.", prompt, 192
    )
    repo = dict(t.repository)
    error = None
    try:
        repo[path] = source_from_response(response)
        verdict = run_cases(sandbox, repo, t.hidden_tests)
    except (ValueError, TypeError, RuntimeError) as exc:
        error = str(exc)
        verdict = {"all_passed": False}
    row = {
        "task_id": t.task_id,
        "category": category,
        "response": response,
        "success": verdict["all_passed"],
        "error": error,
        "latency": coder.timings[-1],
    }
    rows.append(row)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(
        json.dumps(
            {
                "revision": coder.revision,
                "parameters": coder.parameter_count,
                "rows": rows,
                "peak_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            },
            indent=2,
        )
    )
    print(category, row["success"], round(row["latency"]["generation"], 2), flush=True)
