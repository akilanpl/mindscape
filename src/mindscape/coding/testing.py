"""Host-authoritative comparisons; expected outputs never enter guest code."""

import json
import secrets
from dataclasses import asdict


def run_cases(sandbox, repository, suite):
    spec = json.loads(suite)
    module, function = spec["entry"].rsplit(".", 1)
    args = [c["args"] for c in spec["cases"]]
    nonce = secrets.token_hex(16)
    program = f"""import importlib,json,traceback
module=importlib.import_module({module!r})
function=getattr(module,{function!r})
results=[]
for args in {args!r}:
 try:
  value=function(*args)
  results.append({{"value":value,"type":type(value).__name__}})
 except BaseException as error:
  results.append({{"error":type(error).__name__}})
print({nonce!r}+json.dumps(results))
"""
    actual = sandbox.execute(repository, program)
    results = []
    try:
        line = next(
            line[len(nonce) :] for line in actual.stdout.splitlines() if line.startswith(nonce)
        )
        results = json.loads(line)
    except (StopIteration, ValueError):
        pass
    passed = []
    for i, case in enumerate(spec["cases"]):
        value = results[i] if i < len(results) else {}
        passed.append(
            actual.returncode == 0
            and value.get("type") == case["expected_type"]
            and value.get("value") == case["expected"]
            and "error" not in value
        )
    return {
        "all_passed": bool(passed) and all(passed),
        "passed": sum(passed),
        "total": len(passed),
        "case_passes": passed,
        "execution": asdict(actual),
        "observed_values": results,
    }
