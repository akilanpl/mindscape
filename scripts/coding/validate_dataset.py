"""Execute every generated correct/buggy function; host-only expected outputs.

Batches contain only trusted procedural generator output, not untrusted model code.
Production policy execution still isolates each repository independently.
"""

import hashlib
import json
import secrets
from pathlib import Path

from mindscape.coding.sandbox import WasiSandbox

root = Path("results/coding/final_dataset_v1")
dataset = json.loads((root / "dataset.json").read_text())
tasks = [t for ts in dataset.values() for t in ts]
sandbox = WasiSandbox("work/coding/runtime", timeout=10)
records = []
for offset in range(0, len(tasks), 50):
    batch = tasks[offset : offset + 50]
    for version in ("repository", "correct_repository"):
        guest_inputs = [
            {
                "repository": t[version],
                "entry": json.loads(t["hidden_tests"])["entry"],
                "args": [c["args"] for c in json.loads(t["hidden_tests"])["cases"]],
            }
            for t in batch
        ]
        nonce = secrets.token_hex(16)
        program = (
            """import json,sys,types
all_results=[]
for task in """
            + repr(guest_inputs)
            + """:
 module_name,function_name=task['entry'].rsplit('.',1)
 for filename,source in task['repository'].items():
  if filename!=module_name+'.py':
   name=filename[:-3];helper=types.ModuleType(name);exec(compile(source,filename,'exec'),helper.__dict__);sys.modules[name]=helper
 namespace={};exec(compile(task['repository'][module_name+'.py'],module_name+'.py','exec'),namespace)
 function=namespace[function_name];observed=[]
 for args in task['args']:
  try:
   value=function(*args);observed.append({'type':type(value).__name__,'value':value})
  except BaseException as error:observed.append({'error':type(error).__name__})
 all_results.append(observed)
print("""
            + repr(nonce)
            + """+json.dumps(all_results))
"""
        )
        execution = sandbox.execute({}, program)
        if execution.returncode:
            raise RuntimeError("Generator validation batch failed: " + execution.stderr)
        observed = json.loads(
            next(s[len(nonce) :] for s in execution.stdout.splitlines() if s.startswith(nonce))
        )
        if len(observed) != len(batch):
            raise RuntimeError("Incomplete batch")
        for t, values in zip(batch, observed):
            cases = json.loads(t["hidden_tests"])["cases"]
            passed = [
                value.get("type") == case["expected_type"]
                and value.get("value") == case["expected"]
                and "error" not in value
                for value, case in zip(values, cases)
            ]
            all_passed = len(values) == len(cases) and all(passed)
            if all_passed != (version == "correct_repository"):
                raise RuntimeError("Inconsistent correct/buggy pair: " + t["task_id"])
            records.append(
                {
                    "task_id": t["task_id"],
                    "version": version,
                    "all_passed": all_passed,
                    "case_passes": passed,
                    "actual_values": values,
                    "batch_offset": offset,
                    "execution_seconds": execution.duration,
                }
            )
    print("validated task pairs", min(offset + 50, len(tasks)), flush=True)
(root / "validation.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
manifest = {
    "tasks": len(tasks),
    "actual_function_invocations": sum(len(r["case_passes"]) for r in records),
    "correct_all_passed": True,
    "buggy_all_failed": True,
    "validation_sha256": hashlib.sha256((root / "validation.jsonl").read_bytes()).hexdigest(),
    "trusted_generated_code_batched": True,
    "untrusted_model_code_batched": False,
}
(root / "validation_summary.json").write_text(json.dumps(manifest, indent=2))
print(manifest)
