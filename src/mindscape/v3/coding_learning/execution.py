"""Independent host comparison with actual WASI execution and mutation observations."""

import ast
import hashlib
import json
import secrets
from dataclasses import asdict
from pathlib import Path

from mindscape.coding.sandbox import WasiSandbox

FORBIDDEN = {
    "eval",
    "exec",
    "compile",
    "__import__",
    "open",
    "globals",
    "locals",
    "vars",
    "getattr",
    "setattr",
    "delattr",
    "dir",
    "breakpoint",
    "input",
}


def safe_source(repository):
    if not repository or len(repository) > 8:
        raise ValueError("Repository file budget exceeded")
    modules = {p[:-3].replace("/", ".") for p in repository}
    for path, source in repository.items():
        if Path(path).is_absolute() or ".." in Path(path).parts or not path.endswith(".py"):
            raise ValueError("Unsafe repository path")
        if len(source) > 16000:
            raise ValueError("Source budget exceeded")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in FORBIDDEN:
                raise ValueError("Reflective/executable capability forbidden")
            if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
                raise ValueError("Reflective attribute forbidden")
            if isinstance(node, ast.Import) and any(x.name not in modules for x in node.names):
                raise ValueError("External import forbidden")
            if isinstance(node, ast.ImportFrom) and (node.level or node.module not in modules):
                raise ValueError("External import forbidden")
    return True


class Executor:
    def __init__(self, runtime):
        runtime = Path(runtime)
        pins = json.loads(Path("configs/coding/runtime_v1.json").read_text())
        sha = hashlib.sha256((runtime / "python.wasm").read_bytes()).hexdigest()
        if sha != pins["python_wasm_sha256"]:
            raise ValueError("Unpinned WASI binary")
        # Historical runtime is immutable: never invoke its compiler or rewrite its files.
        if not (runtime / "python.cwasm").is_file():
            raise ValueError("Precompiled preserved runtime required")
        self.sandbox = WasiSandbox(runtime)
        self.calls = 0

    def run(self, repository, suite):
        self.calls += 1
        try:
            safe_source(repository)
        except (SyntaxError, ValueError) as error:
            return {
                "all_passed": False,
                "passed": 0,
                "total": len(suite["cases"]),
                "case_passes": [False] * len(suite["cases"]),
                "error": str(error),
                "execution": None,
                "observed": [],
                "actual_vm_call": False,
            }
        module, name = suite["entry"].rsplit(".", 1)
        nonce = secrets.token_hex(16)
        inputs = [c["args"] for c in suite["cases"]]
        program = f"""import importlib,json,copy,traceback
fn=getattr(importlib.import_module({module!r}),{name!r})
observed=[]
for original in {inputs!r}:
 args=copy.deepcopy(original)
 try:
  value=fn(*args)
  observed.append({{"value":value,"type":type(value).__name__,"args_after":args}})
 except BaseException as error:
  observed.append({{"error":type(error).__name__,"trace":traceback.format_exc()[-1200:]}})
print({nonce!r}+json.dumps(observed))
"""
        result = self.sandbox.execute(repository, program)
        try:
            observed = json.loads(
                next(
                    line[len(nonce) :]
                    for line in result.stdout.splitlines()
                    if line.startswith(nonce)
                )
            )
            if not isinstance(observed, list):
                observed = []
        except (ValueError, StopIteration):
            observed = []
        passed = []
        for i, case in enumerate(suite["cases"]):
            actual = observed[i] if i < len(observed) else {}
            okay = (
                isinstance(actual, dict)
                and result.returncode == 0
                and actual.get("value") == case["expected"]
                and actual.get("type") == case["expected_type"]
                and "error" not in actual
            )
            if case.get("unchanged"):
                okay = okay and actual.get("args_after") == case["args"]
            passed.append(bool(okay))
        return {
            "all_passed": bool(passed) and all(passed),
            "passed": sum(passed),
            "total": len(passed),
            "case_passes": passed,
            "observed": observed,
            "execution": asdict(result),
            "actual_vm_call": True,
        }
