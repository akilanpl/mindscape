"""Actual WASI upper bound and controller-negative controls; no neural claim."""

import argparse
import json
from pathlib import Path

from mindscape.v3.coding_learning.data import load_public, load_scorer
from mindscape.v3.coding_learning.execution import Executor


def run(manifest, runtime, output):
    tasks, _ = load_public(manifest)
    private = load_scorer(manifest)
    executor = Executor(runtime)
    rows = []
    for task in tasks:
        if task.split not in ("iid", "ood"):
            continue
        zero = dict(task.repository)
        module = task.contract.entry.rsplit(".", 1)[0] + ".py"
        # Trusted construction of an intentionally uninformative proposal; guest-only execution.
        import ast

        tree = ast.parse(zero[module])
        function_name = task.contract.entry.rsplit(".", 1)[1]
        function = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function_name
        )
        function.body = [ast.Return(value=ast.Constant(value=0))]
        zero[module] = ast.unparse(tree) + "\n"
        for name, repository in (
            ("unchanged_controller_negative", task.repository),
            ("constant_zero_proposal_negative", zero),
            ("teacher_reference_upper_bound", private[task.identity]["teacher_repository"]),
        ):
            visible = executor.run(repository, task.visible_tests)
            terminal = executor.run(repository, private[task.identity]["private_tests"])
            rows.append(
                {
                    "task_id": task.identity,
                    "split": task.split,
                    "family": task.family,
                    "condition": name,
                    "repository": repository,
                    "visible": visible,
                    "terminal": terminal,
                }
            )
    result = {
        "scope": "Software controls and explicit oracle; not zero-Qwen-weight experiments",
        "rows": rows,
        "sandbox_requests": executor.calls,
        "model_calls": 0,
    }
    Path(output).write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                name: sum(r["terminal"]["all_passed"] for r in rows if r["condition"] == name)
                for name in sorted({r["condition"] for r in rows})
            }
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="configs/v3_coding/benchmark_v2")
    p.add_argument("--runtime", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    run(a.manifest, a.runtime, a.output)
