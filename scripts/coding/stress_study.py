"""Matched diagnostic ablations and executed robustness; never tune the lockbox."""

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from mindscape.coding.instrumentation import StageMeter
from mindscape.coding.memory import CodingMemory
from mindscape.coding.model import LocalCoder
from mindscape.coding.policy import ToolCodingPolicy
from mindscape.coding.robustness import perturb
from mindscape.coding.sandbox import WasiSandbox
from mindscape.coding.schema import CodeTask
from mindscape.coding.testing import run_cases

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--output", default="results/coding/stress_v1")
a = p.parse_args()
root = Path(a.output)
root.mkdir(parents=True, exist_ok=True)
data = json.loads(Path("results/coding/final_dataset_v1/dataset.json").read_text())
teacher = json.loads(
    Path("results/coding/completion_collection_v1/teacher_memory.json").read_text()
)
sandbox = WasiSandbox("work/coding/runtime")
coder = LocalCoder(
    a.model,
    adapter="results/coding/completion_gradient_v1/mindscape_c/seed_11/checkpoint_100",
    cache_enabled=False,
)


class ExplicitGoalMask:
    def __getattr__(self, name):
        return getattr(coder, name)

    def generate(self, system, prompt, *args):
        payload = json.loads(prompt)
        payload.pop("goal", None)
        return coder.generate(system, json.dumps(payload), *args)


path = root / "rows.jsonl"
done = (
    {tuple(json.loads(s)["key"]) for s in path.read_text().splitlines()} if path.exists() else set()
)
base = [CodeTask(**v) for v in data["test"][:10] + data["ood_test"][:4]]
cases = [
    ("ablation", ablation, task)
    for ablation in (
        None,
        "no_state",
        "no_relation_graph",
        "no_memory",
        "no_goal",
        "no_trajectory",
        "no_environment_feedback",
        "no_dream",
    )
    for task in base
]
for variant in (
    "rename_functions",
    "rename_variables",
    "reordered_files",
    "reordered_tests",
    "distractor_files",
    "layout",
    "formatting",
):
    for task in base:
        changed = perturb(task, variant)
        if (
            changed.repository == task.repository
            and list(changed.repository) == list(task.repository)
            and changed.visible_tests == task.visible_tests
        ):
            continue
        cases.append(("robustness", variant, changed))
for index, fault in enumerate(("syntax", "wrong_patch", "regression", "timeout")):
    task = base[index]
    entry = json.loads(task.visible_tests)["entry"]
    module, function = entry.rsplit(".", 1)
    source = {
        "syntax": f"def {function}(:\n",
        "wrong_patch": f"def {function}(*args):\n    return 0\n",
        "regression": f"def {function}(*args):\n    return None\n",
        "timeout": f"def {function}(*args):\n    while True: pass\n",
    }[fault]
    cases.append(
        (
            "recovery",
            fault,
            replace(
                task,
                task_id=task.task_id + "_" + fault,
                repository=task.repository | {module + ".py": source},
            ),
        )
    )
for kind, variant, task in cases:
    key = (kind, variant, task.task_id)
    if key in done:
        continue
    if kind == "robustness":
        if not run_cases(sandbox, task.correct_repository, task.hidden_tests)["all_passed"]:
            raise RuntimeError("Perturbation violates specification")
        if run_cases(sandbox, task.repository, task.hidden_tests)["all_passed"]:
            raise RuntimeError("Perturbation removes defect")
    policy = ToolCodingPolicy(
        ExplicitGoalMask() if kind == "ablation" and variant == "no_goal" else coder,
        sandbox,
        CodingMemory(episodic=teacher[:100]),
        ablation=variant if kind == "ablation" else None,
        max_edits=5 if kind == "recovery" else 3,
        max_steps=12 if kind == "recovery" else 8,
    )
    initial_fault = (
        run_cases(sandbox, task.repository, task.visible_tests) if kind == "recovery" else None
    )
    with StageMeter() as meter:
        row = policy.solve(task)
    snapshots = row.get("edit_snapshots", [])
    verdicts = [run_cases(sandbox, files, task.hidden_tests)["all_passed"] for files in snapshots]
    row.update(
        key=key,
        kind=kind,
        variant=variant,
        split=task.metadata["split"],
        stages=meter.summary(),
        generation_cache_enabled=False,
        success_by_edit=verdicts,
        assessment_timing="All private snapshot assessments occur after policy termination",
        injected_fault_execution=initial_fault,
        interactive_success_at_k={str(k): any(verdicts[:k]) for k in (1, 3, 5)},
        ablation_scope="Inference component removal; trained weights remain fixed",
    )
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")
    done.add(key)
    print(kind, variant, len(done), flush=True)
patch_adapter = Path("results/coding/gradient_v1/seed_11/checkpoint_100")
coder.model.load_adapter(patch_adapter, adapter_name="patch_only")
coder.model.set_adapter("patch_only")
coder.revision = (
    Path(a.model).name
    + ":patch_only:"
    + hashlib.sha256((patch_adapter / "adapter_model.safetensors").read_bytes()).hexdigest()
)
for task in base:
    key = ("ablation", "no_trajectory_supervision", task.task_id)
    if key in done:
        continue
    policy = ToolCodingPolicy(coder, sandbox, CodingMemory(episodic=teacher[:100]))
    with StageMeter() as meter:
        row = policy.solve(task)
    row.update(
        key=key,
        kind="ablation",
        variant="no_trajectory_supervision",
        split=task.metadata["split"],
        stages=meter.summary(),
        generation_cache_enabled=False,
        ablation_scope="Patch-only trained adapter replaces action/replay adapter; same unique tasks and seed, but training inputs/labels also differ. Diagnostic, not isolated causal removal.",
    )
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")
    done.add(key)
    print("ablation", "no_trajectory_supervision", len(done), flush=True)
(root / "complete.json").write_text(json.dumps({"episodes": len(done), "complete": True}))
