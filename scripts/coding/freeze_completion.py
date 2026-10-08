"""Immutable, hash-addressed completion snapshot; historical final trees survive."""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

sources = {
    "dataset": Path("results/coding/final_dataset_v1"),
    "lockbox": Path("results/coding/completion_lockbox_v1"),
    "lockbox_validation": Path("results/coding/completion_lockbox_validation_v1"),
    "patch_training": Path("results/coding/gradient_v1"),
    "gradient_study": Path("results/coding/completion_gradient_v1"),
    "actual_collection": Path("results/coding/completion_collection_v1"),
    "actual_tuples": Path("results/coding/teacher_traces_v1"),
    "final_evaluation": Path("results/coding/completion_lockbox_eval_v1"),
    "zero_training": Path("results/coding/zero_budget_v1"),
    "stress": Path("results/coding/stress_v1"),
    "fresh_latency": Path("results/coding/latency_probe_v1"),
    "trace_audit": Path("results/coding/completion_trace_audit_v1"),
    "analysis": Path("results/coding/completion_analysis_v1"),
    "audit": Path("results/coding/completion_audits_v1"),
    "public_benchmark_input": Path("work/coding/humaneval"),
    "humaneval_05b": Path("results/coding/humaneval_05b_v1"),
    "humaneval_15b": Path("results/coding/humaneval_15b_v1"),
    "sql": Path("results/sql/portability_v1"),
    "configs": Path("configs"),
    "scripts": Path("scripts"),
    "docs": Path("docs"),
    "experiments": Path("experiments"),
    "demo": Path("demo/coding"),
    "tests": Path("tests"),
    "wasi_runtime": Path("work/coding/runtime"),
}
for label in (
    "gradient_study",
    "final_evaluation",
    "zero_training",
    "stress",
    "fresh_latency",
    "trace_audit",
    "sql",
):
    if not (sources[label] / "complete.json").exists():
        raise RuntimeError("Incomplete " + label)
for label in ("humaneval_05b", "humaneval_15b"):
    if not (sources[label] / "summary.json").exists():
        raise RuntimeError("Incomplete " + label)
if not (sources["analysis"] / "report_ready.json").exists():
    raise RuntimeError("Final analysis/report not ready")
tests = json.loads((sources["experiments"] / "coding_completion_v2/test_results.json").read_text())
if tests["failures"] or tests["errors"]:
    raise RuntimeError("Failing final tests")
root = Path("results/final/coding_research_v2")
if root.exists():
    raise RuntimeError("Final snapshot already exists; never overwrite")
staging = Path("results/final/.coding_research_v2_staging")
if staging.exists():
    raise RuntimeError("Interrupted snapshot requires inspection before resuming")
staging.mkdir(parents=True)
for label, path in sources.items():
    shutil.copytree(path, staging / label, ignore=shutil.ignore_patterns("__pycache__", ".DS_Store"))
shutil.copytree(
    "src/mindscape", staging / "source/mindscape", ignore=shutil.ignore_patterns("__pycache__")
)
shutil.copy2("pyproject.toml", staging / "pyproject.toml")
shutil.copy2("README.md", staging / "README.md")
files = {
    str(path.relative_to(staging)): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted(staging.rglob("*"))
    if path.is_file()
}
manifest = {
    "release": "mindscape-research-v2",
    "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "files": files,
    "source_paths": {label: str(path) for label, path in sources.items()},
    "foundation_models": {
        "Qwen/Qwen2.5-Coder-0.5B-Instruct": "ea3f2471cf1b1f0db85067f1ef93848e38e88c25",
        "Qwen/Qwen2.5-Coder-1.5B-Instruct": "2e1fd397ee46e1388853d2af2c993145b0f1098a",
    },
    "foundation_weights": "Pinned revisions remain in local HF cache; bootstrap downloads exact revisions before offline reproduction",
    "historical_release": "mindscape-final-study-v1 retained",
    "paid_services_usd": 0,
}
(staging / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
staging.rename(root)
print("Frozen files", len(files), "root", root)
