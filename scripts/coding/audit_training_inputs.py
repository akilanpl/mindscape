"""Execute trusted trainer preprocessing only, inspect exact learner inputs."""

import ast
import json
from argparse import Namespace
from pathlib import Path

from transformers import AutoTokenizer

from mindscape.coding.policy import _model_view
from mindscape.coding.schema import CodeTask

source = ast.parse(Path("scripts/coding/train_regime.py").read_text())
loop = next(
    n
    for n in source.body
    if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "t"
)
end = next(
    i
    for i, n in enumerate(loop.body)
    if isinstance(n, ast.Assign)
    and any(isinstance(t, ast.Name) and t.id == "inputs" for t in n.targets)
)
preprocessing = compile(
    ast.Module(body=loop.body[:end], type_ignores=[]), "trusted_training_preprocessing", "exec"
)
tokenizer = AutoTokenizer.from_pretrained(
    "work/coding/hf/models--Qwen--Qwen2.5-Coder-1.5B-Instruct/snapshots/2e1fd397ee46e1388853d2af2c993145b0f1098a",
    local_files_only=True,
)
tasks = [
    CodeTask(**t)
    for t in json.loads(Path("results/coding/final_dataset_v1/dataset.json").read_text())["train"][
        :100
    ]
]
forbidden = {
    "hidden_tests",
    "correct_repository",
    "ground_truth_patch",
    "state_after",
    "next_state",
}


def keys(value):
    if isinstance(value, dict):
        return set(value).union(*(keys(v) for v in value.values()))
    if isinstance(value, (tuple, list)):
        return set().union(*(keys(v) for v in value))
    return set()


records = []
for condition in ("structured", "mindscape_b", "mindscape_c"):
    ns = {
        "a": Namespace(condition=condition),
        "Path": Path,
        "json": json,
        "_model_view": _model_view,
        "tokenizer": tokenizer,
        "replay": [],
    }
    lengths = []
    for task in tasks:
        ns["t"] = task
        exec(preprocessing, ns)  # noqa: S102 - trusted trainer statements; candidate programs remain strings
        assert not forbidden & keys(ns["payload"])
        lengths.append(len(ns["ids"]))
    total = sum(lengths)
    completed_records = list(
        Path("results/coding/completion_gradient_v1", condition).glob("seed_*/training.json")
    )
    for path in completed_records:
        assert json.loads(path.read_text())["tokens"] == total
    records.append(
        {
            "condition": condition,
            "examples": len(tasks),
            "max_training_sequence_tokens": max(lengths),
            "total_tokens": total,
            "forbidden_input_keys_absent": True,
            "training_record_tokens_match": True,
            "completed_training_records_checked": len(completed_records),
        }
    )
Path("results/coding/completion_audits_v1/training_inputs.json").write_text(
    json.dumps(
        {
            "actual_preprocessing_inspected": True,
            "records": records,
            "candidate_source_executed_on_host": False,
            "scope": "Executes trusted trainer input-building statements only; source programs remain strings",
        },
        indent=2,
    )
)
print(records)
