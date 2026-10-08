"""Small frozen SQL portability study after coding completion, zero SQL training."""

import argparse
import gc
import json
import time
from dataclasses import asdict
from pathlib import Path

from mindscape.coding.model import LocalCoder
from mindscape.data.schemas import BenchmarkExample
from mindscape.evaluation.schemas import PredictionRecord
from mindscape.sql.environment import SqlAction, SqlEnvironment, task
from mindscape.sql.policy import SqlBenchmarkModel

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--output", default="results/sql/portability_v1")
a = p.parse_args()
if not Path("results/coding/latency_probe_v1/complete.json").exists():
    raise RuntimeError("Coding study must finish before SQL inference")
root = Path(a.output)
root.mkdir(parents=True, exist_ok=True)
tasks = [task(i) for i in range(30)]
protocol = {
    "tasks": 30,
    "sql_training_examples": 0,
    "baseline": "unadapted selected backbone",
    "mindscape": "coding-trained mindscape_c budget100 seed11; SQL tools, no SQL adapter training",
    "purpose": "Interface portability and small measured SQL repair; not a SQL data-efficiency or learned-transfer claim",
}
(root / "protocol.json").write_text(json.dumps(protocol, indent=2))
(root / "tasks.json").write_text(json.dumps([asdict(t) for t in tasks], indent=2))
path = root / "rows.jsonl"
done = (
    {tuple(json.loads(s)["key"]) for s in path.read_text().splitlines()} if path.exists() else set()
)
for condition in ("model_only", "mindscape_c"):
    adapter = (
        "results/coding/completion_gradient_v1/mindscape_c/seed_11/checkpoint_100"
        if condition == "mindscape_c"
        else None
    )
    coder = LocalCoder(a.model, adapter=adapter)
    model = SqlBenchmarkModel(coder, condition == "mindscape_c")
    for example in tasks:
        key = (condition, example.task_id)
        if key in done:
            continue
        original = SqlEnvironment()
        initial_state = asdict(original.reset(example))
        if original.final_evaluate()["success"]:
            raise RuntimeError("SQL defect not discriminated")
        original.step(SqlAction("edit_query", example.target_query))
        if not original.final_evaluate()["success"]:
            raise RuntimeError("SQL oracle fails")
        began = time.perf_counter()
        benchmark = BenchmarkExample(
            example.task_id,
            "sql_correction",
            example.problem,
            example.visible(),
            initial_state,
            {"description": example.problem},
            example.target_query,
            [],
            {"private_rows": example.private_rows},
        )
        prediction = model.predict(benchmark.view("experiential"))
        evaluator = SqlEnvironment()
        evaluator.reset(example)
        evaluator.query = prediction.answer
        assessment = evaluator.final_evaluate()
        replay = SqlEnvironment()
        replay.reset(example)
        valid = True
        for transition in prediction.trajectory["transitions"]:
            if replay.query != transition["state_before"]["query"]:
                valid = False
            actual = replay.step(SqlAction(**transition["action"]))
            expected_result = {k: v for k, v in transition["result"].items() if k != "seconds"}
            actual_result = {k: v for k, v in actual.result.items() if k != "seconds"}
            if (
                not actual.valid
                or actual.event != transition["event"]
                or replay.query != transition["state_after"]["query"]
                or json.dumps(expected_result, sort_keys=True)
                != json.dumps(actual_result, sort_keys=True)
            ):
                valid = False
        valid = valid and replay.query == prediction.answer
        evaluation = PredictionRecord(
            example.task_id,
            prediction.answer,
            "independent private database",
            assessment["success"],
            prediction.trajectory,
            valid,
            assessment["success"],
            assessment["success"] and valid,
            "correct"
            if assessment["success"] and valid
            else "invalid_transition"
            if not valid
            else "goal_failure",
            prediction.diagnostics,
        )
        row = {
            "key": key,
            "condition": condition,
            "task_id": example.task_id,
            "success": assessment["success"],
            "prediction": asdict(prediction),
            "evaluation": asdict(evaluation),
            "terminal": assessment,
            "wall_seconds": time.perf_counter() - began,
        }
        with path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        done.add(key)
        print(condition, example.task_id, row["success"], flush=True)
    del model, coder
    gc.collect()
(root / "complete.json").write_text(
    json.dumps({"complete": True, "episodes": len(done), "tasks": 30})
)
