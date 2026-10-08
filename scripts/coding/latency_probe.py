"""Fresh generation only, serialized after other experiments; no cache speed claims."""

import argparse
import gc
import json
import resource
import time
from pathlib import Path

from mindscape.coding.instrumentation import StageMeter
from mindscape.coding.memory import CodingMemory
from mindscape.coding.model import LocalCoder
from mindscape.coding.policy import CodingPolicy, ToolCodingPolicy
from mindscape.coding.sandbox import WasiSandbox
from mindscape.coding.schema import CodeTask

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--output", default="results/coding/latency_probe_v1")
a = p.parse_args()
root = Path(a.output)
root.mkdir(parents=True, exist_ok=True)
data = json.loads(Path("results/coding/final_dataset_v1/dataset.json").read_text())
teacher = json.loads(
    Path("results/coding/completion_collection_v1/teacher_memory.json").read_text()
)
memory = CodingMemory(episodic=teacher[:100])
sandbox = WasiSandbox("work/coding/runtime")
path = root / "rows.jsonl"
done = (
    {(r["condition"], r["task_id"]) for r in [json.loads(s) for s in path.read_text().splitlines()]}
    if path.exists()
    else set()
)
for condition in ("model_only", "structured", "mindscape_b", "mindscape_c"):
    adapter = (
        Path("results/coding/gradient_v1/seed_11/checkpoint_100")
        if condition == "model_only"
        else Path("results/coding/completion_gradient_v1") / condition / "seed_11/checkpoint_100"
    )
    coder = LocalCoder(a.model, adapter=adapter, cache_enabled=False)
    policy = (
        ToolCodingPolicy(coder, sandbox, memory)
        if condition.startswith("mindscape")
        else CodingPolicy(coder, sandbox, condition)
    )
    for value in data["test"][:10] + data["ood_test"][:4]:
        if (condition, value["task_id"]) in done:
            continue
        cpu_before = time.process_time()
        child_before = resource.getrusage(resource.RUSAGE_CHILDREN)
        with StageMeter() as meter:
            row = policy.solve(CodeTask(**value))
        row.update(
            condition=condition,
            stages=meter.summary(),
            cache_enabled=False,
            cpu_process_seconds=time.process_time() - cpu_before,
            child_cpu_seconds=(
                resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime - child_before.ru_utime
            )
            + (resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime - child_before.ru_stime),
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            gpu_used=False,
            concurrency="serialized benchmark jobs; ordinary OS activity not controlled",
        )
        with path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        done.add((condition, value["task_id"]))
        print("latency", condition, len(done), flush=True)
    del policy, coder
    gc.collect()
(root / "complete.json").write_text(
    json.dumps({"episodes": len(done), "fresh_generation": True, "cache_hits": 0}, indent=2)
)
