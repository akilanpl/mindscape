"""Bounded native-MPS study; preserve CPU rows, freeze lockbox policy, checkpoint every case."""

import argparse
import gc
import hashlib
import json
import os
import resource
import signal
import time
from pathlib import Path

import torch

from mindscape.coding.batching import ResidentCoder, evaluate_batch
from mindscape.coding.instrumentation import StageMeter
from mindscape.coding.memory import CodingMemory
from mindscape.coding.policy import CodingPolicy, ToolCodingPolicy
from mindscape.coding.sandbox import WasiSandbox
from mindscape.coding.schema import CodeTask

os.chdir(Path(__file__).resolve().parents[2])
BASE = Path("results/coding")
ROOT = BASE / "emergency_mps_v1"
ROOT.mkdir(parents=True, exist_ok=True)
START = 1791466200.0  # 2026-10-08 13:30:00 UTC; original user deadline, not reset on resume.
END = START + 7200
MODEL = "work/coding/hf/models--Qwen--Qwen2.5-Coder-1.5B-Instruct/snapshots/2e1fd397ee46e1388853d2af2c993145b0f1098a"


def read(path):
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def append(path, row):
    with path.open("a") as stream:
        stream.write(json.dumps(row) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


TEACHER = None


def teacher_records():
    global TEACHER
    if TEACHER is None:
        TEACHER = json.loads((BASE / "final_retrieval_v1/teacher_memory.json").read_text())
    return TEACHER


def memory(budget=100):
    # Exact previously used training-only prefix; never includes evaluation experience.
    records = teacher_records()
    return CodingMemory(episodic=records[:budget])


def adapter(condition, seed=11, budget=100):
    if condition == "model_only":
        return BASE / f"gradient_v1/seed_{seed}/checkpoint_{budget}"
    return BASE / f"completion_gradient_v1/{condition}/seed_{seed}/checkpoint_{budget}"


def solve(task, client, condition, options=None, seed=11, budget=100):
    sandbox = WasiSandbox("work/coding/runtime")
    policy = (ToolCodingPolicy(client, sandbox, memory(budget), **(options or {}))
              if condition.startswith("mindscape") else CodingPolicy(client, sandbox, condition))
    row = policy.solve(task, seed)
    row.update(condition=condition, split=task.metadata["split"], neural_device="mps",
               model_precision=row["attempts"][0]["latency"]["precision"], generation_cache_enabled=False,
               physical_batch_sizes=[a["latency"]["actual_batch_size"] for a in row["attempts"]])
    return row


def run_group(coder, tasks, condition, output, batch_size, cutoff, metadata=None, options=None):
    existing = read(output)
    done = {(r["condition"], r["task_id"]) for r in existing}
    pending = [t for t in tasks if (condition, t.task_id) not in done]
    coder.deadline = cutoff
    def work(task, client):
        row = solve(task, client, condition, options,
                    seed=(metadata or {}).get("seed",11), budget=(metadata or {}).get("budget",100))
        row.update(metadata or {})
        return row
    def save(row):
        append(output, row)
        print("completed", output.parent.name, condition, row["task_id"][:10], flush=True)
        write(ROOT / "progress.json", {"stage": output.parent.name, "condition": condition,
                                       "deadline_unix": END, "last_completed": row["task_id"],
                                       "updated_unix": time.time()})
    return evaluate_batch(coder, pending, work, save, batch_size, cutoff)


class CaptureRequest(Exception):
    pass


class CaptureCoder:
    def __init__(self):
        self.calls = 0
        self.timings = []
    def generate(self, *args):
        self.request = args
        raise CaptureRequest


def profile(coder, data):
    started = time.time()
    cutoff = min(started + 600, END - 1800)
    reference = [r for r in read(BASE / "completion_gradient_v1/rows.jsonl")
                 if r["condition"] == "mindscape_b" and r["seed"] == 11 and r["budget"] == 100]
    selected = [r for split in ("test", "ood_test")
                for r in [q for q in reference if q["split"] == split][:10]]
    if len(selected) != 20:
        raise RuntimeError("Twenty existing matched CPU reference episodes required")
    tasks = {v["task_id"]: CodeTask(**v) for split in ("test", "ood_test") for v in data[split]}
    coder.use_adapter(adapter("mindscape_b"))
    requests = []
    for row in selected[:16]:
        capture = CaptureCoder()
        try:
            solve(tasks[row["task_id"]], capture, "mindscape_b")
        except CaptureRequest:
            pass
        args = capture.request
        legacy_key = hashlib.sha256(json.dumps(
            [coder.revision, *args[:3], "greedy"], sort_keys=True
        ).encode()).hexdigest()
        cached = Path("work/coding/generation_cache") / (legacy_key + ".json")
        if not cached.exists() or json.loads(cached.read_text())["response"] != row["attempts"][0]["response"]:
            raise RuntimeError("Reconstructed request does not match preserved CPU reference")
        requests.append(args)
    precision_trials = []
    for precision in ("float16", "bfloat16"):
        if time.time() >= cutoff:
            break
        try:
            coder.change_precision(precision)
            coder.deadline = cutoff
            began = time.perf_counter()
            candidate = coder.generate_batch(requests[:1])
            precision_trials.append({"precision":precision,"seconds":time.perf_counter()-began,
                                     "supported":True,"response":candidate[0][0]})
        except (RuntimeError, NotImplementedError, TypeError) as exc:
            precision_trials.append({"precision":precision,"supported":False,"error":str(exc)})
    usable = [t for t in precision_trials if t["supported"]]
    if not usable:
        raise RuntimeError("No supported reduced-precision MPS path")
    chosen = min(usable, key=lambda t:t["seconds"])["precision"]
    coder.change_precision(chosen)
    coder.cpu_checkpoint_state = None
    coder.cpu_checkpoint_buffers = None
    gc.collect()
    trials = []
    # Same checkpoint and fixed prompt set; no hidden lockbox outcomes are available here.
    for size in (1, 4, 8, 16):
        if time.time() >= cutoff:
            break
        coder.deadline = cutoff
        before = time.perf_counter()
        try:
            responses = coder.generate_batch(requests[:size])
        except RuntimeError as exc:
            if "memory" not in str(exc).lower():
                raise
            torch.mps.empty_cache()
            continue
        elapsed = time.perf_counter() - before
        trials.append({"batch_size": size, "seconds": elapsed, "requests_per_second": size/elapsed,
                       "outputs": [v[0] for v in responses], "precision": coder.precision,
                       "mps_allocated_bytes": torch.mps.current_allocated_memory(),
                       "mps_driver_bytes": torch.mps.driver_allocated_memory()})
    if not trials:
        raise RuntimeError("No valid MPS batch measurement")
    selected_batch = max(trials, key=lambda r:r["requests_per_second"])["batch_size"]
    outcome_path = ROOT / "smoke_rows.jsonl"
    before = time.perf_counter()
    run_group(coder, [tasks[r["task_id"]] for r in selected], "mindscape_b", outcome_path,
              selected_batch, cutoff, {"purpose": "Matched CPU/MPS diagnostic; not lockbox"})
    elapsed = time.perf_counter() - before
    observed = {r["task_id"]: r for r in read(outcome_path)}
    agreement = [{"task_id": r["task_id"], "cpu_success": r["success"],
                  "mps_success": observed[r["task_id"]]["success"],
                  "final_outcome_agrees": r["success"] == observed[r["task_id"]]["success"],
                  "mps_transitions_valid": all(t["valid"] for t in observed[r["task_id"]]["trajectory"]["transitions"])}
                 for r in selected if r["task_id"] in observed]
    if len(agreement) < 10:
        raise RuntimeError("Fewer than ten complete MPS correctness episodes within profiling cap")
    cpu_seconds = sum(r["wall_seconds"] for r in selected if r["task_id"] in observed)/len(agreement)
    result = {"cpu_seconds_per_episode": cpu_seconds,
              "cpu_reference_scope": "Preserved fresh CPU float32 episodes; not rerun",
              "mps_throughput_seconds_per_episode": elapsed/len(agreement),
              "mps_mean_episode_wall_seconds": sum(r["wall_seconds"] for r in observed.values())/len(observed),
              "throughput_speedup": cpu_seconds/(elapsed/len(agreement)),
              "mps_episodes_per_hour": len(agreement)*3600/elapsed,
              "selected_batch_size": selected_batch, "precision": coder.precision,
              "precision_trials":precision_trials,
              "load_seconds": coder.load_seconds, "comparisons": agreement, "batch_trials": trials,
              "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "mps_allocated_bytes": torch.mps.current_allocated_memory(),
              "mps_driver_bytes": torch.mps.driver_allocated_memory(),
              "profiling_seconds": time.time()-started,
              "interpretation": "Historical CPU vs fresh batched MPS throughput; per-case latency reported separately. Arithmetic precision differs; learning-curve hardware strata remain disclosed."}
    write(ROOT / "profile.json", result)
    return result


def main():
    if time.time() >= END - 1200:
        raise RuntimeError("Insufficient time for a new neural study before original two-hour deadline")
    data = json.loads((BASE / "final_dataset_v1/dataset.json").read_text())
    lockbox = [CodeTask(**v) for v in json.loads((BASE / "completion_lockbox_v1/tasks.json").read_text())]
    write(ROOT / "native_device.json", {"mps_built": torch.backends.mps.is_built(),
          "mps_available": torch.backends.mps.is_available(), "fallback": os.getenv("PYTORCH_ENABLE_MPS_FALLBACK"),
          "start_unix": START, "deadline_unix": END, "native_pid": os.getpid()})
    teacher_records()
    coder = ResidentCoder(MODEL, precision="float16")
    saved_profile = ROOT / "profile.json"
    if saved_profile.exists():
        # Resume the already measured configuration, never repeat completed profiling.
        metrics = json.loads(saved_profile.read_text())
        coder.change_precision(metrics["precision"])
        coder.cpu_checkpoint_state = None
        coder.cpu_checkpoint_buffers = None
        gc.collect()
    else:
        metrics = profile(coder, data)
    # One phase-boundary release: discard precision-trial allocations, not model weights.
    before_release = torch.mps.driver_allocated_memory()
    torch.mps.empty_cache()
    write(ROOT / "phase_memory_release.json", {"before_driver_bytes":before_release,
          "after_driver_bytes":torch.mps.driver_allocated_memory(),
          "live_allocated_bytes":torch.mps.current_allocated_memory(),
          "reason":"Single profiling/evaluation boundary; no per-operation cache clearing",
          "configuration_reused":saved_profile.exists(), "time_unix":time.time()})
    batch_size = metrics["selected_batch_size"]
    protocol = {"backbone": coder.base_revision, "device": "mps", "precision": coder.precision,
                "batch_size": batch_size, "max_steps": 8, "max_edits": 3, "max_new_tokens": 256,
                "seed": 11, "training_budget": 100, "hidden_tests": "terminal external only",
                "lockbox_sha256": hashlib.sha256((BASE / "completion_lockbox_v1/tasks.json").read_bytes()).hexdigest(),
                "checkpoints": {c: hashlib.sha256((adapter(c)/"adapter_model.safetensors").read_bytes()).hexdigest()
                                for c in ("model_only","structured","mindscape_b","mindscape_c")},
                "frozen_before_lockbox": True, "deadline_unix": END,
                "source_hashes":{str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in Path("src/mindscape/coding").glob("*.py")}}
    old = ROOT / "locked_protocol.json"
    if old.exists() and json.loads(old.read_text()) != protocol:
        raise RuntimeError("Locked configuration changed; refuse evaluation")
    write(old, protocol)
    locked_root = BASE / "completion_lockbox_eval_v1"
    locked_root.mkdir(parents=True, exist_ok=True)
    for condition in ("mindscape_c","mindscape_b","model_only","structured"):
        coder.use_adapter(adapter(condition))
        run_group(coder, lockbox, condition, locked_root / "rows.jsonl", batch_size, END - (900 if condition == "mindscape_c" else 1800),
                  {"seed":11,"budget":100,"one_shot_configuration":True,
                   "adapter":str(adapter(condition)),"protocol":"emergency_mps_v1/locked_protocol.json"})
    locked = read(locked_root / "rows.jsonl")
    unique = {(r["condition"],r["task_id"]) for r in locked}
    if len(unique) != len(locked):
        raise RuntimeError("Duplicate locked keys; refuse completion")
    required_ids = {t.task_id for t in lockbox}
    primary_complete = all({r["task_id"] for r in locked if r["condition"] == c} == required_ids
                           for c in ("model_only","mindscape_c"))
    write(locked_root / "scope.json", {"completed":len(locked),"planned":400,"not_run":400-len(locked),
          "primary_100_task_comparison_complete":primary_complete,
          "four_condition_comparison_complete":len(locked)==400,
          "priority":"Flagship C and matched A on all 100 fixed tasks; supplementary B control may remain partial"})
    if not primary_complete:
        raise RuntimeError("Mandatory 100-task flagship/matched-A comparison incomplete; retain actual partial rows")
    write(locked_root / "primary_complete.json", {"tasks":100,"conditions":["model_only","mindscape_c"],
          "actual_saved_episodes":len(locked),"four_condition_complete":len(locked)==400})
    if len(locked)==400:
        write(locked_root / "complete.json", {"complete":True,"tasks":100,"conditions":4,"episodes":400,
                                             "configuration_frozen_before_outcomes":True})
    # Remaining curve: exact unfinished keys only. Leave thirty minutes for evidence/latency/reporting.
    curve_path = BASE / "completion_gradient_v1/rows.jsonl"
    done = {tuple(r["key"]) for r in read(curve_path)}
    for condition in ("mindscape_c","mindscape_b"):
        for seed in (11,23,37):
            for budget in (10,25,50,100):
                if time.time() >= END-1800:
                    break
                pending = [CodeTask(**v) for split in ("test","ood_test") for v in data[split]
                           if (condition,seed,budget,split,v["task_id"]) not in done]
                if not pending:
                    continue
                coder.use_adapter(adapter(condition,seed,budget))
                partial = ROOT / f"curve_{condition}_{seed}_{budget}.jsonl"
                run_group(coder, pending, condition, partial,batch_size,END-1800,
                          {"seed":seed,"budget":budget,"adapter":str(adapter(condition,seed,budget)),
                           "gradient_examples":budget,"retrieval_examples":budget})
                for row in read(partial):
                    key = (condition,seed,budget,row["split"],row["task_id"])
                    if key not in done:
                        row["key"] = list(key)
                        append(curve_path,row);done.add(key)
    write(ROOT/"curve_scope.json", {"completed":len(done),"planned":1920,"not_run":1920-len(done),
                                   "status":"complete" if len(done)==1920 else "partial",
                                   "cpu_rows_preserved":1122,"hardware_precision_stratification_required":True})
    if len(done)==1920:
        write(BASE/"completion_gradient_v1/complete.json", {"complete":True,"episodes":1920,
                                                         "mixed_hardware_precision":True})
    # One representative cache-disabled latency study, using direct sequential resident generation.
    latency_root = BASE/"latency_probe_v1";latency_root.mkdir(parents=True,exist_ok=True)
    representatives = [CodeTask(**v) for v in [data[split][index] for index in range(2) for split in ("test","ood_test")]]
    for condition in ("mindscape_c","model_only","structured","mindscape_b"):
        if time.time() >= END-600:
            break
        coder.use_adapter(adapter(condition));coder.deadline=END-600
        for task in representatives:
            if any(r["condition"]==condition and r["task_id"]==task.task_id for r in read(latency_root/"rows.jsonl")):
                continue
            if time.time() >= END-600:
                break
            def cutoff(signum, frame):
                raise TimeoutError("Latency deadline reached; unfinished episode not counted")
            signal.signal(signal.SIGALRM,cutoff)
            signal.setitimer(signal.ITIMER_REAL,max(.001,END-600-time.time()))
            try:
                with StageMeter() as meter:
                    row = solve(task,coder,condition)
            except TimeoutError:
                print("Latency cutoff reached; preserving completed episodes",flush=True)
                break
            finally:
                signal.setitimer(signal.ITIMER_REAL,0)
            row.update(stages=meter.summary(),cache_enabled=False,
                       peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                       gpu_used=True,mps_allocated_bytes=torch.mps.current_allocated_memory(),
                       mps_driver_bytes=torch.mps.driver_allocated_memory())
            append(latency_root/"rows.jsonl",row)
    latency = read(latency_root/"rows.jsonl")
    write(latency_root/"scope.json", {"completed":len(latency),"planned":16,"not_run":16-len(latency),
                                       "status":"complete" if len(latency)==16 else "partial"})
    if len(latency)==16:
        write(latency_root/"complete.json", {"episodes":16,"fresh_generation":True,"cache_hits":0,
                                           "tasks_per_condition":4,"percentiles":"P50/P95; small descriptive sample"})
    write(ROOT/"neural_stop.json", {"stopped_unix":time.time(),"deadline_unix":END,
                                    "physical_neural_batches":coder.batch_records})
    del coder;gc.collect()
    print("NEURAL STAGES STOPPED; preserved all completed results",flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        main()
    except Exception as exc:
        write(ROOT/"failure.json", {"type":type(exc).__name__,"message":str(exc),"time_unix":time.time()})
        raise
