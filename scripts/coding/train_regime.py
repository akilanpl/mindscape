"""Matched-budget gradient training on actual coding state/experience records."""

import argparse
import json
import random
import resource
import time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--condition", choices=["structured", "mindscape_b", "mindscape_c"], required=True)
p.add_argument("--seed", type=int, required=True)
p.add_argument("--samples", type=int, default=100)
p.add_argument("--output", required=True)
a = p.parse_args()
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from mindscape.coding.device import select_device
from mindscape.coding.policy import _model_view
from mindscape.coding.schema import CodeTask

execution = select_device()
torch.set_num_threads(4)
torch.manual_seed(a.seed)
root = Path(a.output)
root.mkdir(parents=True, exist_ok=True)
tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
model = get_peft_model(
    AutoModelForCausalLM.from_pretrained(a.model, local_files_only=True, torch_dtype=execution.dtype),
    LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0,
        task_type="CAUSAL_LM",
    ),
)
model.to(execution.device)
execution.check_model(model)
optimizer = torch.optim.AdamW([x for x in model.parameters() if x.requires_grad], lr=2e-4)
dataset = json.loads(Path("results/coding/final_dataset_v1/dataset.json").read_text())
tasks = [CodeTask(**v) for v in dataset["train"][: a.samples]]
losses = []
tokens = 0
began = time.perf_counter()
rng = random.Random(a.seed)
replay = []
for t in tasks:
    trace = json.loads(
        (Path("results/coding/teacher_traces_v1") / (t.task_id + ".json")).read_text()
    )
    assert trace["visible_verified"]
    transitions = trace["trajectory"]["transitions"]
    frame = next(x for x in transitions if x["action"]["name"] == "edit")
    assert frame["valid"]
    state = transitions[2]["state_before"] if a.condition == "structured" else frame["state_before"]
    payload = {
        "repository": state["files"],
        "visible_tests": t.visible_tests,
        "task": t.problem_statement,
        "actual_observations": [],
    }
    payload["state"] = {
        "symbols": state["symbols"],
        "changes": state["current_changes"],
        "progress": state["progress"],
        "failures": state["failing_tests"],
    }
    payload["relations"] = state["relations"]
    payload["goal"] = state["goal"]["description"]
    target = dict(frame["action"])
    if a.condition == "structured":
        target = {k: target[k] for k in ("path", "content")}
        system = "You are a Python repair agent. Choose a concrete source edit using the supplied evidence. Return JSON only."
        payload["instruction"] = (
            "Select the file that needs repair and return one JSON object with path and content containing its complete corrected Python source. Do not include hidden tests."
        )
    else:
        system = "You are a bounded Python repair agent. Select and execute one tool action per turn. Output a single JSON action, no explanation."
        payload["state"]["previous_actions"] = state["previous_actions"]
        payload["instruction"] = (
            "Choose exactly one tool action as JSON. Fields: name, optional path/symbol/query/content/old/new/test. For edit, content is the complete corrected Python file. Use actual visible tests to verify changes; finish when satisfied. Inspection and test actions are available; private evaluation is inaccessible."
        )
        target = {k: v for k, v in target.items() if v is not None}
        payload["tools"] = list(state["context"]["tools"])
    if a.condition == "mindscape_c":
        # Only pre-action real observations enter inputs. Future state/result remain audit-only.
        payload["actual_observations"] = state["observations"]
        payload["errors"] = state["errors"]
        replay.append(
            {
                "task_id": t.task_id,
                "state": state,
                "action": frame["action"],
                "event": frame["event"],
                "result": frame["result"],
                "next_state": frame["state_after"],
                "verified_actual_experience": True,
            }
        )
    payload = _model_view(payload)
    prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(payload)}],
        tokenize=False,
        add_generation_prompt=True,
    )
    input_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(json.dumps(target) + tokenizer.eos_token, add_special_tokens=False)[
        "input_ids"
    ]
    ids = input_ids + target_ids
    if len(ids) > 2048:
        raise RuntimeError("Context exceeds fixed cap; no silent truncation")
    inputs = torch.tensor([ids], device=execution.device)
    labels = inputs.clone()
    labels[:, : len(input_ids)] = -100
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = model(input_ids=inputs, labels=labels).loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_([x for x in model.parameters() if x.requires_grad], 1)
    optimizer.step()
    losses.append(float(loss.detach()))
    tokens += len(ids)
    if len(losses) in (10, 25, 50, 100):
        folder = root / f"checkpoint_{len(losses)}"
        model.save_pretrained(folder)
        (folder / "training_budget.json").write_text(
            json.dumps(
                {
                    "samples": len(losses),
                    "condition": a.condition,
                    "seed": a.seed,
                    "tokens": tokens,
                    "training_seconds": time.perf_counter() - began,
                    "losses": losses,
                }
            )
        )
    print(a.condition, a.seed, len(losses), losses[-1], flush=True)
if replay:
    (root / "actual_replay.jsonl").write_text("".join(json.dumps(x) + "\n" for x in replay))
(root / "training.json").write_text(
    json.dumps(
        {
            "condition": a.condition,
            "seed": a.seed,
            "samples": len(losses),
            "tokens": tokens,
            "training_seconds": time.perf_counter() - began,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "trainable_parameters": sum(x.numel() for x in model.parameters() if x.requires_grad),
            "hidden_inputs": False,
            "future_state_input": False,
            "task_ids": [t.task_id for t in tasks],
        },
        indent=2,
    )
)
