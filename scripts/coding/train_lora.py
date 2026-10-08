"""Real supervised gradient adaptation, separate from retrieval sample budgets."""

import argparse
import json
import resource
import time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--samples", type=int, default=10)
p.add_argument("--seed", type=int, default=11)
p.add_argument("--checkpoints", default="")
p.add_argument("--output", default="results/coding/lora_probe_v1")
a = p.parse_args()
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from mindscape.coding.device import select_device
from mindscape.coding.generator import generate

execution = select_device()
torch.set_num_threads(4)
torch.manual_seed(a.seed)
root = Path(a.output)
root.mkdir(parents=True, exist_ok=True)
tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
base = AutoModelForCausalLM.from_pretrained(
    a.model, local_files_only=True, torch_dtype=execution.dtype
)
model = get_peft_model(
    base,
    LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0,
        task_type="CAUSAL_LM",
    ),
)
model.train()
model.to(execution.device)
execution.check_model(model)
optimizer = torch.optim.AdamW((x for x in model.parameters() if x.requires_grad), lr=2e-4)
tasks = generate(counts={"train": a.samples, "validation": 0, "test": 0, "ood_test": 0})["train"]
losses = []
start = time.perf_counter()
for t in tasks:
    path = next(iter(t.ground_truth_patch))
    messages = [
        {
            "role": "system",
            "content": "You repair Python. Return exactly a JSON object containing path and complete corrected source content.",
        },
        {"role": "user", "content": json.dumps(t.visible())},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    target = json.dumps({"path": path, "content": t.ground_truth_patch[path]}) + tokenizer.eos_token
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
    ids = prompt_ids + target_ids
    if len(ids) > 768:
        raise RuntimeError("Training truncation forbidden")
    inputs = torch.tensor([ids], device=execution.device)
    labels = inputs.clone()
    labels[:, : len(prompt_ids)] = -100
    optimizer.zero_grad(set_to_none=True)
    loss = model(input_ids=inputs, labels=labels).loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_([x for x in model.parameters() if x.requires_grad], 1)
    optimizer.step()
    losses.append(float(loss.detach()))
    if str(len(losses)) in a.checkpoints.split(","):
        model.save_pretrained(root / ("checkpoint_" + str(len(losses))))
        (root / ("checkpoint_" + str(len(losses))) / "training_budget.json").write_text(
            json.dumps(
                {
                    "samples": len(losses),
                    "seed": a.seed,
                    "epochs": 1,
                    "training_seconds": time.perf_counter() - start,
                }
            )
        )
    print(len(losses), losses[-1], round(time.perf_counter() - start, 1), flush=True)
model.save_pretrained(root / "adapter")
tokenizer.save_pretrained(root / "adapter")
record = {
    "backbone_revision": Path(a.model).name,
    "samples": a.samples,
    "seed": a.seed,
    "epochs": 1,
    "losses": losses,
    "training_seconds": time.perf_counter() - start,
    "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    "total_parameters": sum(x.numel() for x in model.parameters()),
    "trainable_parameters": sum(x.numel() for x in model.parameters() if x.requires_grad),
    "learning_rate": 2e-4,
    "rank": 8,
    "targets": ["q_proj", "v_proj"],
    "task_ids": [t.task_id for t in tasks],
    "hidden_tests_in_training": False,
}
(root / "training.json").write_text(json.dumps(record, indent=2))
