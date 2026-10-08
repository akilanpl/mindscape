"""Count the exact tokenized sequences used by preserved patch checkpoints."""

import json
from pathlib import Path

from transformers import AutoTokenizer

from mindscape.coding.generator import generate

path = Path(
    "work/coding/hf/models--Qwen--Qwen2.5-Coder-1.5B-Instruct/snapshots/2e1fd397ee46e1388853d2af2c993145b0f1098a"
)
tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
tasks = generate(counts={"train": 100, "validation": 0, "test": 0, "ood_test": 0})["train"]
total = 0
target_total = 0
records = []
for i, t in enumerate(tasks, 1):
    messages = [
        {
            "role": "system",
            "content": "You repair Python. Return exactly a JSON object containing path and complete corrected source content.",
        },
        {"role": "user", "content": json.dumps(t.visible())},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    path = next(iter(t.ground_truth_patch))
    target = json.dumps({"path": path, "content": t.ground_truth_patch[path]}) + tokenizer.eos_token
    total += len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    count = len(tokenizer(target, add_special_tokens=False)["input_ids"])
    total += count
    target_total += count
    if i in (10, 25, 50, 100):
        records.append(
            {
                "budget": i,
                "training_input_and_target_tokens": total,
                "loss_target_tokens": target_total,
                "definition": "Exact deterministic sequence recount from preserved training code; not an inference or runtime estimate",
            }
        )
root = Path("results/coding/completion_audits_v1")
root.mkdir(parents=True, exist_ok=True)
(root / "patch_training_tokens.json").write_text(json.dumps(records, indent=2))
print(records)
