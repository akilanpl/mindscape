"""Actual offline coding LM generation with measured token/latency accounting."""

import hashlib
import json
import re
import time
from pathlib import Path


class LocalCoder:
    def __init__(self, path, adapter=None, cache_enabled=True):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cache_enabled = cache_enabled
        self.torch = torch
        torch.set_num_threads(4)
        self.path = str(Path(path).resolve())
        self.revision = Path(path).name
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            path, local_files_only=True, torch_dtype=torch.float32
        ).eval()
        if adapter is not None:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(
                self.model, adapter, local_files_only=True
            ).eval()
            self.revision += (
                ":adapter:"
                + hashlib.sha256(
                    (Path(adapter) / "adapter_model.safetensors").read_bytes()
                ).hexdigest()
            )
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.timings = []

    def generate(self, system, prompt, max_tokens=256, seed=0):
        key = hashlib.sha256(
            json.dumps(
                [self.revision, system, prompt, max_tokens, "greedy"], sort_keys=True
            ).encode()
        ).hexdigest()
        cache = Path("work/coding/generation_cache") / (key + ".json")
        if self.cache_enabled and cache.exists():
            saved = json.loads(cache.read_text())
            self.timings.append(
                {
                    "ttft": 0,
                    "generation": 0,
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "cache_hit": True,
                    "original": saved["timing"],
                }
            )
            return saved["response"]
        from transformers import StoppingCriteria, StoppingCriteriaList

        torch = self.torch
        torch.manual_seed(seed)
        text = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt")
        start = time.perf_counter()
        first = [None]

        class First(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                if first[0] is None:
                    first[0] = time.perf_counter() - start
                return False

        with torch.inference_mode():
            result = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=StoppingCriteriaList([First()]),
            )
        generated = result[0, inputs["input_ids"].shape[1] :]
        duration = time.perf_counter() - start
        self.calls += 1
        self.input_tokens += inputs["input_ids"].numel()
        self.output_tokens += len(generated)
        self.timings.append(
            {
                "ttft": first[0],
                "generation": duration,
                "tokens_in": inputs["input_ids"].numel(),
                "tokens_out": len(generated),
            }
        )
        response = self.tokenizer.decode(generated, skip_special_tokens=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        if self.cache_enabled:
            cache.write_text(
                json.dumps(
                    {
                        "response": response,
                        "timing": self.timings[-1],
                        "model_revision": self.revision,
                    }
                )
            )
        return response


def source_from_response(text):
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        return blocks[0].strip() + "\n"
    if text.lstrip().startswith(("def ", "from ", "import ", "class ")):
        return text.strip() + "\n"
    raise ValueError("No source code response")


def action_from_response(text):
    from mindscape.coding.schema import CodeAction

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON action")
    value = json.loads(match.group())
    if isinstance(value, dict) and set(value) == {"repository"}:
        mapping = value["repository"]
        if isinstance(mapping, dict) and len(mapping) == 1:
            path, content = next(iter(mapping.items()))
            value = {"name": "edit", "path": path, "content": content}
    if isinstance(value, dict) and set(value) == {"path", "content"}:
        value = {"name": "edit", **value}
    allowed = {"name", "path", "symbol", "query", "content", "old", "new", "test"}
    if not isinstance(value, dict) or set(value) - allowed:
        raise ValueError("Invalid action fields")
    return CodeAction(**value)


def edit_from_response(text, repository, entry=None):
    """Normalize common source-edit envelopes without changing model-proposed code."""
    from mindscape.coding.schema import CodeAction

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        value = json.loads(match.group())
        if isinstance(value, dict) and set(value) == {"repository"}:
            mapping = value["repository"]
            if not isinstance(mapping, dict) or len(mapping) != 1:
                raise ValueError("Exactly one repository edit required")
            path, content = next(iter(mapping.items()))
            value = {"path": path, "content": content}
        elif isinstance(value, dict) and len(value) == 1 and next(iter(value)) in repository:
            path, content = next(iter(value.items()))
            value = {"path": path, "content": content}
        if isinstance(value, dict) and value.get("name") == "edit":
            value = {k: v for k, v in value.items() if k != "name"}
        if not isinstance(value, dict) or set(value) != {"path", "content"}:
            raise ValueError("Unrecognized source edit envelope")
        if value["path"] not in repository or not isinstance(value["content"], str):
            raise ValueError("Unsafe source edit")
        return CodeAction("edit", **value)
    source = source_from_response(text)
    path = entry.rsplit(".", 1)[0].replace(".", "/") + ".py" if entry else None
    if path not in repository:
        raise ValueError("Source edit target is ambiguous")
    return CodeAction("edit", path=path, content=source)
