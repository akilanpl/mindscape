"""One resident neural model, independent episode requests, CPU tool execution."""

import hashlib
import queue
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from mindscape.coding.model import LocalCoder


class ResidentCoder(LocalCoder):
    def __init__(self, path, precision="float16"):
        super().__init__(path, cache_enabled=False, device="mps", precision="float32", retain_cpu_state=True)
        self.change_precision(precision)
        self.base_revision = self.revision
        self.loaded_adapters = {}
        self.batch_records = []
        self.deadline = None
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def change_precision(self, precision):
        from mindscape.coding.device import select_device

        self.execution = select_device("mps", precision, require_mps=True)
        self.precision = precision
        self.model.to(device=self.device, dtype=self.execution.dtype)
        target = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        # Restore every base value directly from its original CPU checkpoint buffer,
        # avoiding successive rounding when comparing reduced precisions.
        if self.cpu_checkpoint_state is not None:
            for key, tensor in target.state_dict().items():
                original = key.replace(".base_layer.", ".")
                if original in self.cpu_checkpoint_state:
                    tensor.copy_(self.cpu_checkpoint_state[original].to(self.device, dtype=tensor.dtype))
        if self.cpu_checkpoint_buffers is not None:
            for name, value in self.cpu_checkpoint_buffers.items():
                parent, leaf = name.rsplit(".", 1) if "." in name else ("", name)
                module = target.get_submodule(parent) if parent else target
                module._buffers[leaf] = value.to(self.device)
        if hasattr(self, "active_adapter_path"):
            from peft.utils.save_and_load import set_peft_model_state_dict
            from safetensors.torch import load_file

            set_peft_model_state_dict(self.model,
                load_file(str(self.active_adapter_path / "adapter_model.safetensors")),
                adapter_name=self.active_adapter_name)
        self.execution.synchronize()
        self.execution.check_model(self.model)

    def use_adapter(self, path):
        from peft import PeftModel

        path = Path(path)
        digest = hashlib.sha256((path / "adapter_model.safetensors").read_bytes()).hexdigest()
        name = "a_" + digest[:20]
        if name not in self.loaded_adapters:
            if not isinstance(self.model, PeftModel):
                self.model = PeftModel.from_pretrained(
                    self.model, path, adapter_name=name, local_files_only=True
                )
            else:
                self.model.load_adapter(path, adapter_name=name)
            self.loaded_adapters[name] = str(path)
        self.active_adapter_path = path
        self.active_adapter_name = name
        self.model.set_adapter(name)
        self.model.to(device=self.device, dtype=self.execution.dtype).eval()
        self.execution.check_model(self.model)
        self.execution.synchronize()
        self.revision = self.base_revision + ":adapter:" + digest

    def generate_batch(self, requests):
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        torch.manual_seed(requests[0][3] if requests and len(requests[0]) > 3 else 0)
        if not requests:
            return []
        limits = {r[2] for r in requests}
        if len(limits) != 1:
            raise ValueError("A batch must share the generation cap")
        texts = [self.tokenizer.apply_chat_template(
            [{"role": "system", "content": r[0]}, {"role": "user", "content": r[1]}],
            tokenize=False, add_generation_prompt=True,
        ) for r in requests]
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, pad_to_multiple_of=64,
            truncation=False,
        ).to(self.device)
        if inputs["input_ids"].shape[1] + next(iter(limits)) > 32768:
            raise RuntimeError("Context cap exceeded; truncation forbidden")
        self.execution.check_model(self.model)
        self.execution.synchronize()
        start = time.perf_counter()
        first = [None]
        execution = self.execution
        deadline = self.deadline

        class First(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                if deadline is not None and time.time() >= deadline:
                    raise TimeoutError("Neural study deadline reached")
                if first[0] is None:
                    execution.synchronize()
                    first[0] = time.perf_counter() - start
                return False

        with torch.inference_mode():
            result = self.model.generate(
                **inputs, max_new_tokens=next(iter(limits)), do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=StoppingCriteriaList([First()]),
            )
        self.execution.synchronize()
        seconds = time.perf_counter() - start
        # Decoding/counting is a CPU utility after neural computation finishes.
        output = result[:, inputs["input_ids"].shape[1]:].cpu().tolist()
        lengths = inputs["attention_mask"].sum(dim=1).cpu().tolist()
        values = []
        for tokens, length in zip(output, lengths, strict=True):
            if self.tokenizer.eos_token_id in tokens:
                tokens = tokens[:tokens.index(self.tokenizer.eos_token_id) + 1]
            timing = {"ttft": first[0], "generation": seconds, "tokens_in": length,
                      "tokens_out": len(tokens), "device": "mps", "precision": self.precision,
                      "actual_batch_size": len(requests), "cache_hit": False}
            values.append((self.tokenizer.decode(tokens, skip_special_tokens=True), timing))
        self.batch_records.append({"batch_size": len(requests), "seconds": seconds,
                                   "padded_input_tokens": inputs["input_ids"].shape[1],
                                   "revision": self.revision, "device": "mps"})
        return values


@dataclass
class Request:
    args: tuple
    future: Future
    submitted_at: float


class EpisodeCoder:
    """Per-episode accounting; only the coordinator owns/calls the neural model."""
    def __init__(self, requests, revision):
        self.requests = requests
        self.revision = revision
        self.calls = 0
        self.timings = []

    def generate(self, *args):
        future = Future()
        self.requests.put(Request(args, future, time.perf_counter()))
        response, timing = future.result()
        self.calls += 1
        self.timings.append(timing)
        return response


def evaluate_batch(coder, jobs, solve, save, batch_size=8, deadline=None):
    """Each worker owns one environment/memory; decisions within it remain sequential."""
    requests = queue.Queue()
    stop = threading.Event()
    completed = 0
    failure = None
    with ThreadPoolExecutor(max_workers=batch_size) as workers:
        futures = []
        for job in jobs:
            def run(value=job):
                if stop.is_set():
                    return None
                client = EpisodeCoder(requests, coder.revision)
                return solve(value, client)
            futures.append(workers.submit(run))
        delivered = set()
        while len(delivered) < len(futures):
            if deadline is not None and time.time() >= deadline:
                stop.set()
            ready = []
            try:
                ready.append(requests.get(timeout=0.02))
                until = time.monotonic() + 0.02
                while len(ready) < batch_size and time.monotonic() < until:
                    try:
                        ready.append(requests.get(timeout=max(0.001, until-time.monotonic())))
                    except queue.Empty:
                        break
            except queue.Empty:
                pass
            if ready:
                if stop.is_set():
                    for request in ready:
                        request.future.set_exception(TimeoutError("Study deadline reached"))
                else:
                    try:
                        values = coder.generate_batch([r.args for r in ready])
                        for request, (response, timing) in zip(ready, values, strict=True):
                            timing["request_wall_seconds"] = time.perf_counter()-request.submitted_at
                            request.future.set_result((response, timing))
                    except Exception as exc:  # noqa: BLE001 - propagate to every waiting client
                        failure = exc
                        stop.set()
                        for request in ready:
                            request.future.set_exception(exc)
            for index, future in enumerate(futures):
                if index in delivered or not future.done():
                    continue
                delivered.add(index)
                try:
                    row = future.result()
                except TimeoutError:
                    continue
                except Exception as exc:  # noqa: BLE001 - drain waiting workers before propagation
                    failure = failure or exc
                    stop.set()
                    continue
                if row is not None:
                    save(row)
                    completed += 1
            if stop.is_set() and not ready:
                # Active clients will submit their next request and receive the deadline error.
                continue
    if failure is not None and not isinstance(failure, TimeoutError):
        raise failure
    return completed
