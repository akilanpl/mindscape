"""Training-only demonstrations, matched information ledgers and resumable Adam state."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .environment import MAX_DIGITS, Claim, LocalState, demonstration
from .learner import (
    GROUPS,
    TransitionNet,
    claim_features,
    local_features,
    split_logits,
    task_features,
)


def sign_features(visible):
    x = task_features(visible)
    x[48:288] = 0  # Signs and magnitude relationships suffice; no arithmetic values.
    x[42] = 1
    return x


def answer_labels(value):
    text = str(abs(value))[::-1]
    if len(text) > MAX_DIGITS:
        raise ValueError("Answer bound exceeded")
    return [int(d) for d in text] + [0] * (MAX_DIGITS - len(text)) + [int(value < 0)]


def build_training(problems):
    if not problems or any(p.split != "train" for p in problems):
        raise ValueError("Training partition required")
    local_x, local_y, sign_x, sign_y, answer_x, answer_y = [], [], [], [], [], []
    memory, demonstrations = {}, []
    for problem in problems:
        trace = demonstration(problem)
        demonstrations.append(trace)
        answer_x.append(task_features(problem.visible()))
        answer_y.append(answer_labels(int(trace["expected"])))
        current = problem.operands[0]
        for event in trace["events"]:
            if "state" in event:
                state, claim = LocalState(**event["state"]), Claim(**event["claim"])
                local_x.append(local_features(state))
                local_y.append(claim.labels())
                memory[state.key] = claim
            else:
                stage = event["stage_result"]
                visible = {
                    "operands": (current, problem.operands[stage + 1]),
                    "operations": (problem.operations[stage],),
                }
                sign_x.append(sign_features(visible))
                sign_y.append(event["sign"])
                current = int(event["value"])
    arrays = {
        "local_x": np.asarray(local_x),
        "local_y": np.asarray(local_y, dtype=np.int64),
        "sign_x": np.asarray(sign_x),
        "sign_y": np.asarray(sign_y, dtype=np.int64),
        "answer_x": np.asarray(answer_x),
        "answer_y": np.asarray(answer_y, dtype=np.int64),
    }
    ledger = {
        "distinct_examples": len(problems),
        "task_ids": [p.identity for p in problems],
        "teacher_demonstrations": len(problems),
        "intermediate_rows": len(local_y),
        "intermediate_scalar_labels": len(local_y) * 6,
        "sign_labels": len(sign_y),
        "final_scalar_labels": len(answer_y) * (MAX_DIGITS + 1),
        "consistency_positive_labels": len(local_y),
        "consistency_negative_labels": len(local_y),
        "environment_feedback_queries": 0,
        "unique_demonstrated_concept_instances": len(memory),
    }
    return arrays, memory, ledger, demonstrations


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def train(
    arrays,
    ledger,
    output,
    seed,
    updates=600,
    regime="procedural",
    encoder=None,
    source_hash="",
    manifest_hash="",
    stop_after=None,
):
    if regime not in ("procedural", "answer_only"):
        raise ValueError("Unknown regime")
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    identity = {
        "seed": seed,
        "updates": updates,
        "regime": regime,
        "ledger": ledger,
        "source_hash": source_hash,
        "manifest_hash": manifest_hash,
        "pretrained_hashes": encoder.hashes if encoder else None,
    }
    receipt = directory / "identity.json"
    if receipt.exists() and json.loads(receipt.read_text()) != identity:
        raise ValueError("Training resume identity mismatch")
    if not receipt.exists():
        receipt.write_text(json.dumps(identity, sort_keys=True))
    model = TransitionNet(seed, encoder.size if encoder else 0)
    opt = torch.optim.Adam(model.parameters(), lr=0.003)
    generator = torch.Generator().manual_seed(seed + 113)
    data = {k: torch.from_numpy(v) for k, v in arrays.items()}
    pretrained = (
        {k: encoder.encode(arrays[k]) for k in ("local_x", "sign_x", "answer_x")} if encoder else {}
    )
    path = directory / "checkpoint.pt"
    start, history, elapsed = 0, [], 0.0
    if path.exists():
        if (directory / "training.json").exists() and file_hash(path) != json.loads(
            (directory / "training.json").read_text()
        )["checkpoint_sha256"]:
            raise ValueError("Training checkpoint checksum mismatch")
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["identity"] != identity:
            raise ValueError("Checkpoint identity mismatch")
        model.load_state_dict(saved["model"])
        opt.load_state_dict(saved["optimizer"])
        generator.set_state(saved["rng"])
        start = saved["step"]
        history = saved["history"]
        elapsed = saved["elapsed"]
        if start == updates:
            return model, json.loads((directory / "training.json").read_text())
    began = time.perf_counter()
    final_step = min(updates, stop_after) if stop_after is not None else updates
    for step in range(start, final_step):
        ai = torch.randint(len(data["answer_x"]), (64,), generator=generator)
        alogits = model(
            data["answer_x"][ai], pretrained.get("answer_x", None)[ai] if encoder else None
        )[1]
        loss = sum(
            F.cross_entropy(h, data["answer_y"][ai, j])
            for j, h in enumerate(alogits.split([10] * MAX_DIGITS + [2], dim=-1))
        ) / (MAX_DIGITS + 1)
        if regime == "procedural":
            li = torch.randint(len(data["local_x"]), (64,), generator=generator)
            lx, ly = data["local_x"][li], data["local_y"][li]
            lp = pretrained["local_x"][li] if encoder else None
            logits = model(lx, lp)[0]
            loss = (
                loss
                + sum(F.cross_entropy(h, ly[:, j]) for j, h in enumerate(split_logits(logits))) / 6
            )
            si = torch.randint(len(data["sign_x"]), (64,), generator=generator)
            signs = model(data["sign_x"][si], pretrained["sign_x"][si] if encoder else None)[2]
            loss = loss + F.cross_entropy(signs, data["sign_y"][si])
            positives = torch.from_numpy(
                np.stack(
                    [
                        claim_features(Claim(int(y[0]), int(y[1]) - 20, *(int(v) for v in y[2:])))
                        for y in ly
                    ]
                )
            )
            negatives = positives.clone()
            # One uniformly sampled claim field is changed without consulting held-out targets.
            columns = torch.randint(6, (64,), generator=generator)
            for row, column in enumerate(columns.tolist()):
                size = GROUPS[column]
                original = int(ly[row, column])
                value = (original + int(torch.randint(1, size, (1,), generator=generator))) % size
                scale = (2, 120, 9, 9, 9, 9)[column]
                negatives[row, column] = value / scale
            joined = torch.cat((positives, negatives))
            cx = lx.repeat(2, 1)
            cp = lp.repeat(2, 1) if encoder else None
            consistency = model.consistency(cx, joined, cp)
            loss = loss + F.cross_entropy(
                consistency,
                torch.cat((torch.ones(64, dtype=torch.long), torch.zeros(64, dtype=torch.long))),
            )
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 100 == 0 or step + 1 == final_step:
            history.append({"update": step + 1, "loss": float(loss.detach())})
    elapsed += time.perf_counter() - began
    saved = {
        "identity": identity,
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "rng": generator.get_state(),
        "step": final_step,
        "history": history,
        "elapsed": elapsed,
    }
    tmp = directory / "checkpoint.tmp"
    torch.save(saved, tmp)
    tmp.replace(path)
    result = {
        "checkpoint_sha256": file_hash(path),
        "updates_completed": final_step,
        "training_seconds": elapsed,
        "loss_history": history,
        "parameters": model.parameter_count,
        "backbone_parameters": encoder.parameter_count if encoder else model.parameter_count,
        "adapter_parameters": sum(
            p.numel() for n, p in model.named_parameters() if "adapter_" in n
        ),
        "ledger": ledger,
        "active_intermediate_labels": ledger["intermediate_scalar_labels"]
        if regime == "procedural"
        else 0,
        "sampled_task_rows": final_step * 64,
        "sampled_local_rows": final_step * 64 if regime == "procedural" else 0,
        "sampled_consistency_rows": final_step * 128 if regime == "procedural" else 0,
        "training_tokens": 0 if encoder is None else encoder.tokens,
        "training_tokens_scope": "cumulative shared encoder cache work, not per-model matched cost",
        "complete": final_step == updates,
    }
    (directory / "training.json").write_text(json.dumps(result, indent=2))
    return model, result
