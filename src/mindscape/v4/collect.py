"""Frozen specialist alternatives; independent scoring only after features are saved."""

import json
import time
from dataclasses import asdict

import numpy as np
import torch

from mindscape.v3.integration.runtime import Request
from mindscape.v3.schema import digest

from .encoding import encode


def proposals(task, assets):
    domain, public = task["domain"], task["public"]
    if domain == "mathematics":
        from mindscape.v3.math_learning.environment import LocalState
        from mindscape.v3.math_learning.learner import (
            claim_features,
            decode_claim,
            local_features,
            split_logits,
        )

        adapter = assets.math(Request(domain, {"operands": [1, 2], "operations": ["add"]}))
        state = LocalState(**public)
        x = torch.as_tensor(local_features(state)[None], device="mps")
        with torch.inference_mode():
            heads = [h[0] for h in split_logits(adapter.model(x)[0])]
            labels = [int(h.argmax().item()) for h in heads]
            claims = []
            for digit in heads[4].topk(2).indices.tolist():
                for carry in heads[5].topk(2).indices.tolist():
                    row = labels.copy()
                    row[4], row[5] = digit, carry
                    claims.append(decode_claim(row))
            cx = torch.as_tensor(np.stack([claim_features(c) for c in claims]), device="mps")
            consistency = adapter.model.consistency(x.repeat(4, 1), cx).softmax(-1)[:, 1]
            logp = [h.log_softmax(-1) for h in heads]
            scores = [
                sum(float(h[v].item()) for h, v in zip(logp, c.labels()))
                + 2 * float(consistency[i].clamp(min=1e-8).log().item())
                for i, c in enumerate(claims)
            ]
        return public, [{"claim": asdict(c)} for c in claims], scores, adapter.provenance
    if domain == "coding":
        request = Request(domain, public)
        adapter = assets.coding(request)
        observed = adapter.observation()
        proposals, scores = adapter.propose(observed)
        # Preserve the complete actual Qwen response and generation timing outside encoder.
        provenance = {**adapter.provenance, "generation": proposals}
        return observed, [{"repository": p["repository"]} for p in proposals], scores, provenance
    from mindscape.v3.physics_world.planning import CANDIDATES
    from mindscape.v3_1.physics.evaluation import rollout

    adapter = assets.physics(Request(domain, public))
    observed = adapter.observation()
    horizon = public["horizon"]
    actions = np.zeros((9, horizon, 3, 4), np.float32)
    actions[:, 0, 0, 2:] = CANDIDATES
    contexts = np.tile(adapter.row["contexts"][3], (9, horizon, 1))
    contexts[:, :, 3] += np.arange(horizon) / 30
    predicted = rollout(
        adapter.model,
        {
            "history": np.repeat(adapter.row["observations"][None, :4], 9, 0),
            "masks": np.repeat(adapter.row["masks"][None, :4], 9, 0),
            "actions": actions,
            "contexts": contexts,
        },
    )
    scores = ((predicted["states"][:, -1, 0, :2] - public["target"]) ** 2).sum(1)
    scores += 0.05 * predicted["variance"][:, -1, 0, :2].sum(1)
    scores += 2 * np.maximum(0, 0.25 - predicted["states"][:, :, 0, 1]).mean(1)
    candidates = [
        {
            "action": actions[i, 0].tolist(),
            "terminal": predicted["states"][i, -1].tolist(),
            "variance": predicted["variance"][i, -1].tolist(),
        }
        for i in range(9)
    ]
    return (
        {**observed, "target": public["target"], "horizon": horizon},
        candidates,
        (-scores).tolist(),
        adapter.provenance,
    )


def collect(tasks, assets, root):
    from .verify import outcomes

    rows = []
    for task in tasks:
        started = time.perf_counter()
        public, candidates, scores, provenance = proposals(task, assets)
        # Keep all actual alternatives; deduplicate identical emitted programs only.
        unique = {}
        for p, s in zip(candidates, scores):
            key = digest(p)
            if key not in unique or s > unique[key][1]:
                unique[key] = (p, s)
        candidates, scores = map(list, zip(*unique.values()))
        row = {
            "identity": task["identity"],
            "domain": task["domain"],
            "split": task["split"],
            "public": public,
            "proposals": candidates,
            "scores": scores,
            "features": [encode(task["domain"], public, p, s) for p, s in zip(candidates, scores)],
            "provenance": provenance,
        }
        # This immutable file proves which features/proposals existed before scoring.
        path = root / "proposals" / f"{task['identity']}.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(row, sort_keys=True, allow_nan=False))
        row["outcomes"] = outcomes(task, candidates, assets.loaded.get("coding", [None] * 4)[3])
        row["labels"] = [float(v["success"]) for v in row["outcomes"]]
        row["seconds"] = time.perf_counter() - started
        row["proposal_digest"] = digest(json.loads(path.read_text()))
        rows.append(row)
        with (root / "rows.jsonl").open("a") as f:
            f.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        print(
            json.dumps(
                {
                    "domain": task["domain"],
                    "split": task["split"],
                    "completed": len(rows),
                    "candidates": len(candidates),
                    "seconds": row["seconds"],
                }
            ),
            flush=True,
        )
    return rows
