"""Fresh predeclared Cartesian validity/operator cells; no outcome-based selection."""

import json
import random
from pathlib import Path

from mindscape.v3.schema import digest
from mindscape.v5.data import graph


def canonical(task):
    p = dict(task["public"])
    if task["domain"] == "coding":
        p.pop("entry", None)
    return digest({"domain": task["domain"], "public": p})


def build(protocol):
    rng = random.Random(protocol["seed"])
    rows = []
    historical = json.loads(Path("configs/v5/tasks.json").read_text())[
        "tasks"
    ]  # Public frozen specs only.
    old = {canonical(t) for t in historical}
    for split, count in protocol["tasks_per_domain"].items():
        composition = split == "composition"
        for domain in ("mathematics", "coding", "physics"):
            # Validity is independently balanced within every coding operation before shuffling.
            cells = [
                (op, valid)
                for _ in range(count // 9)
                for op in ("add", "sub", "mul")
                for valid in (True, True, False)
            ]
            rng.shuffle(cells)
            for i, (op, valid) in enumerate(cells):
                if domain == "mathematics":
                    n = 3 if composition else 2
                    public = {
                        "values": [rng.randint(51, 89) for _ in range(n)],
                        "operation": "add" if op in ("add", "mul") else "sub",
                        "destination": 0 if valid else n,
                        "source": n - 1,
                        "amount": rng.choice([1, 2, 3, 4]),
                    }
                elif domain == "coding":
                    public = {
                        "x": rng.choice([7, 8, 9, 10, -7, -8, -9]),
                        "constants": [rng.randint(11, 19) for _ in range(2 if composition else 1)],
                        "operations": [op]
                        + ([("add", "sub", "mul")[(i + 1) % 3]] if composition else []),
                        "target_leaf": i % 2 if valid and composition else 0 if valid else 9,
                        "increment": rng.choice([-4, -2, 2, 4]),
                        "entry": f"r51_{protocol['seed']}_{split}_{i}",
                    }
                else:
                    n = 2 if composition else 1
                    public = {
                        "positions": [[j * 2.0, 2.0 + rng.random()] for j in range(n)],
                        "velocities": [[rng.uniform(-1, 1), rng.uniform(-1, 1)] for _ in range(n)],
                        "masses": [rng.choice([1.4, 1.6, 1.8, 2.0]) for _ in range(n)],
                        "destination": i % n if valid else n,
                        "impulse": [rng.choice([-2.0, 2.0]), rng.choice([-1.0, 1.0])],
                        "gravity": 0.0,
                        "scene_seed": protocol["seed"]
                        + i
                        + (100 if split == "qualification" else 200 if composition else 0),
                    }
                task = {"domain": domain, "split": split, "public": public}
                task["identity"] = digest({"domain": domain, "public": public})
                if canonical(task) in old:
                    raise ValueError("Freshness collision; stop, never resample")
                rows.append(task)
    identities = [canonical(t) for t in rows]
    if len(identities) != len(set(identities)) or len(rows) != 108:
        raise ValueError("Independent identity/budget failure; no retries")
    return {
        "version": "R5.1",
        "protocol": protocol,
        "protocol_hash": digest(protocol),
        "tasks": rows,
        "task_hash": digest(rows),
        "public_graph_hashes": {t["identity"]: digest(graph(t)) for t in rows},
        "scorers": "Unchanged V5 execute(task): independent register interpreter, actual before/after WASI, pinned Pymunk.",
        "scorer_policy": "Targets never generated at freeze. No historical sealed task contents read.",
    }


if __name__ == "__main__":
    p = Path("configs/r5_1/tasks.json")
    if p.exists():
        raise FileExistsError("Manifest already frozen")
    p.write_text(
        json.dumps(build(json.loads(Path("configs/r5_1/protocol.json").read_text())), indent=2)
        + "\n"
    )
