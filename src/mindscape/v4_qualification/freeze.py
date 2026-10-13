"""Build public tasks/scorers before inference, without consulting candidate outcomes."""

import ast
import json
import random
from pathlib import Path

from mindscape.v3.schema import digest

PROPOSAL = Path("configs/v4/fresh-development-proposal.json")
MANIFEST = Path("configs/v4/qualification_v1/tasks.json")
OFFSETS = ((0, 0), (0.5, 0), (0, 0.5), (-0.5, 0))
DOMAINS = ("mathematics", "physics", "coding")


def canonical_math(p):
    a, b = p["left"], p["right"]
    if p["operation"] == "add":
        a, b = sorted((a, b))
    return p["operation"], a, b, p["incoming"]


def code_key(repository):
    return digest(
        {k: ast.dump(ast.parse(v), include_attributes=False) for k, v in repository.items()}
    )


def physics_public(n, seed):
    import numpy as np

    from mindscape.v3.physics_world.data import generate_scene
    from mindscape.v3.physics_world.environment import Environment
    from mindscape.v3.physics_world.schema import Action, observe

    index = 2000 + n
    scene = generate_scene("motion", "iid", index, seed)
    env = Environment(scene)
    for _ in range(3):
        state = env.step(Action.from_vector(np.zeros((3, 4))))
    observed = observe(state, scene, 3)
    offset = OFFSETS[index % 4]
    target = [float(observed["values"][0, i]) + offset[i] for i in range(2)]
    return {
        "family": "motion",
        "seed": seed,
        "index": index,
        "horizon": 8 if n % 2 == 0 else 16,
        "target": target,
    }


def coding_public(n, seed):
    t = 101 + n
    entry = f"qual_{seed}_{n}"
    family = ("inclusive", "exclusive", "literal", "count")[n % 4]
    condition = f"item {'>' if family == 'exclusive' else '>='} {t}"
    desired = "count" if family == "count" else "sum"
    bug = {
        "inclusive": f"item > {t}",
        "exclusive": f"item >= {t}",
        "literal": f"item >= {t + 7}",
        "count": condition,
    }[family]
    source = f"def {entry}(values):\n    return sum(item for item in values if {bug})\n"

    def suite(values):
        result = []
        for v in values:
            chosen = [x for x in v if (x > t if family == "exclusive" else x >= t)]
            result.append(
                {
                    "args": [v],
                    "expected": len(chosen) if desired == "count" else sum(chosen),
                    "expected_type": "int",
                    "unchanged": True,
                }
            )
        return {"entry": "solution." + entry, "cases": result}

    visible = suite([[t], [t, t + 1, t + 8]])
    private = suite([[], [t - 1], [t, t], [t + 1], [t + 6, t + 7], [t - 2, t, t + 9]])
    public = {
        "repository": {"solution.py": source},
        "contract": {
            "entry": "solution." + entry,
            "description": f"Return the {desired} of values {'strictly greater than' if family == 'exclusive' else 'at least'} {t}. Do not mutate input.",
            "inputs": ["list of integers"],
            "outputs": "int",
            "invariants": ["Inputs remain unchanged."],
        },
        "visible_tests": visible,
    }
    return public, private, family


def build():
    from mindscape.v4.data import tasks as old_tasks

    proposal = json.loads(PROPOSAL.read_text())
    seed = proposal["seed"]
    old = old_tasks()  # Public identities only; no historical result labels or sealed data.
    old_math = {canonical_math(t["public"]) for t in old if t["domain"] == "mathematics"}
    old_code = {code_key(t["public"]["repository"]) for t in old if t["domain"] == "coding"}
    old_scenes = {
        (t["public"]["seed"], t["public"]["index"]) for t in old if t["domain"] == "physics"
    }
    old_ids = {t["identity"] for t in old}
    rng = random.Random(seed)
    pool = [
        {"operation": op, "left": a, "right": b, "incoming": c}
        for op in ("add", "sub")
        for a in range(10)
        for b in range(10)
        for c in (0, 1)
    ]
    rng.shuffle(pool)
    seen = set(old_math)
    domains = {d: [] for d in DOMAINS}
    for n in range(12):
        split = "train" if n < 8 else "qualification"
        op = ("add", "sub")[(n // 2) % 2]
        incoming = n % 2
        p = next(
            p
            for p in pool
            if p["operation"] == op and p["incoming"] == incoming and canonical_math(p) not in seen
        )
        seen.add(canonical_math(p))
        raw = (
            p["left"] + p["right"] + p["incoming"]
            if op == "add"
            else p["left"] - p["right"] - p["incoming"]
        )
        domains["mathematics"].append(
            {
                "domain": "mathematics",
                "split": split,
                "index": n,
                "public": p,
                "scorer": {
                    "kind": "write_only_digit_carry",
                    "digit": raw % 10,
                    "carry": raw // 10 if op == "add" else int(raw < 0),
                },
            }
        )
        p = physics_public(n, seed)
        if (p["seed"], p["index"]) in old_scenes:
            raise ValueError("Old scene identity")
        domains["physics"].append(
            {
                "domain": "physics",
                "split": split,
                "index": n,
                "public": p,
                "scorer": {
                    "kind": "independent_pymunk",
                    "maximum_goal_distance": 0.75,
                    "warmup_steps": 3,
                },
                "target_rule": {
                    "public_observation_step": 3,
                    "offset": list(OFFSETS[(2000 + n) % 4]),
                },
            }
        )
        p, private, family = coding_public(n, seed)
        if code_key(p["repository"]) in old_code:
            raise ValueError("Old coding AST")
        domains["coding"].append(
            {
                "domain": "coding",
                "split": split,
                "index": n,
                "public": p,
                "private": private,
                "family": family,
                "scorer": {"kind": "visible_and_private_WASI_all_pass"},
            }
        )
    # Close gate cohorts early; surplus source training requests occur only after every gate passes.
    ordered = []
    for indices in (range(4), range(8, 12), range(4, 8)):
        for domain in DOMAINS:
            ordered.extend(domains[domain][i] for i in indices)
    for t in ordered:
        t["identity"] = digest({"domain": t["domain"], "public": t["public"]})
    if (
        len(ordered) != 36
        or len({t["identity"] for t in ordered}) != 36
        or any(t["identity"] in old_ids for t in ordered)
    ):
        raise ValueError("Request budget or freshness")
    return {
        "version": 1,
        "proposal_sha256": digest(proposal),
        "seed": seed,
        "maximum_requests": 36,
        "tasks": ordered,
        "tasks_sha256": digest(ordered),
        "execution_order": "first-four train by mathematics/physics/coding; qualification by same order; remaining train by same order",
        "stop_policy": "Check a gate when its four-task cohort is complete; stop before next request on failure; no retries/resume.",
        "scorer_policy": "Only independently executed outcomes count. Specialist scores are predictions, not verification scores.",
    }


if __name__ == "__main__":
    if MANIFEST.exists():
        raise FileExistsError("Frozen manifest already exists")
    MANIFEST.write_text(json.dumps(build(), indent=2, allow_nan=False) + "\n")
