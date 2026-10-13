"""Fresh graph tasks; no neural imports, model outputs, old evidence or sealed task reads."""

import json
import random
from pathlib import Path

from mindscape.v3.schema import digest

from .schema import DIMS, SCALES, validate

DOMAINS = ("mathematics", "coding", "physics")


def tasks(protocol):
    rng = random.Random(protocol["seed"])
    result = []
    for split, count in protocol["tasks_per_domain"].items():
        for domain in DOMAINS:
            for i in range(count):
                invalid = i % 3 == 2
                composition = split == "composition"
                if domain == "mathematics":
                    n = 3 if composition else 2
                    public = {
                        "values": [rng.randint(11, 45) for _ in range(n)],
                        "operation": ("add", "sub")[i % 2],
                        "destination": n if invalid else 0,
                        "source": n - 1,
                        "amount": rng.randint(1, 4),
                    }
                elif domain == "coding":
                    public = {
                        "x": rng.randint(-6, 9),
                        "constants": [rng.randint(2, 9) for _ in range(2 if composition else 1)],
                        "operations": [("add", "sub", "mul")[i % 3]]
                        + (["add" if i % 2 else "mul"] if composition else []),
                        "target_leaf": 9 if invalid else i % (2 if composition else 1),
                        "increment": rng.choice([-4, -2, 2, 4]),
                        "entry": f"v5_{protocol['seed']}_{split}_{i}",
                    }
                else:
                    n = 2 if composition else 1
                    public = {
                        "positions": [[float(j * 2), 2.0 + rng.random()] for j in range(n)],
                        "velocities": [[rng.uniform(-1, 1), rng.uniform(-1, 1)] for _ in range(n)],
                        "masses": [rng.uniform(0.8, 1.8) for _ in range(n)],
                        "destination": n if invalid else i % n,
                        "impulse": [rng.choice([-2.0, 2.0]), rng.choice([-1.0, 1.0])],
                        "gravity": 0.0,
                        "scene_seed": protocol["seed"]
                        + i
                        + (100 if split == "qualification" else 200 if composition else 0),
                    }
                row = {"domain": domain, "split": split, "public": public}
                row["identity"] = digest({"domain": domain, "public": public})
                result.append(row)
    if len(result) != protocol["maximum_tasks"] or len({r["identity"] for r in result}) != len(
        result
    ):
        raise ValueError("Budget/identity collision")
    return result


def expression_nodes(public):
    # Node order is leaves followed by public AST expression nodes, not evaluation targets.
    values = [public["x"], *public["constants"]]
    kinds = ["input"] + ["literal"] * len(public["constants"])
    links = []
    if len(public["constants"]) == 1:
        links.append((len(values), 0, 1, public["operations"][0]))
        kinds.append("expression")
        values.append(0)
    else:
        links.append((3, 0, 1, public["operations"][0]))
        links.append((4, 3, 2, public["operations"][1]))
        kinds += ["expression", "expression"]
        values += [0, 0]
    return values, kinds, links


def graph(task):
    d = task["domain"]
    p = task["public"]
    relations = []
    scale = SCALES[d]
    if d == "mathematics":
        n = len(p["values"])
        entities = [
            [
                v / scale,
                float(i == p["destination"]),
                float(i == p["source"]),
                float(i) / 4,
                0.0,
                1.0,
            ]
            for i, v in enumerate(p["values"])
        ]
        relations = [
            [
                [float(i == p["destination"] and j == p["source"]), float(i == j), 0.0]
                for j in range(n)
            ]
            for i in range(n)
        ]
        action = [
            float(p["operation"] == "add"),
            float(p["operation"] == "sub"),
            p["destination"] / 7,
            p["source"] / 7,
            p["amount"] / 4,
            0.0,
            0.0,
            0.0,
        ]
    elif d == "coding":
        values, kinds, links = expression_nodes(p)
        n = len(values)
        entities = []
        ops = {index: op for index, _, _, op in links}
        for i, (value, kind) in enumerate(zip(values, kinds)):
            entities.append(
                [
                    value / scale,
                    float(kind == "input"),
                    float(kind == "literal"),
                    float(kind == "expression"),
                    float(ops.get(i) == "add"),
                    float(ops.get(i) == "sub"),
                    float(ops.get(i) == "mul"),
                    float(i == p["target_leaf"] + 1),
                    i / 7,
                    1.0,
                ]
            )
        relations = [[[0.0, 0.0, 0.0] for _ in range(n)] for _ in range(n)]
        for parent, left, right, _ in links:
            relations[parent][left] = [1.0, 0.0, 0.0]
            relations[parent][right] = [0.0, 1.0, 0.0]
            relations[left][parent] = [0.0, 0.0, 1.0]
            relations[right][parent] = [0.0, 0.0, 1.0]
        action = [1.0, 0.0, (p["target_leaf"] + 1) / 7, 0.0, p["increment"] / 4, 0.0, 0.0, 0.0]
    else:
        n = len(p["masses"])
        entities = [
            [
                x / scale,
                y / scale,
                vx / scale,
                vy / scale,
                p["masses"][i] / 2,
                float(i == p["destination"]),
                0.25,
                1.0,
            ]
            for i, ((x, y), (vx, vy)) in enumerate(zip(p["positions"], p["velocities"]))
        ]
        relations = [
            [
                [
                    float(i != j),
                    (p["positions"][j][0] - p["positions"][i][0]) / 4,
                    (p["positions"][j][1] - p["positions"][i][1]) / 4,
                ]
                for j in range(n)
            ]
            for i in range(n)
        ]
        action = [
            1.0,
            0.0,
            p["destination"] / 7,
            0.0,
            p["impulse"][0] / 2,
            p["impulse"][1] / 2,
            p["gravity"] / 10,
            1 / 30,
        ]
    observed = [[1] * DIMS[d] for _ in entities]
    if d == "coding":
        for i, kind in enumerate(kinds):
            if kind == "expression":
                observed[i][0] = 0  # Internal prior evaluation is not supplied to inference.
    return validate(
        {
            "domain": d,
            "entities": entities,
            "observed": observed,
            "relations": relations,
            "action": action,
        }
    )


def public_before(task):
    p = task["public"]
    d = task["domain"]
    if d == "mathematics":
        return [[float(v)] for v in p["values"]]
    if d == "physics":
        return [
            [*position, *velocity] for position, velocity in zip(p["positions"], p["velocities"])
        ]
    # Internal expression state is not fed into the encoder; effects for these nodes are still predicted.
    values, _, links = expression_nodes(p)
    for i, a, b, op in links:
        if op == "add":
            values[i] = values[a] + values[b]
        elif op == "sub":
            values[i] = values[a] - values[b]
        else:
            values[i] = values[a] * values[b]
    return [[float(v)] for v in values]


def freeze(path):
    protocol = json.loads(Path("configs/v5/protocol.json").read_text())
    if Path(path).exists():
        raise FileExistsError("Manifest immutable")
    rows = tasks(protocol)
    manifest = {
        "protocol": protocol,
        "protocol_hash": digest(protocol),
        "tasks": rows,
        "task_hash": digest(rows),
        "information": "Graphs computed only from public inputs; targets independently executed afterward.",
        "historical_scope": "New V5 microtasks; no historical sealed task files read. Domain-qualified identities differ from V3/V4 task schemas.",
    }
    Path(path).write_text(json.dumps(manifest, indent=2) + "\n")
