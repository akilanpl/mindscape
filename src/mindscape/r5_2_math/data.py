"""Fresh fixed split generator. No historical math manifest, checkpoints or lockbox reads."""

import json
import random
from pathlib import Path

from mindscape.v3.schema import digest


def identity(task):
    operands = list(task["operands"])
    if len(task["operations"]) == 1 and task["operations"][0] in ("add", "mul"):
        operands.sort()
    return digest(
        {"operands": operands, "operations": task["operations"], "namespace": "R5.2-math"}
    )


def build(protocol):
    rng = random.Random(protocol["seed"])
    rows = []
    used = set()
    historical = json.loads(Path("configs/v3_math/manifest.json").read_text())["body"]

    def original_id(operands, operations):
        values = list(operands)
        if len(operations) == 1 and operations[0] in ("add", "mul"):
            values.sort()
        return digest({"operands": values, "operations": operations})

    forbidden = set(historical["lockbox_ids"])
    forbidden.update(
        original_id(t["operands"], t["operations"])
        for split in historical["tasks"].values()
        for t in split
    )

    def add(split, family, a, b, ops):
        row = {
            "split": split,
            "family": family,
            "operands": [a, b] if len(ops) == 1 else [a, b, rng.randint(-99, 99)],
            "operations": ops,
        }
        row["identity"] = identity(row)
        if row["identity"] in used or original_id(row["operands"], row["operations"]) in forbidden:
            return False
        used.add(row["identity"])
        rows.append(row)
        return True

    for split, count in [
        ("train", protocol["train_tasks"]),
        ("development", protocol["development_tasks"]),
    ]:
        i = 0
        attempt = 0
        while i < count:
            attempt += 1
            if attempt > 10000:
                raise ValueError("Fixed generation collision budget exceeded")
            length = rng.randint(1, 3)
            bound = 10**length - 1
            a = rng.randint(0, bound) * rng.choice([-1, 1])
            b = rng.randint(0, bound) * rng.choice([-1, 1])
            if add(split, "short_signed", a, b, [("add", "sub", "mul")[i % 3]]):
                i += 1
    for family in protocol["families"]:
        for i in range(protocol["ood_tasks_per_family"]):
            op = ("add", "sub", "mul")[i % 3]
            length = 4 + i % 5
            if family == "length":
                a = rng.randint(10 ** (length - 1), 10**length - 1)
                b = rng.randint(10 ** (length - 1), 10**length - 1)
            elif family == "carry_borrow":
                length = 4 + i // 3
                a = (
                    int("9" * length)
                    if op == "add"
                    else 10**length
                    if op == "sub"
                    else int("9" * length)
                )
                b = i + 1
            elif family == "sign":
                a = -rng.randint(1000, 99999)
                b = rng.randint(1000, 99999) * (-1 if i % 2 else 1)
            elif family == "composition":
                a = rng.randint(-9999, 9999)
                b = rng.randint(-9999, 9999)
            elif family == "operand_range":
                a = rng.randint(10**8, 10**10 - 1) * rng.choice([-1, 1])
                b = rng.randint(10**8, 10**10 - 1) * rng.choice([-1, 1])
            else:
                a = int(str(rng.randint(111, 999)) + "8")
                b = int(str(rng.randint(1, 99)) + "7")
            ops = [op, ("add", "sub", "mul")[(i + 1) % 3]] if family == "composition" else [op]
            # Carry-family public specifications can collide; use fixed distinct shifts, never outcomes.
            if not add("ood", family, a, b, ops):
                if family == "carry_borrow":
                    b += 1000 + i
                    if not add("ood", family, a, b, ops):
                        raise ValueError("Frozen task collision")
                else:
                    raise ValueError("Frozen task collision")
    return {
        "protocol": protocol,
        "protocol_hash": digest(protocol),
        "tasks": rows,
        "task_hash": digest(rows),
        "historical_public_and_opaque_id_exclusion": True,
        "sealed_contents_opened": False,
    }


if __name__ == "__main__":
    path = Path("configs/r5_2_math/tasks.json")
    if path.exists():
        raise FileExistsError("Already frozen")
    path.write_text(
        json.dumps(build(json.loads(Path("configs/r5_2_math/protocol.json").read_text())), indent=2)
        + "\n"
    )
