"""Prospective deterministic splits and a separately keyed, encrypted lockbox."""

import hashlib
import json
import random
from dataclasses import asdict
from pathlib import Path

from mindscape.v3.schema import canonical, digest

from .environment import Problem


def make_problem(rng, split, family, index):
    op = ("add", "sub", "mul")[index % 3]
    if family == "carry_chain":
        length = 3 + index % 4
        a, b = (10**length - 1, 1) if op != "sub" else (10**length, 1)
        # Multiplication by 9 creates repeated carry under hand-defined place-value schedule.
        if op == "mul":
            b = 9
        a += (index // 12) * 10 ** (length + 1)
    else:
        length = (
            rng.choice((1, 2))
            if family in ("familiar", "sign_change", "digit_composition", "composition")
            else rng.choice((3, 4, 5, 6))
        )
        alphabet = "01234567" if family != "digit_composition" else "0189"

        def number(n):
            return int(
                rng.choice(alphabet.replace("0", ""))
                + "".join(rng.choice(alphabet) for _ in range(n - 1))
            )

        a, b = (
            number(length),
            number(
                rng.choice((1, 2))
                if family in ("familiar", "sign_change", "digit_composition", "composition")
                else length
            ),
        )
    if family == "sign_change":
        b = -b
        if index % 2:
            a = -a
    elif family == "familiar" and index % 4 == 0:
        a = -a
    if family == "composition":
        third = rng.randint(1, 7)
        return Problem((a, b, third), (op, ("mul", "add", "sub")[index % 3]), split, family)
    return Problem((a, b), (op,), split, family)


def freeze(root, seed=20261013):
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng, used, partitions = random.Random(seed), set(), {}
    plan = {
        "train": [("familiar", 1000)],
        "validation": [("familiar", 60)],
        "iid": [("familiar", 90)],
        "ood": [
            (f, 30)
            for f in ("length", "carry_chain", "digit_composition", "sign_change", "composition")
        ],
        "lockbox": [
            (f, 18)
            for f in ("length", "carry_chain", "digit_composition", "sign_change", "composition")
        ],
    }
    for split, families in plan.items():
        tasks = []
        for family, count in families:
            index, attempts = 0, 0
            while index < count:
                task = make_problem(rng, split, family, attempts)
                attempts += 1
                if task.identity in used:
                    continue
                used.add(task.identity)
                tasks.append(task)
                index += 1
        partitions[split] = tasks
    body = {
        "schema": "mindscape.v3.math/1",
        "seed": seed,
        "base": 10,
        "budgets": [10, 25, 50, 100, 250, 500, 1000],
        "seeds": [0, 1, 2],
        "primary": "all_predeclared_structural_ood_exact_accuracy",
        "target_family": "length",
        "aspirational_target": 0.5,
        "tasks": {s: [asdict(t) for t in ts] for s, ts in partitions.items() if s != "lockbox"},
        "lockbox_ids": [t.identity for t in partitions["lockbox"]],
    }
    raw_lockbox = canonical([asdict(t) for t in partitions["lockbox"]]).encode()
    key = AESGCM.generate_key(bit_length=256)
    nonce = hashlib.sha256(key + b"lockbox").digest()[:12]
    ciphertext = AESGCM(key).encrypt(nonce, raw_lockbox, b"mindscape.v3.math/1")
    body["lockbox_cipher_sha256"] = hashlib.sha256(ciphertext).hexdigest()
    body["lockbox_plain_sha256"] = hashlib.sha256(raw_lockbox).hexdigest()
    for path, content in (
        (root / "manifest.json", canonical({"body": body, "sha256": digest(body)}).encode()),
        (root / "lockbox.aesgcm", nonce + ciphertext),
    ):
        with path.open("xb") as stream:
            stream.write(content)
    key_path = root.parent / "math-lockbox-key.bin"
    with key_path.open("xb") as stream:
        stream.write(key)
    key_path.chmod(0o600)
    return body


def load(root):
    value = json.loads((Path(root) / "manifest.json").read_text())
    if digest(value["body"]) != value["sha256"]:
        raise ValueError("Frozen manifest mismatch")
    tasks = {
        split: tuple(
            Problem(tuple(t["operands"]), tuple(t["operations"]), t["split"], t["family"])
            for t in rows
        )
        for split, rows in value["body"]["tasks"].items()
    }
    ids = []
    for split, rows in tasks.items():
        if any(t.split != split for t in rows):
            raise ValueError("Partition mismatch")
        ids.extend(t.identity for t in rows)
    ids += value["body"]["lockbox_ids"]
    if len(ids) != len(set(ids)):
        raise ValueError("Canonical split leakage")
    return value, tasks
