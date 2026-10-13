"""Separate public encoders. No outcome, split, identity, private result or reference input."""

import hashlib
import json
import math
import re

DIMS = {"mathematics": 20, "coding": 160, "physics": 112}


def encode(domain, public, proposal, score):
    if domain == "mathematics":
        if set(public) != {"operation", "left", "right", "incoming"}:
            raise ValueError("Math public boundary")
        if set(proposal) != {"claim"}:
            raise ValueError("Math proposal boundary")
        c = proposal["claim"]
        if set(c) != {"operation", "raw", "digit", "carry", "next_digit", "next_carry"}:
            raise ValueError("Unexpected claim field")
        x = [float(public["operation"] == op) for op in ("add", "sub", "mul")]
        x += [public[k] / 10 for k in ("left", "right", "incoming")]
        x += [
            c[k] / 100 if k == "raw" else c[k] / 10
            for k in ("operation", "raw", "digit", "carry", "next_digit", "next_carry")
        ]
    elif domain == "coding":
        if set(public) != {"repository", "contract", "visible_tests", "actual_visible_result"}:
            raise ValueError("Coding public boundary")
        if set(proposal) != {"repository"}:
            raise ValueError("Coding proposal boundary")
        # Two independent token bags, retaining numeric literal tokens (unlike V3).
        # Bounded hash collisions and lack of token ordering are disclosed limitations.
        x = []
        for value in (public, proposal):
            bag = [0.0] * 64
            tokens = re.findall(r"[A-Za-z_]+|\d+|[^\w\s]", json.dumps(value, sort_keys=True))
            for token in tokens:
                index = int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], "big") % 64
                bag[index] += 1 / 32
            x += [min(4.0, v) for v in bag]
        for value in (public["repository"], proposal["repository"]):
            nums = [int(v) for v in re.findall(r"(?<!\w)-?\d+", "\n".join(value.values()))]
            x += [math.tanh(v / 100) for v in nums[:8]] + [0.0] * max(0, 8 - len(nums))
    elif domain == "physics":
        if set(public) != {"observations", "masks", "contexts", "target", "horizon"}:
            raise ValueError("Physics public boundary")
        if set(proposal) != {"action", "terminal", "variance"}:
            raise ValueError("Physics proposal boundary")

        def flatten(v):
            if isinstance(v, list):
                return [z for item in v for z in flatten(item)]
            return [float(v)]

        x = flatten(public["observations"][-1]) + flatten(public["masks"][-1])
        x += flatten(public["contexts"][-1]) + flatten(public["target"])
        x += [public["horizon"] / 32] + flatten(proposal["action"])
        x += flatten(proposal["terminal"]) + flatten(proposal["variance"])
        x = [math.tanh(v / 4) for v in x]
    else:
        raise ValueError("Unknown domain")
    x = x[: DIMS[domain] - 1] + [0.0] * max(0, DIMS[domain] - 1 - len(x))
    x.append(math.tanh(float(score) / 10))
    if not all(math.isfinite(v) for v in x):
        raise ValueError("Nonfinite public encoding")
    return x
