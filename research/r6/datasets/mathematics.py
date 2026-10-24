"""Original integer arithmetic; exact references never enter neural inputs.

IID partitions share templates and ranges, but partition canonical expressions
by a stable hash. OOD holds out both magnitudes and parenthesized structures.
Only the evaluator or data builder calls the exact reference functions.
"""
from __future__ import annotations

import ast
import hashlib
import json
import random
from collections import Counter
from fractions import Fraction

SPLITS = ("train", "dev", "final_iid", "final_ood")
VERSION = "r6-v1"
OPERATIONS = ("+", "-", "*")


def _digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _parse(expression):
    if not isinstance(expression, str) or len(expression) > 128:
        raise ValueError("Expected a short arithmetic expression")
    tree = ast.parse(expression, mode="eval")
    nodes = list(ast.walk(tree))
    if len(nodes) > 40:
        raise ValueError("Expression too complex")
    allowed = (ast.Expression, ast.BinOp, ast.Constant, ast.UnaryOp,
               ast.Add, ast.Sub, ast.Mult, ast.USub, ast.UAdd)
    if any(not isinstance(node, allowed) for node in nodes):
        raise ValueError("Unsupported expression syntax")
    if any(type(node.value) is not int or abs(node.value) > 9999
           for node in nodes if isinstance(node, ast.Constant)):
        raise ValueError("Only bounded integer literals are supported")
    return tree.body


def reference_answer(expression):
    """Restricted AST traversal with rational arithmetic, never eval/exec."""
    def visit(node):
        if isinstance(node, ast.Constant):
            return Fraction(node.value)
        if isinstance(node, ast.UnaryOp):
            return -visit(node.operand) if isinstance(node.op, ast.USub) else visit(node.operand)
        left, right = visit(node.left), visit(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        return left * right
    value = visit(_parse(expression))
    if value.denominator != 1:
        raise ValueError("Non-integral result")
    return str(value.numerator)


def structured_answer(structured):
    """Independent integer operator implementation, no AST or expression text."""
    operands = structured["operands"]
    operators = structured["operators"]
    if any(type(value) is not int for value in operands):
        raise ValueError("Integer operands required")
    def apply(left, operator, right):
        if operator == "+":
            return left + right
        if operator == "-":
            return left - right
        if operator == "*":
            return left * right
        raise ValueError("Unknown operator")
    if len(operands) == 2 and len(operators) == 1:
        value = apply(operands[0], operators[0], operands[1])
    elif len(operands) == 3 and len(operators) == 2:
        value = apply(apply(operands[0], operators[0], operands[1]), operators[1], operands[2])
    else:
        raise ValueError("Unsupported structured expression")
    return str(value)


def canonical_key(expression):
    """Normalize signs, associativity and commutative order, without solving.

    Addition/subtraction become sorted signed operand structures. Multiplication
    factors are sorted; no distribution, constant folding or answer collapse.
    """
    def normalize(node, sign=1):
        if isinstance(node, ast.Constant):
            return ("integer", sign * node.value)
        if isinstance(node, ast.UnaryOp):
            return normalize(node.operand, -sign if isinstance(node.op, ast.USub) else sign)
        if isinstance(node.op, (ast.Add, ast.Sub)):
            children = []
            for child in (normalize(node.left, sign), normalize(node.right, -sign if isinstance(node.op, ast.Sub) else sign)):
                children.extend(child[1] if child[0] == "sum" else [child])
            return ("sum", tuple(sorted(children, key=repr)))
        children = []
        product_sign = sign
        for child in (normalize(node.left), normalize(node.right)):
            if child[0] == "product":
                product_sign *= child[2]
                children.extend(child[1])
            else:
                children.append(child)
        # Normalize all literal signs into one sign, including -(a*b).
        positive = []
        for child in children:
            if child[0] == "integer":
                product_sign *= -1 if child[1] < 0 else 1
                child = ("integer", abs(child[1]))
            positive.append(child)
        return ("product", tuple(sorted(positive, key=repr)), product_sign)
    return json.dumps(normalize(_parse(expression)), separators=(",", ":"))


def verify_record(record):
    observed = record["structured_input"]
    expression = observed["expression"]
    if record["input"] != "Calculate " + expression + ".":
        raise ValueError("Text and structured expression differ")
    operands, operators = observed["operands"], observed["operators"]
    if len(operands) == 2:
        rendered = f"({operands[0]}) {operators[0]} ({operands[1]})"
    elif len(operands) == 3:
        rendered = f"(({operands[0]}) {operators[0]} ({operands[1]})) {operators[1]} ({operands[2]})"
    else:
        raise ValueError("Unsupported arity")
    if expression != rendered:
        raise ValueError("Expression does not match observed operands and operators")
    first, second = reference_answer(expression), structured_answer(observed)
    if first != second or record["target"] != {"answer": first}:
        raise ValueError("Independent arithmetic reference mismatch")
    if record["canonical_key"] != canonical_key(expression):
        raise ValueError("Canonical key mismatch")
    return True


def carry_borrow_category(record):
    """Evaluation stratification only; never add this derived tag to inputs."""
    observed = record["structured_input"]
    if len(observed["operands"]) != 2:
        return "compositional"
    a, b = observed["operands"]
    op = observed["operators"][0]
    if a < 0 or b < 0:
        return "signed"
    if op == "*":
        return "multiplication"
    if op == "+":
        carries, incoming = 0, 0
        while a or b:
            incoming = int(a % 10 + b % 10 + incoming >= 10)
            carries += incoming
            a //= 10
            b //= 10
        return "carry" if carries else "no-carry"
    borrows, incoming = 0, 0
    while a or b:
        incoming = int(a % 10 - incoming < b % 10)
        borrows += incoming
        a //= 10
        b //= 10
    return "borrow" if borrows else "no-borrow"


def generate_with_metadata(split: str, seed: int, count: int):
    if split not in SPLITS or type(seed) is not int or type(count) is not int or count < 0:
        raise ValueError("Expected a supported split, integer seed and nonnegative count")
    rng = random.Random(seed)
    rows, seen = [], set()
    rejected = Counter()
    attempts = 0
    while len(rows) < count:
        attempts += 1
        if attempts > max(10000, count * 200):
            raise ValueError("Requested count exceeds practical unique split capacity")
        index = len(rows)
        op = OPERATIONS[index % 3]
        compositional = split == "final_ood" and index % 2 == 1
        lo, hi = (100, 9999) if split == "final_ood" and not compositional else (10, 99)
        operands = [rng.randint(lo, hi) * rng.choice((-1, 1)) for _ in range(3 if compositional else 2)]
        operators = [op, OPERATIONS[(index // 3 + 1) % 3]] if compositional else [op]
        if compositional:
            expression = f"(({operands[0]}) {operators[0]} ({operands[1]})) {operators[1]} ({operands[2]})"
            template = "ood-composition-" + "".join(operators)
        else:
            expression = f"({operands[0]}) {op} ({operands[1]})"
            template = ("ood-range-" if split == "final_ood" else "binary-") + op
        key = canonical_key(expression)
        if split != "final_ood" and int(_digest(key), 16) % 3 != SPLITS.index(split):
            rejected["split-partition"] += 1
            continue
        if key in seen:
            rejected["canonical-duplicate"] += 1
            continue
        structured = {"expression": expression, "operands": operands, "operators": operators}
        answer = structured_answer(structured)
        if answer != reference_answer(expression):
            raise RuntimeError("Independent oracle disagreement; generation aborted")
        row = {"record_id": "math-" + _digest(key)[:24], "domain": "mathematics",
               "topic": "integer-" + {"+": "addition", "-": "subtraction", "*": "multiplication"}[op],
               "difficulty": "structural-ood" if compositional else "range-ood" if split == "final_ood" else "foundation",
               "input": "Calculate " + expression + ".", "structured_input": structured,
               "target": {"answer": answer}, "verification_method": "restricted-ast-fraction+structured-integer",
               "source": "generated", "split": split, "generator_version": VERSION,
               "template_id": template, "canonical_key": key, "features": []}
        verify_record(row)
        rows.append(row)
        seen.add(key)
    metadata = {"split": split, "seed": seed, "requested": count, "accepted": len(rows),
                "attempts": attempts, "rejected": dict(rejected), "invalid": 0,
                "generator_version": VERSION, "template_counts": dict(Counter(r["template_id"] for r in rows)),
                "topic_counts": dict(Counter(r["topic"] for r in rows)),
                "max_input_bytes": max((len(r["input"].encode()) for r in rows), default=0),
                "iid_policy": "canonical hash partition; shared binary templates and signed two-digit operands",
                "ood_policy": "held-out magnitude range or three-operand left-parenthesized structure"}
    return rows, metadata


def generate(split: str, seed: int, count: int) -> list[dict]:
    return generate_with_metadata(split, seed, count)[0]
