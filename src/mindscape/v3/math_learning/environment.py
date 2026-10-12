"""Teacher arithmetic and a write-only executor; learner modules cannot import teacher.

Scheduling and signed-magnitude normalization are declared procedural scaffolds,
not learned planning. No rollout consults teacher targets until it has finished.
"""

from dataclasses import asdict, dataclass

from mindscape.v3.schema import digest

OPS = ("add", "sub", "mul")
RAW_OFFSET = 20
MAX_DIGITS = 17


@dataclass(frozen=True)
class Problem:
    operands: tuple[int, ...]
    operations: tuple[str, ...]
    split: str
    family: str

    def __post_init__(self):
        if len(self.operands) != len(self.operations) + 1 or not 1 <= len(self.operations) <= 2:
            raise ValueError("One or two binary operations required")
        if any(op not in OPS for op in self.operations):
            raise ValueError("Unknown operation")
        if any(type(n) is not int or abs(n) >= 10**8 for n in self.operands):
            raise ValueError("Bounded integer operands required")
        if self.split not in ("train", "validation", "iid", "ood", "lockbox", "unit"):
            raise ValueError("Unknown split")

    @property
    def identity(self):
        values = self.operands
        if len(self.operations) == 1 and self.operations[0] in ("add", "mul"):
            values = tuple(sorted(values))
        return digest({"operands": values, "operations": self.operations})

    def visible(self):
        return {"operands": self.operands, "operations": self.operations}


@dataclass(frozen=True)
class LocalState:
    operation: str
    left: int
    right: int
    incoming: int

    def __post_init__(self):
        if self.operation not in OPS or any(
            type(x) is not int or not 0 <= x <= 9 for x in (self.left, self.right, self.incoming)
        ):
            raise ValueError("Invalid local state")

    @property
    def key(self):
        return (self.operation, self.left, self.right, self.incoming)


@dataclass(frozen=True)
class Claim:
    operation: int
    raw: int
    digit: int
    carry: int
    next_digit: int
    next_carry: int

    def __post_init__(self):
        if not 0 <= self.operation < 3 or not -20 <= self.raw <= 99:
            raise ValueError("Invalid operation/raw vocabulary")
        if any(
            type(x) is not int or not 0 <= x <= 9
            for x in (self.digit, self.carry, self.next_digit, self.next_carry)
        ):
            raise ValueError("Invalid digit vocabulary")

    def labels(self):
        return (
            self.operation,
            self.raw + RAW_OFFSET,
            self.digit,
            self.carry,
            self.next_digit,
            self.next_carry,
        )


def teacher_local(state):
    """Training label generator and independent post-rollout verifier only."""
    a, b, c = state.left, state.right, state.incoming
    if state.operation == "add":
        raw = a + b + c
        digit, carry = raw % 10, raw // 10
    elif state.operation == "sub":
        raw = a - b - c
        digit, carry = raw % 10, int(raw < 0)
    else:
        raw = a * b + c
        digit, carry = raw % 10, raw // 10
    return Claim(OPS.index(state.operation), raw, digit, carry, digit, carry)


def expected(problem):
    value = problem.operands[0]
    for op, right in zip(problem.operations, problem.operands[1:]):
        if op == "add":
            value += right
        elif op == "sub":
            value -= right
        else:
            value *= right
    return value


def digits(n):
    return [int(d) for d in reversed(str(abs(n)))]


def materialize(values, sign):
    # Serialization of already-proposed digits, not arithmetic target computation.
    text = "".join(str(x) for x in reversed(values)).lstrip("0") or "0"
    return int(("-" if sign and text != "0" else "") + text)


def normalize(left, right, op):
    """Declared signed magnitude scaffold; never computes a numerical output."""
    if op == "mul":
        return abs(left), abs(right), "mul"
    right_negative = (right < 0) != (op == "sub")
    same_sign = (left < 0) == right_negative
    a, b = abs(left), abs(right)
    if same_sign:
        return a, b, "add"
    return max(a, b), min(a, b), "sub"


def execute_local(state, claim):
    """Write proposed next values exactly. No semantic rejection or correction."""
    return {
        "written_digit": claim.next_digit,
        "incoming": claim.next_carry,
        "proposed_operation": OPS[claim.operation],
        "kind": "actual",
        "event": "unverified_claim_written",
    }


def run_problem(problem, predictor, max_steps=512):
    events = []
    current = problem.operands[0]
    calls = 0

    def local(op, a, b, carry):
        nonlocal calls
        if calls >= max_steps:
            raise RuntimeError("Rollout action budget exhausted")
        state = LocalState(op, a, b, carry)
        claim, diagnostics = predictor.predict_local(state)
        actual = execute_local(state, claim)
        events.append(
            {
                "state": asdict(state),
                "claim": asdict(claim),
                "prediction": {
                    "next_digit": claim.next_digit,
                    "next_carry": claim.next_carry,
                    "kind": "hypothetical",
                },
                "actual": actual,
                "diagnostics": diagnostics,
            }
        )
        calls += 1
        return actual["written_digit"], actual["incoming"]

    def add_arrays(left, right, op):
        output, carry = [], 0
        for i in range(max(len(left), len(right)) + 1):
            a = left[i] if i < len(left) else 0
            b = right[i] if i < len(right) else 0
            digit, carry = local(op, a, b, carry)
            output.append(digit)
        return output

    for stage, (op, right) in enumerate(zip(problem.operations, problem.operands[1:])):
        visible = {"operands": (current, right), "operations": (op,)}
        sign, sign_diag = predictor.predict_sign(visible)
        a, b, primitive = normalize(current, right, op)
        if primitive in ("add", "sub"):
            output = add_arrays(digits(a), digits(b), primitive)
        else:
            accumulator = [0]
            for j, bdigit in enumerate(digits(b)):
                row, carry = [0] * j, 0
                for adigit in digits(a) + [0]:
                    digit, carry = local("mul", adigit, bdigit, carry)
                    row.append(digit)
                accumulator = add_arrays(accumulator, row, "add")
            output = accumulator
        current = materialize(output, sign)
        events.append(
            {
                "stage_result": stage,
                "sign": sign,
                "sign_diagnostics": sign_diag,
                "written_digits": output,
                "value": str(current),
            }
        )
    # Only now consult arithmetic. Verifier annotations are never predictor inputs.
    correct = expected(problem)
    local_events = [e for e in events if "state" in e]
    errors = {
        "operation": 0,
        "raw": 0,
        "digit": 0,
        "carry": 0,
        "next_state": 0,
        "consistency_false_accept": 0,
        "consistency_false_reject": 0,
    }
    for event in local_events:
        target = teacher_local(LocalState(**event["state"]))
        claim = Claim(**event["claim"])
        event["posthoc_expected"] = asdict(target)
        event["posthoc_consistent"] = claim == target
        errors["operation"] += claim.operation != target.operation
        errors["raw"] += claim.raw != target.raw
        errors["digit"] += claim.digit != target.digit
        errors["carry"] += claim.carry != target.carry
        errors["next_state"] += (claim.next_digit, claim.next_carry) != (target.digit, target.carry)
        consistent = event["diagnostics"].get("consistency", 0) >= 0.5
        errors["consistency_false_accept"] += consistent and claim != target
        errors["consistency_false_reject"] += not consistent and claim == target
    return {
        "task_id": problem.identity,
        "problem": asdict(problem),
        "answer": str(current),
        "expected": str(correct),
        "success": current == correct,
        "strict_rollout": current == correct and all(e["posthoc_consistent"] for e in local_events),
        "steps": calls,
        "events": events,
        "errors": errors,
    }


def demonstration(problem):
    if problem.split not in ("train", "unit"):
        raise ValueError("Teacher demonstrations restricted to train partition")

    class Teacher:
        def predict_local(self, state):
            return teacher_local(state), {"consistency": 1.0}

        def predict_sign(self, visible):
            p = Problem(tuple(visible["operands"]), tuple(visible["operations"]), "unit", "teacher")
            return int(expected(p) < 0), {}

    trace = run_problem(problem, Teacher())
    if not trace["strict_rollout"]:
        raise AssertionError("Invalid teacher demonstration")
    return trace
