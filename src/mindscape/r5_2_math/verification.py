"""Independent exact arithmetic after inference; training targets live here, not model code."""

from .arithmetic import run


def local_target(state):
    a, b, c = state["left"], state["right"], state["incoming"]
    op = state["operation"]
    raw = a + b + c if op == "add" else a - b - c if op == "sub" else a * b + c
    return raw % 10, int(raw < 0) if op == "sub" else raw // 10


def exact(task):
    value = task["operands"][0]
    for op, b in zip(task["operations"], task["operands"][1:]):
        value = value + b if op == "add" else value - b if op == "sub" else value * b
    return value


def verify(task, result):
    expected = str(exact(task))
    correct = []
    seen = []
    for event in result["transitions"]:
        d, c = local_target(event["state"])
        good = (d, c) == (event["written_digit"], event["written_carry"])
        correct.append(good)
        seen.append(
            {
                "key": context_key(event["state"]),
                "correct": good,
                "expected_digit": d,
                "expected_carry": c,
            }
        )
    stage_exact = []
    sign_correct = []
    rolled = task["operands"][0]
    prefix = task["operands"][0]
    for op, b, stage in zip(
        task["operations"], task["operands"][1:], result["stages"], strict=True
    ):
        sign_correct.append(
            stage["sign"] == int(exact({"operands": [rolled, b], "operations": [op]}) < 0)
        )
        prefix = exact({"operands": [prefix, b], "operations": [op]})
        stage_exact.append(stage["answer"] == str(prefix))
        rolled = int(stage["answer"])
    return {
        "exact": result["answer"] == expected,
        "expected": expected,
        "local_correct": sum(correct),
        "local_total": len(correct),
        "strict": result["answer"] == expected
        and all(correct)
        and all(stage_exact)
        and all(sign_correct),
        "stage_correct": sum(stage_exact),
        "stage_total": len(stage_exact),
        "sign_correct": sum(sign_correct),
        "sign_total": len(sign_correct),
        "posthoc_transitions": seen,
    }


def context_key(state):
    return "|".join(
        map(str, (state["operation"], state["left"], state["right"], state["incoming"]))
    )


class Teacher:
    def reset(self):
        pass

    def local(self, state):
        d, c = local_target(state)
        return d, c, {"kind": "training_or_reference_teacher"}

    def sign(self, a, b, op):
        return int(exact({"operands": [a, b], "operations": [op]}) < 0), {
            "kind": "training_or_reference_teacher"
        }


def demonstration(task):
    if task["split"] != "train":
        raise ValueError("Training demonstrations only")
    result = run(task, Teacher())
    if not verify(task, result)["strict"]:
        raise ValueError("Scaffold/reference mismatch")
    return result
