"""Signed place-value scheduling; predictions written unchanged. No verifier in inference."""

OPS = ("add", "sub", "mul")
MAX_DIGITS = 40


def normalize(a, b, op):
    if op == "mul":
        return abs(a), abs(b), "mul"
    same = (a < 0) == ((b < 0) != (op == "sub"))
    return (abs(a), abs(b), "add") if same else (max(abs(a), abs(b)), min(abs(a), abs(b)), "sub")


def digits(n):
    return [int(v) for v in reversed(str(abs(n)))]


def sign_features(a, b, op):
    # Observed signs/magnitude ordering/length only; never arithmetic result or sign label.
    return [
        float(a < 0),
        float(b < 0),
        *[float(op == v) for v in OPS],
        float(abs(a) < abs(b)),
        float(abs(a) == abs(b)),
        float(a == 0),
        float(b == 0),
    ]


def run(task, predictor):
    current = task["operands"][0]
    events = []
    stages = []

    def column(op, a, b, incoming, index):
        state = {"operation": op, "left": a, "right": b, "incoming": incoming, "index": index}
        digit, carry, diag = predictor.local(state)
        if type(digit) != int or not 0 <= digit <= 9 or type(carry) != int or not 0 <= carry <= 9:
            raise ValueError("Prediction outside declared vocabulary")
        events.append(
            {"state": state, "written_digit": digit, "written_carry": carry, "prediction": diag}
        )
        if len(events) > 4096:
            raise ValueError("Transition budget")
        return digit, carry

    def arrays(a, b, op):
        predictor.reset()
        result = []
        carry = 0
        for i in range(max(len(a), len(b)) + 1):
            d, carry = column(op, a[i] if i < len(a) else 0, b[i] if i < len(b) else 0, carry, i)
            result.append(d)
        return result

    for op, right in zip(task["operations"], task["operands"][1:]):
        sign, sign_diag = predictor.sign(current, right, op)
        a, b, primitive = normalize(current, right, op)
        if primitive != "mul":
            written = arrays(digits(a), digits(b), primitive)
        else:
            written = [0]
            for j, bd in enumerate(digits(b)):
                predictor.reset()
                row = [0] * j
                carry = 0
                for i, ad in enumerate(digits(a) + [0]):
                    d, carry = column("mul", ad, bd, carry, i)
                    row.append(d)
                written = arrays(written, row, "add")
        if len(written) > MAX_DIGITS:
            raise ValueError("Output bound exceeded")
        text = "".join(str(v) for v in reversed(written)).lstrip("0") or "0"
        current = int(("-" if sign and text != "0" else "") + text)
        stages.append(
            {
                "answer": str(current),
                "sign": int(sign),
                "sign_prediction": sign_diag,
                "written_digits": written,
            }
        )
    return {"answer": str(current), "transitions": events, "stages": stages}
