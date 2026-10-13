"""Versioned public mathematics contract; bounded symbolic tools explicitly distinguished."""

import math
from dataclasses import asdict, dataclass
from fractions import Fraction

from mindscape.v3.schema import digest

from .arithmetic import run
from .verification import verify


@dataclass(frozen=True)
class Request:
    family: str
    inputs: dict
    version: str = "R5.2-math-1"

    def __post_init__(self):
        if self.version != "R5.2-math-1":
            raise ValueError("Version")
        allowed = {
            "integer": {"operands", "operations"},
            "division": {"a", "b"},
            "fractions": {"a", "b", "operation"},
            "algebra": {"coefficients", "x"},
            "equation": {"a", "b", "c"},
            "matrix": {"a", "b"},
            "geometry": {"a", "b"},
            "calculus": {"coefficients", "lower", "upper"},
            "statistics": {"values"},
            "probability": {"successes", "trials"},
        }
        if self.family not in allowed or set(self.inputs) != allowed[self.family]:
            raise ValueError("Unexpected/private input fields")
        digest(self.inputs)
        if len(str(self.inputs)) > 4096:
            raise ValueError("Input size bound")

        def bounded(v, allow_strings=False):
            if isinstance(v, list):
                if len(v) > 16:
                    raise ValueError("List bound")
                for x in v:
                    bounded(x, allow_strings)
            elif type(v) in (int, float):
                if not math.isfinite(v) or abs(v) > 10**20:
                    raise ValueError("Numeric bound")
            elif not (allow_strings and isinstance(v, str)):
                raise ValueError("Unsupported data")

        for name, value in self.inputs.items():
            bounded(value, name in ("operation", "operations"))
        if self.family == "integer":
            x = self.inputs
            if (
                not 1 <= len(x["operations"]) <= 2
                or len(x["operands"]) != len(x["operations"]) + 1
                or any(type(v) != int for v in x["operands"])
                or any(op not in ("add", "sub", "mul") for op in x["operations"])
            ):
                raise ValueError("Arithmetic request")


def solve(request, predictor=None):
    x = request.inputs
    family = request.family
    if family == "integer":
        if predictor is None:
            raise ValueError("Neural predictor required; no silent symbolic fallback")
        from .model import Neural

        if not isinstance(predictor, Neural):
            raise ValueError(
                "Integer capability method requires an actual neural predictor, not a reference/memory control"
            )
        result = run(x, predictor)
        evaluation = verify(x, result)
        return {
            "version": request.version,
            "method": "neural_with_declared_scaffold",
            "family": family,
            "result": result,
            "verification": evaluation,
            "request_hash": digest(asdict(request)),
        }
    import sympy as sp

    trace = []
    verified = True
    if family == "division":
        if type(x["a"]) != int or type(x["b"]) != int or x["b"] == 0:
            raise ValueError("Integer division bounds/zero denominator")
        q, r = divmod(x["a"], abs(x["b"]))
        if x["b"] < 0:
            q = -q
        result = {"quotient": str(q), "remainder": str(r)}
        verified = x["a"] == q * x["b"] + r and 0 <= r < abs(x["b"])
        trace = [{"identity": "a=q*b+r", "checked": verified}]
    elif family == "fractions":
        if len(x["a"]) != 2 or len(x["b"]) != 2:
            raise ValueError("Fraction shape")
        if any(type(v) != int for v in x["a"] + x["b"]) or x["a"][1] == 0 or x["b"][1] == 0:
            raise ValueError("Bounded integer fractions require nonzero denominators")
        a = sp.Rational(*x["a"])
        b = sp.Rational(*x["b"])
        op = x["operation"]
        if op not in ("add", "sub", "mul", "div") or (op == "div" and b == 0):
            raise ValueError("Fraction operation")
        ans = {
            "add": lambda: a + b,
            "sub": lambda: a - b,
            "mul": lambda: a * b,
            "div": lambda: a / b,
        }[op]()
        aa = Fraction(*x["a"])
        bb = Fraction(*x["b"])
        ref = {
            "add": lambda: aa + bb,
            "sub": lambda: aa - bb,
            "mul": lambda: aa * bb,
            "div": lambda: aa / bb,
        }[op]()
        result = str(ans)
        verified = ans == sp.Rational(ref.numerator, ref.denominator)
        trace = [
            {
                "normalized_left": str(a),
                "normalized_right": str(b),
                "operation": op,
                "independent_fraction_result": str(ref),
            }
        ]
    elif family == "algebra":
        coefficients = x["coefficients"]
        v = sp.Rational(x["x"])
        z = sp.Symbol("x")
        poly = sum(sp.Rational(c) * z**i for i, c in enumerate(coefficients))
        ans = poly.subs(z, v)
        result = str(ans)
        verified = ans == sum(
            Fraction(c) * Fraction(x["x"]) ** i for i, c in enumerate(coefficients)
        )
        trace = [{"polynomial": str(poly)}]
    elif family == "equation":
        a, b, c = map(sp.Rational, (x["a"], x["b"], x["c"]))
        if a == 0:
            raise ValueError("Nondegenerate linear equation required")
        ans = (c - b) / a
        result = str(ans)
        verified = sp.simplify(a * ans + b - c) == 0
        trace = [{"equation": "a*x+b=c", "residual": str(a * ans + b - c)}]
    elif family == "matrix":
        a = sp.Matrix(x["a"])
        b = sp.Matrix(x["b"])
        if max(a.rows, a.cols, b.rows, b.cols) > 4 or a.cols != b.rows:
            raise ValueError("Matrix dimensions")
        ans = a * b
        result = [[str(v) for v in row] for row in ans.tolist()]
        ref = [
            [
                sum(Fraction(x["a"][i][k]) * Fraction(x["b"][k][j]) for k in range(a.cols))
                for j in range(b.cols)
            ]
            for i in range(a.rows)
        ]
        verified = all(
            ans[i, j] == sp.Rational(ref[i][j].numerator, ref[i][j].denominator)
            for i in range(a.rows)
            for j in range(b.cols)
        )
    elif family == "geometry":
        if len(x["a"]) != 2 or len(x["b"]) != 2:
            raise ValueError("2D points only")
        delta = [sp.Rational(a) - sp.Rational(b) for a, b in zip(x["a"], x["b"])]
        squared = sum(v * v for v in delta)
        result = {"distance_squared": str(squared), "distance": str(sp.sqrt(squared))}
        trace = [
            {
                "coordinate_differences": [str(v) for v in delta],
                "squared_terms": [str(v * v) for v in delta],
            }
        ]
        verified = squared == sum((Fraction(a) - Fraction(b)) ** 2 for a, b in zip(x["a"], x["b"]))
    elif family == "calculus":
        z = sp.Symbol("x")
        co = x["coefficients"]
        poly = sum(sp.Rational(c) * z**i for i, c in enumerate(co))
        derivative = sp.diff(poly, z)
        lo, hi = sp.Rational(x["lower"]), sp.Rational(x["upper"])
        integral = sp.integrate(poly, (z, lo, hi))
        result = {"derivative": str(derivative), "definite_integral": str(integral)}
        ref = sum(
            (Fraction(c) / (i + 1))
            * (Fraction(x["upper"]) ** (i + 1) - Fraction(x["lower"]) ** (i + 1))
            for i, c in enumerate(co)
        )
        verified = sp.expand(
            derivative - sum(i * sp.Rational(c) * z ** (i - 1) for i, c in enumerate(co) if i)
        ) == 0 and integral == sp.Rational(ref.numerator, ref.denominator)
        trace = [
            {
                "polynomial": str(poly),
                "derivative": str(derivative),
                "antiderivative": str(sp.integrate(poly, z)),
                "lower": str(lo),
                "upper": str(hi),
                "independent_integral": str(ref),
            }
        ]
    elif family == "statistics":
        values = [Fraction(v) for v in x["values"]]
        if not values:
            raise ValueError("Nonempty bounded sample")
        avg = sum(values) / len(values)
        variance = sum((v - avg) ** 2 for v in values) / len(values)
        result = {
            "mean": str(sp.Rational(avg.numerator, avg.denominator)),
            "population_variance": str(sp.Rational(variance.numerator, variance.denominator)),
        }
        verified = (
            variance == sum(v * v for v in values) / len(values) - avg * avg and variance >= 0
        )
    else:
        s, n = x["successes"], x["trials"]
        if type(s) != int or type(n) != int or not 0 <= s <= n or n <= 0:
            raise ValueError("Binomial event-count bounds")
        p = sp.Rational(s, n)
        result = str(p)
        trace = [{"event_count": s, "total_count": n, "normalized_ratio": str(p)}]
        verified = 0 <= p <= 1 and p == sp.Rational(
            Fraction(s, n).numerator, Fraction(s, n).denominator
        )
    if not trace:
        trace = [{"operation": family, "independent_check": bool(verified)}]
    return {
        "version": request.version,
        "method": "symbolic_tool",
        "tool": "Python Euclidean integer division"
        if family == "division"
        else "SymPy+independent arithmetic checks",
        "family": family,
        "result": result,
        "trace": trace,
        "verification": {"success": bool(verified)},
        "request_hash": digest(asdict(request)),
    }
