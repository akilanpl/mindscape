"""Stage A: bounded deterministic interpretation, never answer generation.

Every accepted request becomes typed operands or an explicitly named tool call.
This grammar is not a trained language model and intentionally abstains widely.
"""
from __future__ import annotations
import ast
import math
import re

_NUMBER = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)'
_VECTOR = rf'(?:\(\s*({_NUMBER})\s*,\s*({_NUMBER})\s*\)|\[\s*({_NUMBER})\s*,\s*({_NUMBER})\s*\]|({_NUMBER}))'
_LIMITS = ['Deterministic bounded grammar; no general language understanding or learned generalization.']


def _result(kind, problem=None, limitations=(), tool=None):
    out = {'supported': kind != 'unsupported', 'kind': kind, 'problem': problem or {},
           'interpretation': 'deterministic-grammar', 'limitations': list(_LIMITS) + list(limitations)}
    if tool is not None: out['tool'] = tool
    return out


def _tool(name, args, domain, operation):
    return _result('tool', {'domain': domain, 'operation': operation},
                   ['External tool assistance; not native neural reasoning.'], {'name': name, 'args': args})


def _rename(message):
    match = re.fullmatch(r'\s*(?:please\s+)?rename\s+(?:the\s+)?(?:variable\s+)?([A-Za-z_]\w*)\s+to\s+([A-Za-z_]\w*)\s*(?:in\s+(?:this|the)\s+(?:function|code)\s*)?[:.]?\s*```(?:python|py)?\s*\n([\s\S]+?)```\s*', message, re.I)
    if not match: return None
    old, target, source = match.groups()
    if len(source) > 4000 or len(old) > 32 or len(target) > 32: return _result('unsupported')
    try: tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError): return _result('unsupported')
    allowed = (ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return, ast.Assign,
               ast.Name, ast.Load, ast.Store, ast.Constant, ast.BinOp, ast.UnaryOp,
               ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.UAdd)
    nodes = list(ast.walk(tree))
    if len(nodes) > 120 or len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef): return _result('unsupported')
    function = tree.body[0]
    if function.decorator_list or function.returns or any(isinstance(n, ast.FunctionDef) and n is not function for n in nodes) or any(a.annotation for a in function.args.args + function.args.kwonlyargs + function.args.posonlyargs) or function.args.defaults or function.args.kw_defaults or any(not isinstance(n, allowed) for n in nodes): return _result('unsupported')
    if any(isinstance(n, ast.Constant) and (type(n.value) not in (int, float) or (isinstance(n.value, int) and n.value.bit_length() > 128) or (isinstance(n.value, float) and not math.isfinite(n.value))) for n in nodes): return _result('unsupported')
    names = {n.id for n in nodes if isinstance(n, ast.Name)} | {n.arg for n in nodes if isinstance(n, ast.arg)}
    import keyword
    if old not in names or target in names or target == function.name or keyword.iskeyword(target): return _result('unsupported', limitations=['Rename must reference an existing name and avoid capture.'])
    candidates = sorted(names | {target, 'value', 'result', 'temp', 'item', 'a', 'b', 'x', 'y'})
    if len(candidates) > 16: return _result('unsupported')
    return _result('native', {'domain': 'coding', 'operation': 'rename', 'source': source,
                              'old_name': old, 'requested_name': target, 'candidates': candidates, 'target_name': target},
                   ['Single pure Python function only; no calls, imports, attributes, nested control flow or execution.',
                    'target_name is verifier-only and must be excluded from neural candidate scoring inputs.'])


def _physics(message):
    text = message.strip()
    if not re.search(r'\b(?:velocity|force|mass)\b', text, re.I): return None
    if re.search(r'\b(?:drag|friction|relativistic|variable force|acceleration|collision|not|without|pounds?|lbs?|grams?|mph|kilometers?|minutes?|hours?)\b', text, re.I): return _result('unsupported')
    mass = list(re.finditer(rf'\bmass\s*(?:of|is|=|:)??\s*({_NUMBER})\s*(?:kg)?', text, re.I))
    force = list(re.finditer(rf'\bforce\s*(?:of|is|=|:)??\s*{_VECTOR}\s*(?:N\b|newtons?\b)?', text, re.I))
    velocity = list(re.finditer(rf'\b(?:initial\s+velocity|velocity|v0)\s*(?:of|is|=|:)??\s*{_VECTOR}\s*(?:m/s)?', text, re.I))
    duration = list(re.finditer(rf'\b(?:time|dt|duration|for)\s*(?:of|is|=|:)??\s*({_NUMBER})\s*(?:s\b|seconds?\b)?', text, re.I))
    if not all(len(x) == 1 for x in (mass, force, velocity, duration)): return _result('unsupported', limitations=['Specify mass (kg), force vector (N), initial velocity vector (m/s), and duration (s) exactly once.'])
    def vector(match):
        values = [float(x) for x in match.groups() if x is not None]
        return values if len(values) == 2 else [values[0], 0.0]
    spans = [m.span() for group in (mass, force, velocity, duration) for m in group]
    remainder = ''.join(' ' if any(a <= i < b for a,b in spans) else ch for i,ch in enumerate(text))
    if re.search(_NUMBER, remainder): return _result('unsupported', limitations=['Unrecognized numeric constraint.'])
    m, dt, v, f = float(mass[0][1]), float(duration[0][1]), vector(velocity[0]), vector(force[0])
    if not all(math.isfinite(x) and abs(x) <= 1e6 for x in [m, dt, *v, *f]) or m <= 0 or dt < 0: return _result('unsupported')
    return _result('native', {'domain': 'physics', 'operation': 'force', 'mass': m,
                              'velocity': v, 'force': f, 'dt': dt},
                   ['Constant force, constant positive mass, Newtonian point particle, SI units; scalar inputs lie on x axis.'])


def interpret(message, history=None):
    """Return typed interpretation only. History is not concatenated or inferred from."""
    if not isinstance(message, str) or not 1 <= len(message) <= 16000:
        return _result('unsupported', limitations=['Message must contain 1–16000 characters.'])
    if history is not None and (not isinstance(history, list) or len(history) > 40):
        return _result('unsupported', limitations=['Invalid or oversized history.'])
    text = message.strip()
    # Resolve only standalone requests; pronouns and context-dependent tasks abstain.
    arithmetic = re.fullmatch(r'(?:please\s+)?(?:(?:what is|calculate|compute|evaluate)\s+)?(\d{1,18})\s*(\+|plus|added to)\s*(\d{1,18})\s*\??', text, re.I)
    alternate = re.fullmatch(r'(?:please\s+)?add\s+(\d{1,18})\s+(?:and|to)\s+(\d{1,18})\s*[?.]?', text, re.I)
    if arithmetic or alternate:
        left, right = (arithmetic[1], arithmetic[3]) if arithmetic else alternate.groups()
        return _result('native', {'domain': 'math', 'operation': 'add', 'left': left, 'right': right}, ['Nonnegative base-10 integers, at most 18 digits per operand.'])
    subtraction = re.fullmatch(r'(?:(?:what is|calculate|compute)\s+)?([0-9])\s*(?:-|minus)\s*([0-9])\s*\??', text, re.I)
    if subtraction:
        return _result('native', {'domain': 'math', 'operation': 'borrow_digit', 'left': int(subtraction[1]), 'right': int(subtraction[2]), 'borrow_in': 0}, ['Single base-10 digit subtraction only.'])
    renamed = _rename(text)
    if renamed is not None: return renamed
    calculus = re.fullmatch(r'(differentiate|integrate|simplify|solve)\s+(.+?)(?:\s+(?:with respect to|for)\s+([xyzt]))?\s*\??', text, re.I)
    if calculus:
        operation, expression, variable = calculus.groups()
        if len(expression) > 512: return _result('unsupported')
        if operation.lower() == 'solve' and '=' in expression:
            left, right = expression.split('=', 1); expression = f'({left})-({right})'
        return _tool('math_calculate', {'expression': expression, 'operation': operation.lower(), 'variable': variable or 'x'}, 'math', operation.lower())
    code = re.fullmatch(r'(inspect|analyze|analyse|check|execute|run)\s*(?:this\s+)?(?:code\s*)?[:.]?\s*```([A-Za-z+#]*)\n([\s\S]+?)```\s*', text, re.I)
    if code:
        action, language, source = code.groups(); language = {'py':'python','js':'javascript','ts':'typescript','c++':'cpp'}.get(language.lower(),language.lower() or 'python')
        if action.lower() in {'execute','run'}:
            if language != 'python': return _result('unsupported', limitations=['Execution supports configured WASI Python only.'])
            return _tool('coding_execute', {'source': source}, 'coding', 'execute')
        return _tool('coding_analyze', {'source': source, 'language': language}, 'coding', 'inspect')
    # Explicit JSON simulation data is parsed as data, not as code or inferred values.
    simulation = re.fullmatch(r'simulate\s+(projectile|oscillator|rc_circuit)\s+(\{[\s\S]*\})', text, re.I)
    if simulation:
        import json
        try: params = json.loads(simulation[2], parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
        except (ValueError, RecursionError): return _result('unsupported')
        if not isinstance(params, dict): return _result('unsupported')
        return _tool('physics_simulate', {'model': simulation[1].lower(), 'parameters': params}, 'physics', 'simulate')
    physics = _physics(text)
    if physics is not None: return physics
    return _result('unsupported', limitations=['No supported grammar matched; no answer was generated.'])
