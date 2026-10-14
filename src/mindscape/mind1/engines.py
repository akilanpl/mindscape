"""Bounded tools. Source text is data; guest execution is WASI-only.

These tools provide explicit external assistance, not evidence of learned reasoning.
"""
from __future__ import annotations

import ast
import importlib.util
import math
import os
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any


class Unavailable(RuntimeError):
    """An optional runtime is absent."""


def capabilities() -> dict:
    runtime = Path(os.environ.get('MIND1_WASI_RUNTIME', os.environ.get('MINDSCAPE_WASI_RUNTIME', 'work/coding/runtime')))
    return {'symbolic_math': importlib.util.find_spec('sympy') is not None,
            'numerical_physics': True, 'static_analysis': ['python', 'javascript', 'typescript', 'rust', 'go', 'c', 'cpp'],
            'wasi_python': importlib.util.find_spec('wasmtime') is not None and (runtime / 'python.wasm').is_file(),
            'host_source_execution': False, 'research_scope': 'external tools, not learned capability'}


def _sympy():
    try:
        import sympy
        return sympy
    except ImportError as exc:
        raise Unavailable('Install sympy for symbolic mathematics') from exc


def _expression(source: str):
    """Build symbolic objects from an allowlisted AST, never eval/sympify text."""
    s = _sympy()
    if not isinstance(source, str) or len(source) > 512:
        raise ValueError('Expression must be at most 512 characters')
    tree = ast.parse(source, mode='eval')
    if sum(1 for _ in ast.walk(tree)) > 100:
        raise ValueError('Expression exceeds 100 AST nodes')
    functions = {name: getattr(s, name) for name in ('sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'Abs')}
    symbols = {name: s.Symbol(name, real=True) for name in ('x', 'y', 'z', 't')}
    def visit(node, depth=0):
        if depth > 16:
            raise ValueError('Expression nesting exceeds 16')
        sub = lambda n: visit(n, depth + 1)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            if not math.isfinite(node.value) or abs(node.value) > 1e12:
                raise ValueError('Literal out of bounds')
            return s.Rational(str(node.value))
        if isinstance(node, ast.Name) and node.id in symbols:
            return symbols[node.id]
        if isinstance(node, ast.Name) and node.id in ('pi', 'E'):
            return getattr(s, node.id)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            return -sub(node.operand) if isinstance(node.op, ast.USub) else sub(node.operand)
        if isinstance(node, ast.BinOp):
            a, b = sub(node.left), sub(node.right)
            if isinstance(node.op, ast.Add): return a + b
            if isinstance(node.op, ast.Sub): return a - b
            if isinstance(node.op, ast.Mult): return a * b
            if isinstance(node.op, ast.Div):
                if b == 0: raise ValueError('Division by zero')
                return a / b
            if isinstance(node.op, ast.Pow):
                if not isinstance(node.right, ast.Constant) or type(node.right.value) not in (int, float) or abs(node.right.value) > 12:
                    raise ValueError('Exponent must be a literal between -12 and 12')
                if a.is_Rational and (int(a.p).bit_length() + int(a.q).bit_length()) * max(1, abs(float(b))) > 4096:
                    raise ValueError('Numeric power exceeds resource budget')
                return a ** b
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in functions and len(node.args) == 1 and not node.keywords:
            return functions[node.func.id](sub(node.args[0]))
        raise ValueError('Unsupported expression syntax')
    return visit(tree.body)


def _variable(name):
    if name not in ('x', 'y', 'z', 't'): raise ValueError('Variable must be x, y, z or t')
    return _sympy().Symbol(name, real=True)


def _math(args):
    s = _sympy()
    expression = _expression(args['expression'])
    operation = args.get('operation', 'evaluate')
    variable = _variable(args.get('variable', 'x'))
    if operation == 'evaluate':
        values = args.get('values', {})
        if not isinstance(values, dict) or len(values) > 4: raise ValueError('At most four substitutions')
        expression = expression.subs({_variable(k): _number(v) for k, v in values.items()})
        result = s.N(expression, 16) if not expression.free_symbols else expression
    elif operation == 'differentiate': result = s.diff(expression, variable)
    elif operation == 'integrate':
        # Polynomial integrals have predictable resource costs; other integrands fail explicitly.
        if not expression.is_polynomial(variable) or s.degree(expression, variable) > 12:
            raise ValueError('Integration supports polynomials of degree at most 12')
        result = s.integrate(expression, variable)
    elif operation == 'solve':
        if expression.free_symbols - {variable} or not expression.is_polynomial(variable):
            raise ValueError('Solve requires a univariate polynomial')
        if s.degree(expression, variable) > 4: raise ValueError('Solve supports degree at most four')
        return {'solutions': [str(v) for v in s.solve(expression, variable)], 'variable': str(variable)}
    elif operation == 'simplify': result = s.cancel(expression) if expression.is_rational_function() else expression
    else: raise ValueError('Unknown mathematics operation')
    if result.has(s.zoo, s.oo, -s.oo, s.nan): raise ValueError('Result is not finite')
    return {'result': str(result), 'operation': operation}


def _number(value):
    if type(value) not in (int, float) or not math.isfinite(value) or abs(value) > 1e12:
        raise ValueError('Expected a finite number with magnitude at most 1e12')
    return float(value)


def _matrix(args):
    s = _sympy()
    rows = args['matrix']
    if not isinstance(rows, list) or not 1 <= len(rows) <= 6 or any(not isinstance(r, list) for r in rows):
        raise ValueError('Matrix must contain 1 to 6 rows')
    width = len(rows[0])
    if not 1 <= width <= 6 or any(len(r) != width for r in rows): raise ValueError('Matrix must be rectangular with 1 to 6 columns')
    matrix = s.Matrix([[s.Rational(str(_number(v))) for v in row] for row in rows])
    op = args.get('operation', 'determinant')
    if op in ('determinant', 'inverse') and matrix.rows != matrix.cols: raise ValueError('Square matrix required')
    if op == 'determinant': return {'result': str(matrix.det())}
    if op == 'rank': return {'result': matrix.rank()}
    if op == 'inverse': return {'result': [[str(v) for v in row] for row in matrix.inv().tolist()]}
    raise ValueError('Unknown matrix operation')


def _physics(args):
    model = args['model']
    p = args.get('parameters', {})
    def n(key, default=None):
        if key not in p and default is None: raise ValueError('Missing parameter: ' + key)
        return _number(p.get(key, default))
    duration = n('duration', 10)
    steps = args.get('steps', 100)
    if type(steps) is not int or not 1 <= steps <= 1000 or not 0 < duration <= 10000: raise ValueError('Require 1..1000 steps and duration in (0,10000]')
    samples = []
    if model == 'projectile':
        speed, angle, gravity, height = n('speed'), math.radians(n('angle_degrees', 45)), n('gravity', 9.80665), n('height', 0)
        if speed < 0 or gravity <= 0 or height < 0: raise ValueError('Speed/height must be nonnegative and gravity positive')
        vx, vy = speed * math.cos(angle), speed * math.sin(angle)
        impact = (vy + math.sqrt(vy * vy + 2 * gravity * height)) / gravity
        duration = min(duration, impact)
        for i in range(steps + 1):
            t = duration * i / steps
            samples.append({'t': t, 'x': vx * t, 'y': max(0., height + vy * t - gravity * t * t / 2)})
        return {'model': model, 'samples': samples, 'impact_time': impact, 'range': vx * impact, 'method': 'analytic constant gravity; no drag', 'units': 'SI; angle in degrees'}
    if model == 'oscillator':
        mass, spring, damping = n('mass', 1), n('spring', 1), n('damping', 0)
        x, v = n('position', 1), n('velocity', 0)
        if mass <= 0 or spring <= 0 or damping < 0: raise ValueError('Mass/spring positive; damping nonnegative')
        dt = duration / steps
        if dt * math.sqrt(spring / mass) > .2 or dt * damping / mass > .2: raise ValueError('Increase steps or reduce duration for numerical stability')
        def f(x, v): return v, -(spring * x + damping * v) / mass
        for i in range(steps + 1):
            samples.append({'t': i * dt, 'position': x, 'velocity': v, 'energy': (mass * v*v + spring*x*x)/2})
            a,b = f(x,v); c,d = f(x+dt*a/2,v+dt*b/2); e,g = f(x+dt*c/2,v+dt*d/2); h,j = f(x+dt*e,v+dt*g)
            x += dt*(a+2*c+2*e+h)/6; v += dt*(b+2*d+2*g+j)/6
        return {'model': model, 'samples': samples, 'method': 'fixed-step RK4', 'units': 'SI'}
    if model == 'rc_circuit':
        resistance, capacitance, voltage, initial = n('resistance'), n('capacitance'), n('voltage'), n('initial_voltage', 0)
        if resistance <= 0 or capacitance <= 0: raise ValueError('Resistance/capacitance must be positive')
        tau = resistance * capacitance
        for i in range(steps + 1):
            t = i * duration / steps
            samples.append({'t': t, 'voltage': voltage + (initial - voltage) * math.exp(-t/tau)})
        return {'model': model, 'samples': samples, 'time_constant': tau, 'method': 'analytic ideal RC charging', 'units': 'SI'}
    raise ValueError('Unknown physics model')


def _code(args):
    source, language = args['source'], args.get('language', 'python').lower()
    if not isinstance(source, str) or len(source.encode()) > 64000: raise ValueError('Source exceeds 64KB')
    if language not in capabilities()['static_analysis']: raise ValueError('Unsupported language')
    findings = []
    if language == 'python':
        try: tree = ast.parse(source)
        except SyntaxError as exc: return {'language': language, 'syntax_valid': False, 'findings': [{'line': exc.lineno, 'message': exc.msg}], 'executed': False}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = ast.unparse(node.func)
                if name in ('eval', 'exec', 'os.system', 'subprocess.run', 'subprocess.Popen', 'pickle.loads'):
                    findings.append({'line': node.lineno, 'message': 'Review dynamic execution/deserialization: ' + name})
        valid = True
    else:
        valid = None  # lexical checks must never pretend to be a compiler
        patterns = {'javascript': r'\beval\s*\(|\binnerHTML\s*=', 'typescript': r'\beval\s*\(|\binnerHTML\s*=', 'rust': r'\bunsafe\b', 'go': r'\bexec\.Command\s*\(', 'c': r'\b(gets|strcpy|sprintf|system)\s*\(', 'cpp': r'\b(gets|strcpy|sprintf|system)\s*\('}
        for match in re.finditer(patterns[language], source):
            findings.append({'line': source.count('\n', 0, match.start()) + 1, 'message': 'Review potentially unsafe construct: ' + match.group(0)})
    return {'language': language, 'syntax_valid': valid, 'findings': findings[:100], 'executed': False, 'limitations': 'Heuristic findings are not a security proof; non-Python syntax is not compiled.'}


def _execute(args):
    if not capabilities()['wasi_python']: raise Unavailable('WASI Python runtime unavailable; host execution is forbidden')
    source = args['source']
    if not isinstance(source, str) or len(source.encode()) > 64000: raise ValueError('Source exceeds 64KB')
    from mindscape.coding.sandbox import WasiSandbox
    sandbox = WasiSandbox(os.environ.get('MIND1_WASI_RUNTIME', os.environ.get('MINDSCAPE_WASI_RUNTIME', 'work/coding/runtime')), timeout=5)
    sandbox.probe()
    return {'isolation': 'WASI', **asdict(sandbox.execute({}, source))}


_HANDLERS = {'math_calculate': _math, 'math_matrix': _matrix, 'physics_simulate': _physics, 'coding_analyze': _code, 'coding_execute': _execute}


def run_tool(name: str, args: dict) -> dict:
    """Return JSON-compatible results or a structured, fail-closed error."""
    try:
        if name not in _HANDLERS: raise ValueError('Unknown tool: ' + str(name))
        if not isinstance(args, dict): raise ValueError('Tool arguments must be an object')
        return {'ok': True, 'tool': name, 'result': _HANDLERS[name](args)}
    except Unavailable as exc:
        return {'ok': False, 'tool': name, 'error': {'code': 'unavailable', 'message': str(exc)}}
    except (ValueError, TypeError, KeyError, SyntaxError, ArithmeticError, RuntimeError, NotImplementedError) as exc:
        return {'ok': False, 'tool': name, 'error': {'code': 'invalid_request', 'message': str(exc)[:500]}}


def tool_specs() -> list:
    definitions = [
        ('math_calculate', 'Bounded symbolic evaluation, differentiation, polynomial integration and solving (expression = 0).', {'expression': {'type': 'string'}, 'operation': {'type': 'string', 'enum': ['evaluate', 'differentiate', 'integrate', 'solve', 'simplify']}, 'variable': {'type': 'string', 'enum': ['x', 'y', 'z', 't']}, 'values': {'type': 'object', 'additionalProperties': {'type': 'number'}}}, ['expression']),
        ('math_matrix', 'Determinant, inverse or rank for matrices up to 6 by 6.', {'matrix': {'type': 'array', 'items': {'type': 'array', 'items': {'type': 'number'}}}, 'operation': {'type': 'string', 'enum': ['determinant', 'inverse', 'rank']}}, ['matrix']),
        ('physics_simulate', 'SI-unit projectile, oscillator and RC circuit models with stated assumptions.', {'model': {'type': 'string', 'enum': ['projectile', 'oscillator', 'rc_circuit']}, 'parameters': {'type': 'object', 'additionalProperties': {'type': 'number'}}, 'steps': {'type': 'integer', 'minimum': 1, 'maximum': 1000}}, ['model']),
        ('coding_analyze', 'Parse Python or inspect lexical risk patterns in seven languages without executing code.', {'source': {'type': 'string'}, 'language': {'type': 'string', 'enum': capabilities()['static_analysis']}}, ['source']),
        ('coding_execute', 'Execute Python only inside the configured capability-isolated WASI runtime; fails closed when absent.', {'source': {'type': 'string'}}, ['source']),
    ]
    return [{'type': 'function', 'function': {'name': name, 'description': desc, 'parameters': {'type': 'object', 'properties': props, 'required': required, 'additionalProperties': False}}} for name, desc, props, required in definitions]
