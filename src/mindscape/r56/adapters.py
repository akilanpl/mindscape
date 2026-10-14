"""Three deliberately narrow examples with separate domain semantics."""
import ast
import math
import keyword
from .core import Concept, Registry, State


def _rename(source: str, old: str, new: str) -> str:
    tree = ast.parse(source)
    class Rename(ast.NodeTransformer):
        def visit_Name(self, node):
            return ast.copy_location(ast.Name(id=new if node.id == old else node.id, ctx=node.ctx), node)
    return ast.unparse(Rename().visit(tree))


def _coding_pre(s, p):
    if (len(s['source']) > 4096 or not p['old'].isidentifier() or not p['new'].isidentifier()
            or keyword.iskeyword(p['old']) or keyword.iskeyword(p['new']) or p['old'] == p['new']):
        return False
    try:
        tree = ast.parse(s['source'])
    except (SyntaxError, ValueError, RecursionError):
        return False
    # Restrict to a single straight-line assignment: no binding/capture claims for general Python.
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assign):
        return False
    return all(isinstance(n, (ast.Module, ast.Assign, ast.Name, ast.Load, ast.Store,
                             ast.Constant, ast.BinOp, ast.Add, ast.Sub, ast.Mult))
               for n in ast.walk(tree)) and not any(isinstance(n, ast.Name) and n.id == p['new'] for n in ast.walk(tree))


def _coding_post(s, p, out):
    if len(out['source']) > 4096:
        return False
    try:
        before, after = ast.parse(s['source']), ast.parse(out['source'])
    except (SyntaxError, ValueError, RecursionError):
        return False
    # Verify by inverse alpha rename and full AST equality, independent of forward transformer.
    for node in ast.walk(after):
        if isinstance(node, ast.Name) and node.id == p['new']:
            node.id = p['old']
    return ast.dump(before) == ast.dump(after) and all(
        not isinstance(n, ast.Name) or n.id != p['old'] for n in ast.walk(ast.parse(out['source'])))


def default_registry() -> Registry:
    registry = Registry()
    registry.register(Concept(
        'math.carry', 'math', {'total': int, 'digit': int, 'carry': int}, {'base': int},
        lambda s, p: 2 <= p['base'] <= 36 and 0 <= s['total'] <= 1000000 and s['digit'] == 0 and s['carry'] == 0,
        lambda s, p: State.create('math', total=s['total'], digit=s['total'] % p['base'], carry=s['total'] // p['base']),
        lambda s, p, o: o['total'] == s['total'] and 0 <= o['digit'] < p['base'] and o['carry'] >= 0 and o['carry'] * p['base'] + o['digit'] == s['total'],
        ('bounded decomposition', 'quantity preservation')))
    registry.register(Concept(
        'coding.rename_assignment', 'coding', {'source': str}, {'old': str, 'new': str},
        _coding_pre, lambda s, p: State.create('coding', source=_rename(s['source'], p['old'], p['new'])),
        _coding_post, ('structure preservation',)))
    registry.register(Concept(
        'physics.impulse', 'physics', {'mass': float, 'velocity': float}, {'impulse': float},
        lambda s, p: all(math.isfinite(x) for x in (*s.values(), *p.values())) and 0 < s['mass'] <= 1e6 and abs(s['velocity']) <= 1e6 and abs(p['impulse']) <= 1e6,
        lambda s, p: State.create('physics', mass=s['mass'], velocity=s['velocity'] + p['impulse'] / s['mass']),
        lambda s, p, o: o['mass'] == s['mass'] and math.isfinite(o['velocity']) and math.isclose((o['velocity'] - s['velocity']) * s['mass'], p['impulse'], rel_tol=1e-9, abs_tol=1e-8),
        ('quantity update',)))
    return registry
