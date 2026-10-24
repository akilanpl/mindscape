"""Original bounded Python tracing tasks; no eval/exec or arbitrary code execution.

Answers must agree between a restricted AST interpreter and an independent
closed-form/template oracle. This module neither imports nor invokes a model.
"""
import ast
import copy
import hashlib
import random

SPLITS = ('train', 'dev', 'final_iid', 'final_ood')
IID_TEMPLATES = ('affine', 'for_sum', 'while_sum', 'conditional')
OOD_TEMPLATES = ('nested_loops', 'nested_function', 'branch_loop')
GENERATOR_VERSION = 'r6-v1'


def canonicalize(program):
    tree = ast.parse(program)
    names = {}
    class Rename(ast.NodeTransformer):
        def name(self, name):
            if name == 'range':
                return name
            if name not in names:
                names[name] = 'v' + str(len(names))
            return names[name]
        def visit_Name(self, node):
            node.id = self.name(node.id)
            return node
        def visit_arg(self, node):
            node.arg = self.name(node.arg)
            return node
        def visit_FunctionDef(self, node):
            node.name = self.name(node.name)
            return self.generic_visit(node)
    return ast.dump(Rename().visit(tree), include_attributes=False)


class _Return(Exception):
    def __init__(self, value):
        self.value = value


def interpret(program):
    """Interpret a deliberately small AST whitelist with a global step budget.

    No attributes, imports, files, comprehensions, arbitrary callable values,
    negative powers, or unrestricted Python execution are supported.
    """
    tree = ast.parse(program)
    budget = [10000]
    def tick():
        budget[0] -= 1
        if budget[0] < 0:
            raise ValueError('program step limit')
    def expr(node, env):
        tick()
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value
        if isinstance(node, ast.Name):
            return env[node.id]
        if isinstance(node, ast.BinOp):
            a, b = expr(node.left, env), expr(node.right, env)
            ops = {ast.Add: lambda: a+b, ast.Sub: lambda: a-b,
                   ast.Mult: lambda: a*b, ast.Mod: lambda: a%b,
                   ast.FloorDiv: lambda: a//b}
            if type(node.op) in ops:
                value = ops[type(node.op)]()
                if abs(value) > 10000000:
                    raise ValueError('integer bound')
                return value
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            a, b = expr(node.left, env), expr(node.comparators[0], env)
            ops = {ast.Lt: a < b, ast.Gt: a > b, ast.Eq: a == b}
            if type(node.ops[0]) in ops:
                return ops[type(node.ops[0])]
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            args = [expr(x, env) for x in node.args]
            if node.func.id == 'range' and 1 <= len(args) <= 3:
                result = range(*args)
                if len(result) > 100:
                    raise ValueError('range bound')
                return result
            fn, closure = env[node.func.id]
            if len(args) != len(fn.args.args):
                raise ValueError('arity')
            local = dict(closure)
            local.update({p.arg: value for p, value in zip(fn.args.args, args)})
            try:
                block(fn.body, local)
            except _Return as result:
                return result.value
            raise ValueError('missing return')
        raise ValueError('unsupported expression: ' + type(node).__name__)
    def block(nodes, env):
        last = None
        for node in nodes:
            tick()
            if isinstance(node, ast.FunctionDef):
                if node.decorator_list or node.args.defaults or node.args.vararg or node.args.kwarg or node.args.kwonlyargs:
                    raise ValueError('unsupported function')
                env[node.name] = (node, env)
            elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                last = expr(node.value, env)
                env[node.targets[0].id] = last
            elif isinstance(node, ast.Return):
                raise _Return(expr(node.value, env))
            elif isinstance(node, ast.If):
                block(node.body if expr(node.test, env) else node.orelse, env)
            elif isinstance(node, ast.For) and isinstance(node.target, ast.Name) and not node.orelse:
                for item in expr(node.iter, env):
                    env[node.target.id] = item
                    block(node.body, env)
            elif isinstance(node, ast.While) and not node.orelse:
                while expr(node.test, env):
                    tick()
                    block(node.body, env)
            else:
                raise ValueError('unsupported statement: ' + type(node).__name__)
        return last
    result = block(tree.body, {})
    if type(result) is not int:
        raise ValueError('expected final integer assignment')
    return result


def _program(template, a, b, c, names, indent=' '):
    f, x, s, i, j, g, out = names
    lines = {
        'affine': [f'def {f}({x}):', f' {s}={x}*{b}+{c}', f' return {s}'],
        'for_sum': [f'def {f}({x}):', f' {s}={c}', f' for {i} in range({b}):', f'  {s}={s}+{x}+{i}', f' return {s}'],
        'while_sum': [f'def {f}({x}):', f' {s}={c}', f' {i}=0', f' while {i}<{b}:', f'  {s}={s}+{x}', f'  {i}={i}+1', f' return {s}'],
        'conditional': [f'def {f}({x}):', f' if {x}%2==0:', f'  return {x}+{b}+{c}', ' else:', f'  return {x}*{b}-{c}'],
        'nested_loops': [f'def {f}({x}):', f' {s}={c}', f' for {i} in range({b}):', f'  for {j} in range(2):', f'   {s}={s}+{x}+{i}+{j}', f' return {s}'],
        'nested_function': [f'def {f}({x}):', f' def {g}({i}):', f'  return {i}*{b}+{c}', f' return {g}({x})+{g}({x}+1)'],
        'branch_loop': [f'def {f}({x}):', f' {s}={c}', f' for {i} in range({b}):', f'  if {i}%2==0:', f'   {s}={s}+{x}', '  else:', f'   {s}={s}-1', f' return {s}'],
    }[template]
    return '\n'.join(indent * (len(line)-len(line.lstrip())) + line.lstrip() for line in lines) + f'\n{out}={f}({a})'


def _oracle(template, a, b, c):
    if template == 'affine': return a*b+c
    if template == 'for_sum': return c+b*a+b*(b-1)//2
    if template == 'while_sum': return c+a*b
    if template == 'conditional': return a+b+c if a%2 == 0 else a*b-c
    if template == 'nested_loops': return c+2*b*a+b*(b-1)+b
    if template == 'nested_function': return 2*a*b+b+2*c
    if template == 'branch_loop': return c+((b+1)//2)*a-b//2
    raise ValueError('unknown template')


def _record(split, seed, template, a, b, c, names, indent):
    program = _program(template, a, b, c, names, indent)
    answer = interpret(program)
    if answer != _oracle(template, a, b, c):
        raise AssertionError('independent verifier disagreement')
    canonical = canonicalize(program)
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    return {'record_id': f'coding-{split}-{seed}-{digest[:20]}', 'domain': 'coding',
            'topic': template.replace('_', '-'), 'difficulty': 'structural-ood' if split == 'final_ood' else 'foundation',
            'input': program, 'structured_input': {'ast': ast.dump(ast.parse(program), include_attributes=False),
                'parameters': {'a': a, 'b': b, 'c': c}, 'output_binding': names[-1]},
            'target': {'answer': str(answer)}, 'features': [],
            'verification_method': 'restricted-ast-interpreter+independent-template-oracle',
            'source': 'generated', 'split': split, 'generator_version': GENERATOR_VERSION,
            'template_id': 'python-' + template, 'canonical_key': digest}


def _generate(split, seed, count):
    if split not in SPLITS or type(count) is not int or count < 0:
        raise ValueError('invalid split/count')
    if count > 1000:
        raise ValueError('pilot generator capped at 1000 examples per split')
    rng = random.Random(f'{GENERATOR_VERSION}:coding:{split}:{seed}')
    templates = OOD_TEMPLATES if split == 'final_ood' else IID_TEMPLATES
    residue = SPLITS.index(split)
    rows, seen = [], set()
    attempts = 0
    while len(rows) < count:
        attempts += 1
        if attempts > 100000:
            raise RuntimeError('generation rejection bound')
        # Disjoint literal coordinate classes hold for every seed, not merely
        # the recorded pilot seeds. OOD also has unseen program structures.
        a = 4*rng.randrange(1, 31)+residue
        b, c = rng.randrange(1, 10), rng.randrange(1, 10)
        template = templates[len(rows) % len(templates)]
        names = rng.sample(['a','b','c','d','e','f','g','h','i','j','k','m','n','p','q','r','s','t','u','v','w','y','z'], 7)
        row = _record(split, seed, template, a, b, c, names, rng.choice((' ', '  ')))
        if row['canonical_key'] in seen:
            continue
        seen.add(row['canonical_key'])
        rows.append(row)
    return rows, {'attempted': attempts, 'accepted': len(rows),
                  'rejected_canonical_duplicate': attempts-len(rows),
                  'rejected_invalid': 0,
                  'unique_templates': len({r['template_id'] for r in rows}),
                  'input_ascii_bytes': {'min': min((len(r['input']) for r in rows), default=0),
                                        'max': max((len(r['input']) for r in rows), default=0),
                                        'total': sum(len(r['input']) for r in rows)}}


def generate(split, seed, count):
    return _generate(split, seed, count)[0]


def generation_statistics(split, seed, count):
    """Recorded acceptance/rejection counts for the same deterministic run."""
    return _generate(split, seed, count)[1]


def perturbations(row):
    """Metamorphic and answer-changing interventions, freshly AST-verified."""
    tree = ast.parse(row['input'])
    names = sorted({n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} |
                   {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)} |
                   {n.arg for n in ast.walk(tree) if isinstance(n, ast.arg)})
    mapping = {name: 'z'+str(i) for i, name in enumerate(n for n in names if n != 'range')}
    class Rename(ast.NodeTransformer):
        def visit_Name(self, n): n.id = mapping.get(n.id, n.id); return n
        def visit_FunctionDef(self, n): n.name = mapping[n.name]; return self.generic_visit(n)
        def visit_arg(self, n): n.arg = mapping[n.arg]; return n
    renamed = ast.unparse(Rename().visit(copy.deepcopy(tree)))
    formatted = ast.unparse(tree) + '\n'
    changed = copy.deepcopy(tree)
    changed.body[-1].value.args[0].value += 4
    altered = ast.unparse(changed)
    control = copy.deepcopy(tree)
    fn = control.body[0]
    fn.body = [ast.If(test=ast.Compare(left=ast.Name(id=fn.args.args[0].arg, ctx=ast.Load()),
                       ops=[ast.Gt()], comparators=[ast.Constant(value=0)]),
                       body=fn.body, orelse=[ast.Return(value=ast.Constant(value=0))])]
    control_program = ast.unparse(ast.fix_missing_locations(control))
    result = {}
    for name, program in [('renamed', renamed), ('reformatted', formatted),
                          ('control_flow_equivalent', control_program), ('literal_intervention', altered)]:
        item = copy.deepcopy(row)
        item['input'] = program
        item['target'] = {'answer': str(interpret(program))}
        item['canonical_key'] = hashlib.sha256(canonicalize(program).encode()).hexdigest()
        item['record_id'] += '-' + name
        item['structured_input']['ast'] = ast.dump(ast.parse(program), include_attributes=False)
        if name == 'renamed':
            item['structured_input']['output_binding'] = mapping[item['structured_input']['output_binding']]
        if name == 'literal_intervention':
            item['structured_input']['parameters']['a'] += 4
        params = item['structured_input']['parameters']
        expected = _oracle(row['template_id'].removeprefix('python-'), **params)
        if item['target']['answer'] != str(expected):
            raise AssertionError('perturbation verifier disagreement')
        result[name] = item
    return result
