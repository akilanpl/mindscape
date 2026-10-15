"""Bounded, ephemeral context. Only server-produced results can ground a reply."""
import ast
import copy
import hashlib
import json
import re
import time
from collections import OrderedDict

MAX_HISTORY = 24
MAX_TEXT = 16000
MAX_TOTAL = 64000
MAX_CONTEXTS = 32
TTL = 1800


def validate(data):
    history = data.get('history', [])
    cid = data.get('conversation_id', '')
    if not isinstance(cid, str) or (cid and not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', cid)):
        raise ValueError('Invalid conversation_id')
    if not isinstance(history, list) or len(history) > MAX_HISTORY:
        raise ValueError('History must contain at most 24 messages')
    total = 0
    for message in history:
        if not isinstance(message, dict) or set(message) != {'role', 'content'}:
            raise ValueError('History messages require only role and content')
        if message['role'] not in ('user', 'assistant') or not isinstance(message['content'], str):
            raise ValueError('Invalid history role or content')
        total += len(message['content'])
        if not 1 <= len(message['content']) <= MAX_TEXT or total > MAX_TOTAL:
            raise ValueError('History exceeds bounded text limits')
    return cid, history


def intent(question):
    if not isinstance(question, str):
        return False
    return bool(re.match(r'^\s*(?:(?:can|could|would) you\s+)?(?:please\s+)?(?:explain\b|give (?:me )?an example\b|why (?:did|do|is|does|was|use)\b)', question, re.I))


def key(cid, question, answer):
    return hashlib.sha256(json.dumps([cid, question, answer], ensure_ascii=False).encode()).digest()


class Contexts:
    """Access under Assistant._lock; no last-answer fallback across requests."""
    def __init__(self):
        self.entries = OrderedDict()

    def prune(self):
        now = time.monotonic()
        for k, (created, _) in list(self.entries.items()):
            if now - created > TTL:
                del self.entries[k]

    def previous(self, cid, history):
        self.prune()
        if len(history) < 2 or history[-2]['role'] != 'user' or history[-1]['role'] != 'assistant':
            return None
        item = self.entries.get(key(cid, history[-2]['content'], history[-1]['content']))
        return copy.deepcopy(item[1]) if item else None

    def remember(self, cid, question, response, context=None):
        self.prune()
        if response.get('status') == 'verified' and response.get('provenance', {}).get('tool_verified') is True:
            context = {'question': question, 'spec': response['structured_problem'], 'solver': response['solver_result']}
        elif response.get('status') != 'explained':
            context = None
        # Large responses remain usable, but are not retained in the context cache.
        if len(response['answer']) > MAX_TEXT:
            return
        k = key(cid, question, response['answer'])
        self.entries[k] = (time.monotonic(), copy.deepcopy(context))
        self.entries.move_to_end(k)
        while len(self.entries) > MAX_CONTEXTS:
            self.entries.popitem(last=False)


def clarification(reason):
    return {'status': 'clarification', 'domain': 'unsupported', 'answer': reason,
            'method': 'context clarification', 'tools': [], 'trace': [],
            'provenance': {'tool_verified': False, 'explanation_verified': False, 'pretrained_used': False, 'optimizer_updates': 0}}


def derivative_rules(expression):
    """Describe expression operations without evaluating code or re-solving."""
    expression = expression.replace('^', '**')
    try:
        tree = ast.parse(expression, mode='eval')
        nodes = list(ast.walk(tree))
        if len(nodes) > 256:
            return ''
    except (SyntaxError, ValueError, RecursionError):
        return ''
    lines = []
    if isinstance(tree.body, ast.BinOp) and isinstance(tree.body.op, ast.Mult):
        left = ast.get_source_segment(expression, tree.body.left)
        right = ast.get_source_segment(expression, tree.body.right)
        lines.append('Here the product factors are '+left+' and '+right+'. Keep each factor unchanged while differentiating the other, then add those two contributions.')
    for node in nodes:
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and len(node.args) == 1:
            inner = ast.get_source_segment(expression, node.args[0])
            name = node.func.id
            if name in ('sin', 'cos', 'exp'):
                outer = {'sin': 'cos('+inner+')', 'cos': '-sin('+inner+')', 'exp': 'exp('+inner+')'}[name]
                lines.append('For '+name+'('+inner+'), the outer derivative is '+outer+'; multiply it by the derivative of the inner expression '+inner+'.')
                if name == 'cos':
                    lines.append('The negative sign comes from differentiating cosine: d cos(u)/dx = -sin(u)*du/dx.')
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow) and isinstance(node.right, ast.Constant):
            lines.append('For '+str(ast.get_source_segment(expression, node))+', use the power rule: bring the exponent down, reduce it by one, and multiply by the derivative of the base.')
    return '\n'.join(lines)


def explain(question, context):
    if context is None:
        return clarification('Please resend the problem you want explained. I do not have a matching verified answer for this chat. A failed, edited, expired or missing answer cannot ground an explanation.')
    spec, solver = context['spec'], context['solver']
    steps = list(solver.get('steps', []))
    if spec.get('subject') == 'coding':
        for event in solver.get('trace', [])[-8:]:
            steps.append('Trace line '+str(event.get('line'))+' (call depth '+str(event.get('call_depth', 0))+'): '+json.dumps(event.get('variables', {}), ensure_ascii=False))
    domain = spec.get('subject')
    requested = 'physics' if re.search(r'\bphysics\b', question, re.I) else 'coding' if re.search(r'\b(?:python|code|output)\b', question, re.I) else None
    if requested and requested != domain:
        return clarification('The matching verified problem is in '+str(domain)+'. Please send the '+requested+' problem or output you want explained.')
    selection = re.search(r'\b(?:step\s*(\d+)|the\s+(first|second|third|fourth)\s+step)\b', question, re.I)
    if selection:
        if selection[1] and len(selection[1].lstrip('0')) > 3:
            return clarification('Please choose a step number from the displayed worked solution.')
        index = int(selection[1]) if selection[1] else {'first': 1, 'second': 2, 'third': 3, 'fourth': 4}[selection[2].lower()]
        if not 1 <= index <= len(steps):
            return clarification('The verified solution has '+str(len(steps))+' steps. Please choose one of those step numbers.')
        body = 'Step '+str(index)+' from the verified worked solution:\n'+steps[index-1]
    else:
        body = 'The original problem was:\n'+context['question']+'\n\n'
    derivative = spec.get('task') in ('differentiate', 'derivative')
    if derivative:
        # Explain rules without calculating or simplifying a second derivative.
        body += '\n\nFor differentiation, the product rule says: differentiate the first factor and keep the second, then keep the first and differentiate the second. The chain rule says: for a function inside another function, multiply the outer derivative by the inner derivative.'
        expression = str(spec.get('expression', spec.get('expr', '')))
        body += '\n'+derivative_rules(expression)
        if re.search(r'example', question, re.I):
            body += '\nFor example, the chain rule for sin(2*x) gives cos(2*x)*2: differentiate sine, then multiply by the derivative of 2*x. This illustrates the rule; it does not replace your answer.'
    elif re.search(r'negative|minus', question, re.I):
        return clarification('Please identify the negative term in the displayed calculation. I can repeat its checked steps, but cannot infer a physical sign convention or a program operation that was not supplied.')
    elif re.search(r'example', question, re.I):
        body += '\nThe original supplied values and their substitution below provide the grounded example. Send different values if you want a separate calculation.'
    elif domain == 'physics':
        body += '\nThe worked solution identifies the physical relation, substitutes the supplied quantities, and reports the resulting units. Its validity is limited to the assumptions in that solution.'
    elif domain == 'coding':
        body += '\nRead the program in execution order. The trace records the actual line, call depth and variable values from the bounded interpreter. The final answer is the exact output, including whitespace.'
    if not selection:
        body += '\n\nVerified worked solution:\n'+'\n'.join(f'{i+1}. {step}' for i, step in enumerate(steps))
    answer = solver.get('answer')
    rendered = answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False)
    final = rendered + ((' '+solver['units']) if isinstance(solver.get('units'), str) else '')
    body += '\n\nExact verified final answer:\n'+final+'\n\nThese deterministic explanations refer to the previous tool-verified result. The explanatory prose has not been independently verified.\nOriginal check: '+str(solver.get('verification', {}).get('check', 'Passed.'))
    return {'status': 'explained', 'domain': domain, 'answer': body, 'method': 'deterministic explanation of prior verified result',
            'verified_answer': copy.deepcopy(answer), 'verified_units': solver.get('units'), 'tools': [],
            'trace': [{'kind': 'conversation context', 'summary': 'Matched server-owned prior response; no new solver or model inference.'}],
            'provenance': {'tool_verified': False, 'previous_result_verified': True, 'explanation_verified': False,
                           'pretrained_used': False, 'native_r6_used': False, 'optimizer_updates': 0}}
