"""Observable graph packing, uncorrected decoding, and separate post-hoc verifiers."""
from __future__ import annotations
import ast
import hashlib
import math
from copy import deepcopy
import torch
from .neural import GraphInput

DOMAIN_IDS = {'math': 0, 'coding': 1, 'physics': 2}


def _finite(value):
    if type(value) not in (int, float) or not math.isfinite(value) or abs(value) > 1e6:
        raise ValueError('Require finite numeric observations with magnitude <=1e6')
    return float(value)


def _digit(value):
    if type(value) is not int or not 0 <= value <= 9: raise ValueError('Digit must be an integer in 0..9')
    return value


def _incoming(value):
    if type(value) is not int or value not in (0, 1): raise ValueError('Incoming carry/borrow must be 0 or 1')
    return value


def _concepts(problem):
    concepts = problem.get('concept_ids', [])
    if not isinstance(concepts, list) or len(concepts) > 16 or any(type(v) is not int or not 0 <= v < 32 for v in concepts):
        raise ValueError('concept_ids must explicitly supply up to 16 integers in 0..31')
    return concepts


def _decimal(value):
    text = str(value)
    if len(text) > 128 or not text or any(c not in '0123456789' for c in text):
        raise ValueError('Addition operands must be unsigned decimal strings of at most 128 digits')
    return text


def pack_problem(problem: dict, device='cpu') -> GraphInput:
    domain = problem.get('domain')
    if domain not in DOMAIN_IDS: raise ValueError('domain must be math, coding or physics')
    rows, types, edges, relations = [], [], [], []
    action = [0., 0., 0., 0.]
    if domain == 'math':
        operation = problem.get('operation', 'add_digit')
        if operation not in ('add_digit', 'borrow_digit'): raise ValueError('Graph packing accepts local digit transitions')
        left, right = _digit(problem['left']), _digit(problem['right'])
        incoming = _incoming(problem.get('incoming', problem.get('carry_in', problem.get('borrow_in', 0))))
        rows = [[left/9, 0, 0, 0, 0, 0, 0, 0], [right/9, 0, 0, 0, 0, 0, 0, 0],
                [incoming, 0, 0, 0, 0, 0, 0, 0]]
        types = [0, 1, 2]
        edges = [(0, 2), (1, 2), (2, 0), (2, 1)]
        relations = [0, 1, 2, 2]
        action[0 if operation == 'add_digit' else 1] = 1
    elif domain == 'coding':
        source, old = problem['source'], problem['old_name']
        candidates = problem['candidates']
        if not isinstance(source, str) or len(source.encode()) > 16000 or not isinstance(old, str) or not old.isidentifier() or len(old)>64:
            raise ValueError('Coding requires <=16KB Python source and an identifier old_name')
        if not isinstance(candidates, list) or not 1 <= len(candidates) <= 16 or any(not isinstance(v, str) or not v.isidentifier() or len(v)>64 for v in candidates):
            raise ValueError('Supply 1..16 identifier candidate classes')
        tree = ast.parse(source)
        nodes = list(ast.walk(tree))
        if len(nodes) + len(candidates) + 1 > 256: raise ValueError('Coding AST exceeds node budget')
        rows.append([min(len(old), 64)/64, 0, 0, 0, 0, 0, 0, 0]); types.append(3)
        lookup = {id(node): i+1 for i,node in enumerate(nodes)}
        for node in nodes:
            digest = hashlib.sha256(type(node).__name__.encode()).digest()
            name = getattr(node, 'id', getattr(node, 'arg', ''))
            rows.append([digest[0]/255, digest[1]/255, float(name == old), min(getattr(node, 'lineno', 0), 256)/256,
                         min(len(name), 64)/64, 0, 0, 0]); types.append(4)
            for child in ast.iter_child_nodes(node):
                edges.append((lookup[id(node)], lookup[id(child)])); relations.append(3)
                edges.append((lookup[id(child)], lookup[id(node)])); relations.append(4)
        for index, name in enumerate(candidates):
            digest = hashlib.sha256(name.encode()).digest()
            rows.append([min(len(name),64)/64, index/15, digest[0]/255, digest[1]/255, 0, 0, 0, 0]); types.append(5)
            edges.append((0,len(rows)-1)); relations.append(5)
        requested = problem.get('requested_name')
        if requested is not None:
            if not isinstance(requested, str) or not requested.isidentifier() or len(requested)>64: raise ValueError('requested_name must be an identifier task input')
            if len(rows) >= 256: raise ValueError('Requested-name node exceeds node budget')
            digest = hashlib.sha256(requested.encode()).digest()
            rows.append([min(len(requested),64)/64, 0, digest[0]/255, digest[1]/255, 0,0,0,0]); types.append(9)
            request_index = len(rows)-1
            for candidate_index in range(len(nodes)+1,len(nodes)+1+len(candidates)):
                edges.append((request_index,candidate_index)); relations.append(5)
        action[2] = 1
    else:
        velocity = problem['velocity']; force = problem.get('force',[0,0]); impulse = problem.get('impulse',[0,0])
        if any(not isinstance(v,(list,tuple)) or len(v)!=2 for v in (velocity,force,impulse)): raise ValueError('Physics vectors must have two components')
        mass, dt = _finite(problem['mass']), _finite(problem.get('dt',1))
        if mass <= 0 or dt < 0: raise ValueError('Mass positive; dt nonnegative')
        rows = [[_finite(velocity[0]), _finite(velocity[1]), mass, 0,0,0,0,0],
                [_finite(force[0]), _finite(force[1]), dt, 0,0,0,0,0],
                [_finite(impulse[0]), _finite(impulse[1]), 0,0,0,0,0,0]]
        types=[6,7,8]; edges=[(1,0),(2,0),(0,1),(0,2)]; relations=[6,7,6,7]; action[3]=1
    edge_tensor = torch.tensor(edges, dtype=torch.long, device=device).T.contiguous() if edges else torch.empty((2,0),dtype=torch.long,device=device)
    return GraphInput(torch.tensor(rows,dtype=torch.float32,device=device), torch.tensor(types,dtype=torch.long,device=device),
                      edge_tensor, torch.tensor(relations,dtype=torch.long,device=device),
                      torch.tensor(action,dtype=torch.float32,device=device), torch.tensor(_concepts(problem),dtype=torch.long,device=device), DOMAIN_IDS[domain])


def unpack_prediction(problem, output):
    """Decode model choices without applying targets, constraints or corrections."""
    domain = problem['domain']
    scores = output['action_scores'].detach().cpu().tolist()
    base = {'action_scores': scores, 'selected_action': max(range(len(scores)), key=scores.__getitem__),
            'selection_policy': 'deterministic argmax of learned action scores; domain transition schedule supplied by controller'}
    if domain == 'math':
        return {**base, 'digit': int(output['math_digit_logits'].argmax().item()),
                'outgoing': int(output['math_carry_logits'].argmax().item())}
    if domain == 'coding':
        logits = output['coding_logits'][:len(problem['candidates'])]
        index = int(logits.argmax().item())
        return {**base, 'candidate_index': index, 'name': problem['candidates'][index]}
    return {**base, 'velocity': [float(v)+float(delta) for v,delta in zip(problem['velocity'], output['physics_delta'].detach().cpu().tolist())],
            'predicted_delta': output['physics_delta'].detach().cpu().tolist()}


def verify_prediction(problem, prediction):
    """Independent post-prediction evaluator. Never passed back into the model."""
    domain = problem['domain']
    if domain == 'math':
        if problem.get('operation') == 'add':
            expected = str(int(_decimal(problem['left'])) + int(_decimal(problem['right'])))
            return {'passed': prediction['answer'] == expected, 'expected': expected, 'scope': 'exact full answer; post-hoc only'}
        left, right, incoming = _digit(problem['left']), _digit(problem['right']), _incoming(problem.get('incoming',problem.get('carry_in',problem.get('borrow_in',0))))
        value = left-right-incoming if problem.get('operation') == 'borrow_digit' else left+right+incoming
        expected = {'digit': value%10, 'outgoing': int(value<0) if problem.get('operation')=='borrow_digit' else value//10}
        return {'passed': all(prediction[k]==v for k,v in expected.items()), 'expected': expected, 'scope': 'post-hoc local arithmetic'}
    if domain == 'coding':
        target = problem.get('target_name')
        if target is None: return {'passed': None, 'reason': 'No independently supplied target candidate; no correctness claim'}
        if target not in problem['candidates']: raise ValueError('Verifier target must be one of supplied candidates')
        renamed = rename_ast(problem['source'],problem['old_name'],prediction['name'])
        return {'passed': prediction['name']==target, 'expected': target, 'renamed_source': renamed,
                'scope': 'candidate-label correctness and deterministic AST serialization; no behavioral-equivalence claim'}
    mass = _finite(problem['mass']); dt = _finite(problem.get('dt',1))
    force, impulse = problem.get('force',[0,0]), problem.get('impulse',[0,0])
    expected = [float(v)+(float(f)*dt+float(j))/mass for v,f,j in zip(problem['velocity'],force,impulse)]
    residuals = [float(v)-e for v,e in zip(prediction['velocity'],expected)]
    return {'passed': all(abs(r)<=1e-4*max(1,abs(e)) for r,e in zip(residuals,expected)),
            'expected': expected, 'residuals': residuals, 'scope': 'post-hoc Newtonian impulse/constant-force reference'}


def rename_ast(source, old, new):
    """Deterministic action application, not learned rename selection or safe refactoring."""
    tree=ast.parse(source)
    class Rename(ast.NodeTransformer):
        def visit_Name(self,node):
            if node.id==old: node.id=new
            return node
        def visit_arg(self,node):
            if node.arg==old: node.arg=new
            return node
    return ast.unparse(ast.fix_missing_locations(Rename().visit(tree)))


def infer_problem(model, problem: dict, *, stateless=False):
    original=deepcopy(problem)
    device=next(model.parameters()).device
    was_training=model.training
    model.eval()
    try:
        with torch.no_grad():
            if problem.get('domain')=='math' and problem.get('operation')=='add':
                left,right=_decimal(problem['left']),_decimal(problem['right'])
                incoming=0; memory=None; digits=[]; transitions=[]
                for index in range(max(len(left),len(right))):
                    local={'domain':'math','operation':'add_digit','left':int(left[-1-index]) if index<len(left) else 0,
                           'right':int(right[-1-index]) if index<len(right) else 0,'incoming':incoming,
                           'concept_ids':_concepts(problem)}
                    out=model(pack_problem(local,device),memory,stateless=stateless)
                    step=unpack_prediction(local,out); transitions.append(step); digits.append(str(step['digit']))
                    incoming=step['outgoing']; memory=out['memory']
                if incoming: digits.append(str(incoming))
                prediction={'answer': ''.join(reversed(digits)).lstrip('0') or '0', 'transitions':transitions,
                            'controller': 'supplied LSD traversal; own predicted carries; no verifier feedback'}
            else:
                out=model(pack_problem(problem,device),stateless=stateless)
                prediction=unpack_prediction(problem,out)
        sealed=deepcopy(prediction)
        verification=verify_prediction(original,deepcopy(sealed))
        return {'prediction':sealed,'verification':verification,'model_status':'untrained_or_checkpoint_supplied',
                'learned_claim':False,'concept_source':'explicit caller concept IDs; trainable embeddings, no learned concept-label claim',
                'stateless':stateless}
    finally:
        model.train(was_training)
