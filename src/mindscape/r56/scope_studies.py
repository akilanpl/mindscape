"""Untrained supplemental semantics/language study candidates; no optimizer paths."""
from __future__ import annotations
import ast
import hashlib
import json
from .language_model import ARGUMENT_SLOTS, tokenize

STATUS = 'SUPPLEMENTAL PROPOSAL ONLY — NO SCIENTIFIC APPROVAL'


class RestrictedReference:
    """Independent bounded integer AST semantics. Never eval/exec/compile source."""
    def __init__(self): self.steps=0; self.functions={}; self.depth=0
    def tick(self):
        self.steps+=1
        if self.steps>2000: raise ValueError('Reference step budget exceeded')
    def value(self,node,env):
        self.tick()
        if isinstance(node,ast.Constant) and type(node.value) is int and abs(node.value)<=10000: return node.value
        if isinstance(node,ast.Name) and node.id in env: return env[node.id]
        if isinstance(node,ast.BinOp):
            left,right=self.value(node.left,env),self.value(node.right,env)
            ops={ast.Add:lambda:left+right,ast.Sub:lambda:left-right,ast.Mult:lambda:left*right,ast.Mod:lambda:left%right}
            if type(node.op) not in ops: raise ValueError('Unsupported arithmetic')
            result=ops[type(node.op)]()
            if abs(result)>1000000: raise ValueError('Integer magnitude exceeded')
            return result
        if isinstance(node,ast.Compare) and len(node.ops)==1:
            a,b=self.value(node.left,env),self.value(node.comparators[0],env)
            if isinstance(node.ops[0],ast.Lt): return a<b
            if isinstance(node.ops[0],ast.Eq): return a==b
            if isinstance(node.ops[0],ast.Gt): return a>b
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and not node.keywords and node.func.id in self.functions:
            function=self.functions[node.func.id]
            if len(node.args)!=len(function.args.args) or self.depth>=4: raise ValueError('Call bounds exceeded')
            values=[self.value(arg,env) for arg in node.args]
            self.depth+=1
            try:
                returned,value=self.block(function.body,dict(zip([a.arg for a in function.args.args],values)))
                if not returned: raise ValueError('Explicit return required')
                return value
            finally:self.depth-=1
        raise ValueError('Unsupported expression')
    def block(self,body,env):
        for node in body:
            self.tick()
            if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name): env[node.targets[0].id]=self.value(node.value,env)
            elif isinstance(node,ast.Return): return True,self.value(node.value,env)
            elif isinstance(node,ast.If):
                returned,value=self.block(node.body if self.value(node.test,env) else node.orelse,env)
                if returned:return returned,value
            elif isinstance(node,ast.For) and isinstance(node.target,ast.Name) and not node.orelse:
                call=node.iter
                if not isinstance(call,ast.Call) or not isinstance(call.func,ast.Name) or call.func.id!='range' or len(call.args)!=1 or call.keywords: raise ValueError('Bounded range only')
                stop=self.value(call.args[0],env)
                if type(stop) is not int or not 0<=stop<=20: raise ValueError('Range bound exceeded')
                for index in range(stop):
                    env[node.target.id]=index;returned,value=self.block(node.body,env)
                    if returned:return returned,value
            else:raise ValueError('Unsupported statement')
        return False,None
    def evaluate(self,source):
        if not isinstance(source,str) or len(source)>4000:raise ValueError('Source bound exceeded')
        tree=ast.parse(source)
        nodes=list(ast.walk(tree))
        if len(nodes)>200:raise ValueError('AST budget exceeded')
        allowed=(ast.Module,ast.FunctionDef,ast.arguments,ast.arg,ast.Assign,ast.Name,ast.Store,ast.Load,ast.Constant,ast.BinOp,ast.Add,ast.Sub,ast.Mult,ast.Mod,ast.Compare,ast.Lt,ast.Gt,ast.Eq,ast.Call,ast.Return,ast.If,ast.For)
        if any(not isinstance(n,allowed) for n in nodes):raise ValueError('Unsupported static syntax')
        body=[]
        for node in tree.body:
            if isinstance(node,ast.FunctionDef):
                if node.decorator_list or node.returns or node.args.defaults or node.args.kwonlyargs or node.args.vararg or node.args.kwarg or node.args.posonlyargs or any(a.annotation for a in node.args.args):raise ValueError('Plain positional function only')
                if node.name in self.functions or node.name=='range':raise ValueError('Duplicate/reserved function')
                self.functions[node.name]=node
            else:body.append(node)
        env={};self.block(body,env)
        if type(env.get('result')) is not int:raise ValueError('Integer result assignment required')
        return env['result']


def semantic_reference(source): return RestrictedReference().evaluate(source)


def semantic_candidates():
    """Protected references are returned separately from source-only observations."""
    forms={'train':[
        ('straight','a = 3\nb = 4\nresult = a * b + 2'),
        ('single-loop','result = 0\nfor i in range(5):\n result = result + i'),
        ('single-branch','a = 3\nif a < 5:\n result = 7\nelse:\n result = 2'),
        ('straight-function','def f(x):\n return x * 2 + 1\nresult = f(4)')],
        'heldout':[
        ('loop-with-branch','result = 0\nfor i in range(7):\n if i % 2 == 0:\n  result = result + i\n else:\n  result = result + 1'),
        ('nested-loop','result = 0\nfor i in range(4):\n for j in range(3):\n  result = result + i * j'),
        ('function-with-branch','def f(x):\n if x > 3:\n  return x * x\n else:\n  return x + 2\nresult = f(5) + f(2)'),
        ('function-with-loop','def f(x):\n total = 0\n for i in range(x):\n  total = total + i\n return total\nresult = f(6)')]}
    problems={};references={}
    for split,rows in forms.items():
        problems[split]=[]
        for structure,source in rows:
            identity=hashlib.sha256(source.encode()).hexdigest()
            problems[split].append({'identity':identity,'structure':structure,'source':source,'task':'predict integer result'})
            references[identity]={'result':semantic_reference(source),'provenance':'restricted AST reference; requires independent WASI agreement'}
    return problems,references


def training_source_vocabulary(problems):
    from .language_model import FrozenVocabulary
    return FrozenVocabulary.from_training_forms([r['source'] for r in problems['train']],max_length=128)


def wasi_crosscheck(source, expected, runtime):
    """Existing WASI sandbox only. No fallback execution or expected value in guest."""
    from mindscape.coding.sandbox import WasiSandbox
    try:
        # Reference validation bounds guest source before any execution.
        semantic_reference(source)
        sandbox=WasiSandbox(runtime,timeout=5)
        sandbox.probe()
        execution=sandbox.execute({},source+'\nprint(result)\n')
    except (ValueError,RuntimeError,OSError) as exc:
        return {'status':'unavailable','agreement':None,'detail':type(exc).__name__,'host_execution':False}
    text=execution.stdout.strip()
    valid=execution.returncode==0 and not execution.timeout and text.lstrip('-').isdigit()
    return {'status':'checked' if valid else 'execution_failed','agreement':int(text)==expected if valid else None,
            'returncode':execution.returncode,'host_execution':False,'isolation':'existing WASI'}


def language_metrics(rows,predictions,vocabulary):
    """Exact spans/text, present/null, extraction precision/recall and abstention.

    Prediction format: {intent, spans:{slot:[start,end]}, abstained:bool}; BOS index
    zero is null. Protected annotations are consumed only after predictions exist.
    """
    if len(rows)!=len(predictions):raise ValueError('Prediction alignment required')
    groups={}
    for row,pred in zip(rows,predictions):
        if not isinstance(pred,dict) or type(pred.get('abstained',False)) is not bool:raise ValueError('Malformed prediction')
        scope=row.get('split','unknown');g=groups.setdefault(scope,dict(examples=0,intent_exact=0,all_arguments_exact=0,present=0,null=0,present_exact=0,null_correct=0,predicted_present=0,extraction_true_positive=0,abstained=0,abstained_unsupported=0,unsupported=0,oov_tokens=0))
        g['examples']+=1;abstained=pred.get('abstained',False);g['abstained']+=abstained;unsupported=row['intent']=='unsupported';g['unsupported']+=unsupported;g['abstained_unsupported']+=abstained and unsupported
        g['intent_exact']+=not abstained and pred.get('intent')==row['intent'];g['oov_tokens']+=vocabulary.encode(row['text'])['unknown_tokens']
        spans=pred.get('spans',{})
        if not isinstance(spans,dict) or set(spans)-set(ARGUMENT_SLOTS):raise ValueError('Invalid slot prediction')
        pieces=tokenize(row['text']);all_exact=not abstained
        for slot in ARGUMENT_SLOTS:
            target=tuple(row['spans'].get(slot,(0,0)));raw=spans.get(slot,(0,0));guess=(0,0) if abstained else tuple(raw)
            if len(guess)!=2 or any(type(v) is not int for v in guess) or not (guess==(0,0) or 1<=guess[0]<=guess[1]<=len(pieces)):raise ValueError('Invalid predicted span')
            present=target!=(0,0);predicted=guess!=(0,0);exact=guess==target and not abstained
            g['present']+=present;g['null']+=not present;g['present_exact']+=present and exact;g['null_correct']+=not present and exact;g['predicted_present']+=predicted;g['extraction_true_positive']+=present and exact;all_exact=all_exact and exact
        g['all_arguments_exact']+=all_exact
    for g in groups.values():
        ratio=lambda a,b:a/b if b else None
        g['extraction_precision']=ratio(g['extraction_true_positive'],g['predicted_present']);g['extraction_recall']=ratio(g['extraction_true_positive'],g['present']);g['present_slot_exact_rate']=ratio(g['present_exact'],g['present']);g['null_slot_accuracy']=ratio(g['null_correct'],g['null']);g['abstention_precision']=ratio(g['abstained_unsupported'],g['abstained']);g['abstention_recall']=ratio(g['abstained_unsupported'],g['unsupported'])
    return {'groups':groups,'status':STATUS,'scope':'Exact token-boundary extraction only; no language reasoning or semantic qualification.'}


def freeze_supplement(directory):
    """Separate candidate package; does not modify or add fits to the 72-fit study."""
    from pathlib import Path
    problems,references=semantic_candidates()
    destination=Path(directory);destination.mkdir(parents=True,exist_ok=False)
    vocabulary=training_source_vocabulary(problems)
    artifacts={'problems.json':problems,'protected-references.json':references,'vocabulary.json':vocabulary.manifest()}
    locks={}
    for name,payload in artifacts.items():
        raw=json.dumps(payload,sort_keys=True,allow_nan=False).encode();(destination/name).write_bytes(raw);locks[name]=hashlib.sha256(raw).hexdigest()
    manifest={'version':'supplemental-semantics-candidate-v1','status':STATUS,'scientific_decision':'NO_GO_UNREVIEWED',
        'optimizer_updates':0,'primary_campaign_fits_added':0,'artifacts':locks,
        'limitation':'Eight illustrative programs only; not a sufficient benchmark. WASI crosschecks must precede use of references.'}
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest
