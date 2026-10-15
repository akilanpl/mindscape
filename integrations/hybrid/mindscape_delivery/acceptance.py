"""Independent held-out evaluator. Never feeds references into the application."""
from __future__ import annotations
import argparse,ast,hashlib,json,math,re,time,urllib.request
from pathlib import Path


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def freeze_sources(source:Path,output:Path,model_manifest:Path|None=None):
    files={str(p.relative_to(source)):digest(p) for p in sorted(source.rglob('*')) if p.is_file() and (p.suffix in ('.py','.html','.js','.css','.toml','.json','.sh') or p.name.startswith('requirements')) and not set(p.parts)&{'__pycache__','.git','.cache','.pytest_cache','node_modules'}}
    record={'source_root':str(source.resolve()),'files':files,'freeze_unix_seconds':time.time()}
    if model_manifest:record['model_manifest']={'path':str(model_manifest.resolve()),'sha256':digest(model_manifest)}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(record,indent=2,sort_keys=True)+'\n');return record

def check_frozen(record):
    root=Path(record['source_root'])
    actual={str(p.relative_to(root)):digest(p) for p in sorted(root.rglob('*')) if p.is_file() and (p.suffix in ('.py','.html','.js','.css','.toml','.json','.sh') or p.name.startswith('requirements')) and not set(p.parts)&{'__pycache__','.git','.cache','.pytest_cache','node_modules'}}
    if actual!=record['files']:raise RuntimeError('Implementation changed after acceptance freeze; do not tune against held-out cases.')
    m=record.get('model_manifest')
    if m and digest(m['path'])!=m['sha256']:raise RuntimeError('Model manifest changed after freeze')

def safe_symbolic(text):
    """Restricted AST to SymPy; no eval, arbitrary attributes, imports or strings."""
    import sympy as s
    text=str(text).strip().replace('^','**').replace('−','-').replace('π','pi')
    if len(text)>300:raise ValueError('Expression too long')
    functions={'sin':s.sin,'cos':s.cos,'tan':s.tan,'exp':s.exp,'log':s.log,'sqrt':s.sqrt}
    def visit(n):
        if isinstance(n,ast.Constant) and isinstance(n.value,(int,float)) and not isinstance(n.value,bool):return s.sympify(n.value)
        if isinstance(n,ast.Name) and n.id in ('x','C','pi','E'):return {'x':s.Symbol('x'),'C':s.Symbol('C'),'pi':s.pi,'E':s.E}[n.id]
        if isinstance(n,ast.UnaryOp) and isinstance(n.op,(ast.UAdd,ast.USub)):return visit(n.operand) if isinstance(n.op,ast.UAdd) else -visit(n.operand)
        if isinstance(n,ast.BinOp) and isinstance(n.op,(ast.Add,ast.Sub,ast.Mult,ast.Div,ast.Pow)):
            a,b=visit(n.left),visit(n.right)
            if isinstance(n.op,ast.Pow) and (not b.is_number or abs(float(b))>20):raise ValueError('Power limit')
            if isinstance(n.op,ast.Add):return a+b
            if isinstance(n.op,ast.Sub):return a-b
            if isinstance(n.op,ast.Mult):return a*b
            if isinstance(n.op,ast.Div):return a/b
            return a**b
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in functions and len(n.args)==1 and not n.keywords:return functions[n.func.id](visit(n.args[0]))
        raise ValueError('Unsupported symbolic expression')
    tree=ast.parse(text,mode='eval')
    if sum(1 for _ in ast.walk(tree))>120:raise ValueError('Expression complexity')
    return visit(tree.body)

def candidate_answer(response):
    sr=response.get('solver_result')
    if isinstance(sr,dict) and sr.get('supported') is True and 'answer' in sr:
        answer=sr['answer']
        if isinstance(answer,dict):
            if 'image_distance' in answer:return answer['image_distance']
            if 'rows' in answer:return answer['rows']
        return answer
    answer=response.get('answer')
    if isinstance(answer,(int,float,list,dict)):return answer
    # Structured neural response, if available, remains an unverified prediction.
    if isinstance(answer,str):
        try:
            value=json.loads(answer)
            if isinstance(value,dict) and 'answer' in value:return value['answer']
        except (ValueError,TypeError):pass
    return answer

def answer_matches(case,actual):
    if actual is None:return False
    kind=case['answer_kind'];expected=case['expected']
    if kind in ('symbolic','antiderivative'):
        import sympy as s
        try:
            a=safe_symbolic(actual);e=safe_symbolic(expected)
            if kind=='antiderivative':return s.simplify(s.diff(a-e,s.Symbol('x')))==0
            return s.simplify(a-e)==0
        except Exception:return False
    if kind in ('matrix','roots','table'):
        try:
            value=actual
            if isinstance(value,str):value=ast.literal_eval(value)
            if kind=='roots':return sorted(float(x) for x in value)==sorted(float(x) for x in expected)
            if kind=='matrix':
                return [[float(safe_symbolic(v)) for v in row] for row in value]==[[float(v) for v in row] for row in expected]
            return value==expected
        except Exception:return False
    if kind=='stdout':return str(actual).strip().replace('\r\n','\n')==expected
    if kind=='numeric':
        try:
            if isinstance(actual,list) and len(actual)==1:actual=actual[0]
            if isinstance(actual,str):
                a=actual.strip()
                # Only trailing physical units permitted; never search arbitrary reasoning for the target number.
                a=re.sub(r'\s*(N/C|N|A|V|cm|ohm|Ω)\s*$','',a)
                actual=float(a)
            return not isinstance(actual,bool) and math.isfinite(float(actual)) and math.isclose(float(actual),float(expected),rel_tol=1e-5,abs_tol=1e-8)
        except (ValueError,TypeError,OverflowError):return False
    return False

def safe_unsupported(response):
    status=str(response.get('status','')).lower()
    sr=response.get('solver_result') or {}
    explicit=status in ('unsupported','needs_clarification','clarification','refused') or sr.get('supported') is False
    text=str(response.get('answer','')).lower()
    lexical=any(w in text for w in ('unsupported','cannot','not supported','missing','not implemented','unsafe','unresolved','insufficient','could not'))
    return bool(explicit or lexical)

UNIT_INFO={'N':('force',1.0),'newton':('force',1.0),'N/C':('electric-field',1.0),'V/m':('electric-field',1.0),'A':('current',1.0),'mA':('current',.001),'V':('voltage',1.0),'mV':('voltage',.001),'ohm':('resistance',1.0),'Ω':('resistance',1.0),'m':('length',1.0),'cm':('length',.01),'mm':('length',.001)}

def response_unit(response):
    sr=response.get('solver_result') or {}
    unit=sr.get('units',response.get('units'))
    if isinstance(unit,dict):unit=unit.get('answer',unit.get('result',unit.get('image_distance','')))
    return str(unit).strip() if unit is not None else None

def unit_matches(case,response):
    if not case.get('unit'):return None
    actual=response_unit(response);expected=case['unit']
    return actual in UNIT_INFO and expected in UNIT_INFO and UNIT_INFO[actual][0]==UNIT_INFO[expected][0]

def response_answer_correct(case,response):
    actual=candidate_answer(response)
    if case.get('unit') and unit_matches(case,response):
        try:
            actual=float(actual)*UNIT_INFO[response_unit(response)][1]/UNIT_INFO[case['unit']][1]
        except (ValueError,TypeError):pass
    return answer_matches(case,actual)

def summarize(rows,protocol):
    topics={};arms={}
    for arm in protocol['arms']:
        selected=[r for r in rows if r['arm']==arm]; supported=[r for r in selected if r['case']['supported_expected']]
        for topic in sorted(set(r['case']['topic'] for r in selected)):
            rs=[r for r in selected if r['case']['topic']==topic];n=len(rs)
            explanation=[r.get('explanation_correct') for r in rs if r['case']['supported_expected']]
            units=[r['unit_valid'] for r in rs if r['unit_valid'] is not None]
            correctness=(sum(r['answer_correct'] for r in rs)/n if n else None) if arm!='native-neural-only' else None
            topic_result={'count':n,'answer_accuracy':correctness,'explanation_accuracy':sum(v is True for v in explanation)/len(explanation) if explanation else None,'explanation_review_complete':all(v is not None for v in explanation),'unit_accuracy':sum(units)/len(units) if units else None,'unsupported_safe_handling':sum(r['unsupported_handled'] is True for r in rs)/n if topic=='unsupported' else None,'guaranteed_demo_scope':False}
            if topic!='unsupported':topic_result['guaranteed_demo_scope']=bool(correctness==1 and topic_result['explanation_review_complete'] and topic_result['explanation_accuracy']==1 and topic_result['unit_accuracy'] in (None,1) and arm=='hybrid')
            topics[f'{arm}/{topic}']=topic_result
        lat=sorted(r['latency_seconds'] for r in selected)
        arms[arm]={'cases':len(selected),'answer_correct_supported':sum(r['answer_correct'] for r in supported),'supported_denominator':len(supported),'supported_answer_accuracy':(sum(r['answer_correct'] for r in supported)/len(supported) if supported else None) if arm!='native-neural-only' else None,'unsupported_on_supported_tasks':sum(r['response'].get('status') in ('unsupported','unavailable') for r in supported),'latency_median_seconds':lat[len(lat)//2] if lat else None,'latency_p95_seconds':lat[max(0,math.ceil(.95*len(lat))-1)] if lat else None,'interpretation':'Native tasks outside the CBSE interface are unsupported; a zero answer count is coverage failure, not a measured neural response accuracy.' if arm=='native-neural-only' else 'Frozen held-out predictions; no tuning after inspection.'}
    for arm,data in arms.items():
        data['latency_acceptance_pass']=bool(data['latency_p95_seconds'] is not None and data['latency_p95_seconds']<=protocol['predeclared_acceptance']['p95_warm_latency_seconds'])
    if not arms.get('hybrid',{}).get('latency_acceptance_pass'):
        for key,data in topics.items():
            if key.startswith('hybrid/'):data['guaranteed_demo_scope']=False
    return {'arms':arms,'topics':topics,'guaranteed_demo_topics':[k.split('/',1)[1] for k,v in topics.items() if k.startswith('hybrid/') and v['guaranteed_demo_scope']],'thresholds':protocol['predeclared_acceptance'],'warning':'Two questions per topic establish smoke-test behavior only. Explanation grading is independently reviewed, never inferred from answer equality or verification labels.'}

def run(args):
    cases=json.loads(Path(args.cases).read_text());protocol=json.loads(Path(args.protocol).read_text());frozen=json.loads(Path(args.freeze).read_text())
    if digest(args.cases)!=protocol['dataset_sha256']:raise RuntimeError('Protected data hash mismatch')
    check_frozen(frozen);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'raw-predictions.jsonl').exists():raise FileExistsError('A final evaluation already exists; no test feedback retries allowed')
    rows=[]
    map_modes={'native-neural-only':'native','pretrained-neural-only':'pretrained','hybrid':'hybrid'}
    with (out/'raw-predictions.jsonl').open('x') as stream:
        for arm in protocol['arms']:
            check_frozen(frozen)
            for case in cases:
                started=time.perf_counter()
                try:
                    payload={'question':case['question'],'message':case['question'],'mode':map_modes[arm],'inference_mode':map_modes[arm]}
                    request=urllib.request.Request(args.endpoint,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
                    with urllib.request.urlopen(request,timeout=args.timeout) as resp:response=json.load(resp)
                except Exception as exc:response={'status':'error','answer':None,'error':f'{type(exc).__name__}: {str(exc)}'}
                latency=time.perf_counter()-started;correct=response_answer_correct(case,response) if case['supported_expected'] else safe_unsupported(response)
                row={'arm':arm,'case':case,'response':response,'answer_correct':bool(correct),'unit_valid':unit_matches(case,response),'explanation_correct':None if case['supported_expected'] else None,'unsupported_handled':safe_unsupported(response) if not case['supported_expected'] else None,'latency_seconds':latency}
                stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();rows.append(row)
    check_frozen(frozen)
    (out/'summary-provisional.json').write_text(json.dumps(summarize(rows,protocol),indent=2,sort_keys=True)+'\n')
    (out/'evaluation-receipt.json').write_text(json.dumps({'status':'completed-pending-independent-explanation-review','cases_sha256':digest(args.cases),'protocol_sha256':digest(args.protocol),'freeze_sha256':digest(args.freeze),'raw_predictions_sha256':digest(out/'raw-predictions.jsonl'),'prediction_count':len(rows),'optimizer_updates':0,'source_unchanged':True,'no_retry':True},indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cases',required=True);p.add_argument('--protocol',required=True);p.add_argument('--freeze',required=True);p.add_argument('--output',required=True);p.add_argument('--endpoint',default='http://127.0.0.1:8765/api/chat');p.add_argument('--timeout',type=int,default=120);run(p.parse_args())
