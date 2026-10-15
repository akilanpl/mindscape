import ast,json,math,re,time,sys,urllib.request
from pathlib import Path

def safe_symbolic(text):
    """Restricted AST to SymPy; no eval, arbitrary attributes, imports or strings."""
    import sympy as s
    text=str(text).strip().replace('^','**').replace('−','-').replace('π','pi')
    if len(text)>300:raise ValueError('Expression too long')
    functions={'sin':s.sin,'cos':s.cos,'tan':s.tan,'exp':s.exp,'log':s.log,'sqrt':s.sqrt}
    def visit(n):
        if isinstance(n,ast.Constant) and isinstance(n.value,(int,float)) and not isinstance(n.value,bool):return s.sympify(n.value)
        if isinstance(n,ast.Name) and (re.fullmatch('[a-z][a-z0-9_]{0,15}',n.id) or n.id in ('C','E')):return {'C':s.Symbol('C'),'pi':s.pi,'E':s.E}.get(n.id,s.Symbol(n.id))
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
            if kind=='antiderivative':
                variables=[v for v in e.free_symbols if str(v)!='C']
                if len(variables)!=1:return False
                return s.simplify(s.diff(a-e,variables[0]))==0
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

def grade(c,r):
 a=candidate_answer(r)
 if c['answer_kind']=='mapping':
  try:return set(a)==set(c['expected']) and all(answer_matches({'answer_kind':'symbolic','expected':v},a[k]) for k,v in c['expected'].items())
  except:return False
 if c['answer_kind']=='vector':
  try:return len(a)==len(c['expected']) and all(math.isclose(float(x),y,rel_tol=1e-5,abs_tol=1e-8) for x,y in zip(a,c['expected']))
  except:return False
 # Neural answers: accept only a literal final Answer/boxed line, never target searching.
 if r.get('status')=='unverified':
  raw=r.get('neural',{}).get('raw_answer','');matches=re.findall(r'(?:Answer|Final answer)\s*:\s*([^\n]+)',raw,re.I)
  if matches:a=matches[-1].strip().strip('`$').rstrip('.')
 if c['answer_kind']=='numeric' and isinstance(a,str):
  try:
   symbolic=safe_symbolic(a)
   if not symbolic.free_symbols:a=float(symbolic)
  except Exception:pass
 return bool(answer_matches(c,a))

if __name__=='__main__':
 cases=json.loads(Path(sys.argv[1]).read_text());out=Path(sys.argv[2]);port=int(sys.argv[3]);mode=sys.argv[4];out.parent.mkdir(parents=True,exist_ok=True)
 assert not out.exists(),'No evaluation overwrite'
 with out.open('x') as f:
  for c in cases:
   t=time.perf_counter();req=urllib.request.Request(f'http://127.0.0.1:{port}/api/chat',data=json.dumps({'message':c['question'],'mode':mode}).encode(),headers={'Content-Type':'application/json'})
   try:r=json.load(urllib.request.urlopen(req,timeout=120))
   except Exception as e:r={'status':'error','error':str(e)}
   row={'case':c,'response':r,'correct':grade(c,r),'latency':time.perf_counter()-t};f.write(json.dumps(row)+'\n');f.flush();time.sleep(.15)
 print(out)
