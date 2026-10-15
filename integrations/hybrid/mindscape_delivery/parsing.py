"""Conservative supplied grammars used to validate neural interpretations.

These rules are software, not capabilities learned by native R6 or Qwen.
"""
import ast,json,re

def _expr(text):
    text=text.strip().rstrip('?.').strip().replace('−','-').replace('×','*')
    text=re.sub(r'(?<=\d)(?=[xyz]\b)','*',text)
    return text

def _python_source(question):
    """Extract literal submitted code, never a model-written replacement."""
    block=re.search(r'```(?:python|py)?\s*\n(.*?)```',question,re.S|re.I)
    if block:return block[1]
    if not re.search(r'output|print|python|program|trace|code',question,re.I):return None
    lines=question.splitlines(keepends=True)
    marker=re.compile(r'^(?:def\s|for\s|while\s|if\s|import\s|from\s|class\s|try:|with\s|print\s*\(|[A-Za-z_]\w*\s*(?:=|\+=|-=|\*=))')
    for i,line in enumerate(lines):
        if not marker.match(line):continue
        # Keep all code from its first statement, including rejected statements.
        # A question suffix may be removed only at an explicit blank boundary.
        source=''.join(lines[i:]).strip()
        try:ast.parse(source);return source
        except SyntaxError:
            for j in range(len(lines)-1,i,-1):
                if lines[j-1].strip():continue
                prefix=''.join(lines[i:j-1]).strip()
                try:ast.parse(prefix);return prefix
                except SyntaxError:pass
            return None
    inline=re.search(r'(?:output|trace|execute|program).*?:\s*(.+)$',question,re.I)
    if inline:
        try:ast.parse(inline[1]);return inline[1]
        except SyntaxError:pass
    return None

def _sql_source(question):
    """Read one literal SELECT and explicit JSON or tuple-table data."""
    if not re.search(r'\bSELECT\b',question,re.I):return None
    # Decode literal table mapping independently of question prose ordering.
    data=re.search(r'\btables?\s*[:=]\s*(?=\{)',question,re.I)
    if data:
        try:
            tables,used=json.JSONDecoder().raw_decode(question[data.end():])
            remainder=question[:data.start()]+' '+question[data.end()+used:]
            query=re.search(r'\bSELECT\b[^;?`]+',remainder,re.I)
            if query and isinstance(tables,dict):
                text=re.sub(r'\s+(?:with\s+)?$', '',query[0]).strip()
                return {'subject':'sql','task':'select','query':text,'tables':tables}
        except ValueError:pass
    old=re.search(r'\b(SELECT\b.*?)\s+(?:with\s+)?tables?\s*[:=]\s*(\{.*\})\s*$',question,re.I|re.S)
    if old:
        try:return {'subject':'sql','task':'select','query':old[1].strip().rstrip(';'),'tables':json.loads(old[2])}
        except ValueError:return None
    query_match=re.search(r'\bSELECT\b[^;?`]+',question,re.I)
    if not query_match:return None
    query=query_match[0].strip()
    query=re.split(r'\n\s*(?:table|data|rows)\b|\s+(?:with\s+)?tables?\s*:',query,flags=re.I)[0].strip()
    query=re.sub(r'\s+(?:return|returns|produce|produces)\s*$', '',query,flags=re.I).strip()
    # Explicit JSON record table, e.g. Table scores: [{"name":"A",...}].
    for match in re.finditer(r'\btable\s+([A-Za-z_]\w*)\s*[:=]\s*',question,re.I):
        try:
            rows,_=json.JSONDecoder().raw_decode(question[match.end():])
            if isinstance(rows,list) and all(isinstance(r,dict) for r in rows):return {'subject':'sql','task':'select','query':query,'tables':{match[1]:rows}}
        except ValueError:pass
    table=re.search(r'\btable\s+([A-Za-z_]\w*)\s*\(([^()]+)\)\s*(?:with\s+)?(?:rows|data)?\s*[:=]?\s*',question,re.I)
    if not table:return None
    columns=[x.strip() for x in table[2].split(',')]
    if not columns or any(not re.fullmatch(r'[A-Za-z_]\w*',x) for x in columns):return None
    tail=question[table.end():]
    tail=re.split(r'\bSELECT\b|\b(?:What|Which|Execute|Run|Query)\b',tail,flags=re.I)[0].strip().rstrip('.;')
    try:
        # literal_eval accepts constants/tuples/lists only, not names/calls/code.
        candidate=tail if tail.startswith('[') else '['+tail+']'
        if len(candidate)>12000:return None
        tree=ast.parse(candidate,mode='eval')
        if sum(1 for _ in ast.walk(tree))>2000:return None
        rows=ast.literal_eval(tree)
        if not isinstance(rows,(list,tuple)) or not 1<=len(rows)<=200:return None
        if any(not isinstance(r,(list,tuple)) or len(r)!=len(columns) for r in rows):return None
        return {'subject':'sql','task':'select','query':query,'tables':{table[1]:[dict(zip(columns,row)) for row in rows]}}
    except (ValueError,SyntaxError,TypeError):return None

def _explicit_constant(question):
    """Restricted numeric k declaration; malformed declarations are not ignored."""
    m=re.search(r'\b(?:k|coulomb(?:\'s)? constant|electrostatic constant)\s*(?:=|:|is)\s*([0-9.eE+\-*/^() ×x]+)',question,re.I)
    if not m:
        return None, bool(re.search(r'\b(?:k\s*[:=]|coulomb(?:\'s)? constant|electrostatic constant)',question,re.I))
    suffix=question[m.end():].lstrip()
    if suffix and suffix[0] in '⁰¹²³⁴⁵⁶⁷⁸⁹\\*/^×':return None,True
    if re.match(r'(?:N|C|V|ohm)(?:\b|[/*·])',suffix):
        if not re.match(r'N(?:[ *·]*m(?:\^?2|²))/(?:C(?:\^?2|²))(?=$|[\s,;.)])',suffix):return None,True
    text=m[1].strip().replace('×','*').replace('x','*').replace('^','**')
    try:
        tree=ast.parse(text,mode='eval')
        def visit(n):
            if isinstance(n,ast.Constant) and type(n.value) in (int,float):return float(n.value)
            if isinstance(n,ast.UnaryOp) and isinstance(n.op,(ast.USub,ast.UAdd)):return -visit(n.operand) if isinstance(n.op,ast.USub) else visit(n.operand)
            if isinstance(n,ast.BinOp):
                a,b=visit(n.left),visit(n.right)
                if isinstance(n.op,ast.Mult):return a*b
                if isinstance(n.op,ast.Pow) and abs(b)<=12:return a**b
            raise ValueError('Numeric constant only')
        value=visit(tree.body)
        if not 0<value<=1e12:raise ValueError('constant bound')
        return {'value':value,'unit':'N*m^2/C^2'},False
    except (ValueError,SyntaxError,OverflowError):return None,True

def supplied_parse(question):
    q=question.strip();low=q.lower()
    if len(q)>6000:return None
    # Explicit structured submission is also useful for the offline fallback.
    if q.startswith('{'):
        try:
            spec=json.loads(q)
            return spec if isinstance(spec,dict) and 'subject' in spec and 'task' in spec else None
        except ValueError:return None
    code=_python_source(q)
    if code is not None:return {'subject':'coding','task':'trace','code':code}
    sql=_sql_source(q)
    if sql is not None:return sql
    match=re.match(r'^(?:please\s+)?(?:find\s+(?:the\s+)?)?(differentiate|derivative\s+of|integrate|integral\s+of|solve)\s+(.+)$',q,re.I|re.S)
    if match:
        command=match[1].lower();expression=match[2].strip();variable='x';bounds={}
        expression=re.sub(r'^(?:the\s+)?equation\s+','',expression,flags=re.I)
        if command!='solve':expression=re.sub(r'^[fy]\s*\(\s*[a-z]\s*\)\s*=\s*','',expression,flags=re.I)
        m=re.search(r'\s+(?:with\s+respect\s+to|w\.?r\.?t\.?|for)\s+([a-z])\s*[?.]?$',expression,re.I)
        if m:variable=m[1];expression=expression[:m.start()]
        m=re.search(r'\s+from\s+(-?\d+(?:\.\d+)?)\s+to\s+(-?\d+(?:\.\d+)?)\s*[?.]?$',expression,re.I)
        if m:bounds={'lower':m[1],'upper':m[2]};expression=expression[:m.start()]
        expression=re.sub(r'\s+d([a-z])\s*[?.]?$',lambda m:'',expression)
        task='solve_equation' if command=='solve' else 'integrate' if command.startswith('integ') else 'differentiate'
        return {'subject':'math','task':task,'expression':_expr(expression),'variable':variable,**bounds}
    if any(word in low for word in ('determinant','inverse','transpose','matrices','matrix')):
        matrices=[];offset=0
        while (start:=q.find('[[',offset))>=0:
            try:matrix,used=json.JSONDecoder().raw_decode(q[start:])
            except ValueError:return None
            matrices.append(matrix);offset=start+used
        if matrices:
            op='multiply' if re.search(r'multipl|product|matrix.*@',low) else 'add' if re.search(r'\badd|sum',low) else next((x for x in ('determinant','inverse','transpose') if x in low),None)
            if op in ('add','multiply'):
                if len(matrices)!=2:return None
                return {'subject':'math','task':'matrix','operation':op,'matrix':matrices[0],'matrix_b':matrices[1]}
            if op and len(matrices)==1:return {'subject':'math','task':'determinant' if op=='determinant' else 'matrix','operation':op,'matrix':matrices[0]}
    # Physical quantities stay dimensional: no unlabeled numeric guessing.
    number=r'[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?'
    units=r'(?:microcoulombs?|microcoulomb|uC|μC|µC|nC|mC|coulombs?|C|kilohms?|kohm|ohms?|Ω|kΩ|volts?|V|milliamps?|mA|amps?|amperes?|A|centimeters?|cm|millimeters?|mm|meters?|metres?|m)'
    items=[]
    for m in re.finditer(r'('+number+r')\s*('+units+r')(?![a-zA-Z])',q,re.I):
        val=float(m[1]);u=m[2];ul=u.lower()
        if ul in ('uc','μc','µc') or ul.startswith('microcoulomb'):val*=1e-6;unit='C'
        elif u=='nC':val*=1e-9;unit='C'
        elif u=='mC':val*=1e-3;unit='C'
        elif ul in ('c','coulomb','coulombs'):unit='C'
        elif ul in ('kohm','kω') or ul.startswith('kilohm'):val*=1000;unit='ohm'
        elif ul in ('ohm','ohms','ω'):unit='ohm'
        elif ul in ('v','volt','volts'):unit='V'
        elif ul in ('ma','milliamp','milliamps'):val*=.001;unit='A'
        elif ul in ('a','amp','amps','ampere','amperes'):unit='A'
        elif ul in ('cm','centimeter','centimeters'):val*=.01;unit='m'
        elif ul in ('mm','millimeter','millimeters'):val*=.001;unit='m'
        else:unit='m'
        items.append(({'value':val,'unit':unit},m.start(),m.end()))
    by={u:[v for v,_,_ in items if v['unit']==u] for u in ('C','m','V','A','ohm')}
    if re.search(r'coulomb|electrostatic force|force.*charg|charg.*force',low) and len(by['C'])==2 and len(by['m'])==1:
        k,bad=_explicit_constant(q)
        if bad:return None
        return {'subject':'physics','task':'coulomb','values':{'q1':by['C'][0],'q2':by['C'][1],'r':by['m'][0],**({'k':k} if k else {})}}
    if 'electric field' in low and len(by['C'])==1 and len(by['m'])==1:
        k,bad=_explicit_constant(q)
        if bad:return None
        return {'subject':'physics','task':'electric_field','values':{'q':by['C'][0],'r':by['m'][0],**({'k':k} if k else {})}}
    if 'current' in low and re.search(r'kirchhoff|loop|series',low) and len(by['V'])==1 and len(by['ohm'])>=1:
        return {'subject':'physics','task':'kirchhoff_loop','values':{'emf':by['V'][0],'resistors':by['ohm']}}
    if 'resistan' in low and ('series' in low or 'parallel' in low) and len(by['ohm'])>=2:
        return {'subject':'physics','task':'resistance_parallel' if 'parallel' in low else 'resistance_series','values':{'resistors':by['ohm']}}
    if 'lens' in low or 'mirror' in low:
        vals={}
        for v,start,end in items:
            if v['unit']!='m':continue
            label=re.search(r'\b(f|u|focal length|object distance)\s*(?:=|:|of|is)?\s*$',q[:start],re.I)
            if label:vals[{'focal length':'f','object distance':'u'}.get(label[1].lower(),label[1].lower())]=v
        if set(vals)=={'f','u'}:
            return {'subject':'physics','task':'mirror' if 'mirror' in low else 'lens','values':vals}
        return None # Do not silently choose a sign convention for optics.
    if any(w in low for w in ('current','voltage','ohm','resistance')):
        possible=[('current','A',{'voltage':by['V'],'resistance':by['ohm']}),('voltage','V',{'current':by['A'],'resistance':by['ohm']}),('resistance','ohm',{'voltage':by['V'],'current':by['A']})]
        for unknown,unit,vals in possible:
            if not by[unit] and all(len(v)==1 for v in vals.values()):
                return {'subject':'physics','task':'ohm','unknown':unknown,'values':{k:v[0] for k,v in vals.items()}}
    return None

def ground_neural(question,plan):
    """Allow only exact expression/code/matrix copies for out-of-grammar plans.

    Complex physics and database interpretations require supplied grammar or an
    explicit structured problem; model-generated numeric interpretations are
    never silently trusted.
    """
    if not plan.get('valid'):return None
    p={k:v for k,v in plan.items() if k not in ('valid','raw','method','parse_error')}
    subject=p.get('subject');task=p.get('task')
    if subject=='math' and task in ('differentiate','integrate','solve_equation'):
        expression=p.get('expression')
        if isinstance(expression,str) and len(expression)>1 and expression in question:
            expected={'differentiate':r'differentiat|derivative','integrate':r'integrat|integral','solve_equation':r'solve|equation'}[task]
            if not re.search(expected,question,re.I):return None
            variable=p.get('variable','x')
            if not isinstance(variable,str) or not re.fullmatch('[a-zA-Z]',variable) or not re.search(r'\b'+re.escape(variable)+r'\b',question):return None
            # Definite-integral endpoints require the supplied explicit grammar;
            # a neural model cannot invent or reorder integration bounds.
            if 'lower' in p or 'upper' in p:return None
            if set(p)-{'subject','task','expression','variable'}:return None
            return p
    if subject=='coding' and task=='trace' and isinstance(p.get('code'),str) and len(p['code'])>3 and p['code'] in question:return p
    return None
