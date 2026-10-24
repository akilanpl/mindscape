"""Conservative supplied grammars used to validate neural interpretations.

These rules are software, not capabilities learned by native R6 or Qwen.
"""
import json,re

def _expr(text):
    text=text.strip().rstrip('?.').strip().replace('−','-').replace('×','*')
    text=re.sub(r'(?<=\d)(?=[xyz]\b)','*',text)
    return text

def supplied_parse(question):
    q=question.strip();low=q.lower()
    if len(q)>6000:return None
    # Explicit structured submission is also useful for the offline fallback.
    if q.startswith('{'):
        try:
            spec=json.loads(q)
            return spec if isinstance(spec,dict) and 'subject' in spec and 'task' in spec else None
        except ValueError:return None
    block=re.search(r'```(?:python|py)?\s*\n(.*?)```',q,re.S|re.I)
    if block and re.search(r'output|trace|print|execute|run|predict',low):
        return {'subject':'coding','task':'trace','code':block[1]}
    if q.startswith(('print(','def ','for ','if ')) or '\n' in q and re.match(r'^[a-zA-Z_]\w*\s*=',q):
        return {'subject':'coding','task':'trace','code':q}
    if re.search(r'\b(select)\b',low):
        # Table data must be explicit; natural-language database invention is forbidden.
        match=re.search(r'\b(SELECT\b.*?)\s+(?:with\s+)?tables?\s*[:=]\s*(\{.*\})\s*$',q,re.I|re.S)
        if match:
            try:return {'subject':'sql','task':'select','query':match[1].strip().rstrip(';'),'tables':json.loads(match[2])}
            except ValueError:return None
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
    if any(word in low for word in ('determinant','inverse','transpose')):
        start=q.find('[[')
        if start>=0:
            try:matrix,offset=json.JSONDecoder().raw_decode(q[start:])
            except ValueError:return None
            op=next(x for x in ('determinant','inverse','transpose') if x in low)
            return {'subject':'math','task':'determinant' if op=='determinant' else 'matrix','operation':op,'matrix':matrix}
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
    if re.search(r'coulomb|force.*charg|charg.*force',low) and len(by['C'])==2 and len(by['m'])==1:
        return {'subject':'physics','task':'coulomb','values':{'q1':by['C'][0],'q2':by['C'][1],'r':by['m'][0]}}
    if 'electric field' in low and len(by['C'])==1 and len(by['m'])==1:
        return {'subject':'physics','task':'electric_field','values':{'q':by['C'][0],'r':by['m'][0]}}
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
