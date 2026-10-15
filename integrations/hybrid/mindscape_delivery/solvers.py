"""Bounded, deterministic STEM tools. These are tools, never neural predictions."""
from __future__ import annotations
import ast
import copy
import math
import re
import sqlite3

class Unsupported(ValueError):
    pass


def _result(answer, steps, method, verification, **extra):
    def finite(v):
        if isinstance(v, float) and not math.isfinite(v):
            raise Unsupported('Nonfinite result is outside the supported domain.')
        if isinstance(v, dict):
            for item in v.values(): finite(item)
        elif isinstance(v, (list, tuple)):
            for item in v: finite(item)
    finite(answer)
    return dict(supported=True, answer=answer, steps=steps, method=method,
                verification=verification, **extra)


def _expr(text):
    import sympy as s
    if not isinstance(text, (str, int, float)) or len(str(text)) > 512:
        raise Unsupported('Expression must be at most 512 characters.')
    tree = ast.parse(str(text).strip().replace('^', '**'), mode='eval')
    if sum(1 for _ in ast.walk(tree)) > 120:
        raise Unsupported('Expression exceeds the supported complexity.')
    funcs = {n: getattr(s, n) for n in ('sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'Abs')}
    def visit(n):
        if isinstance(n, ast.Constant) and type(n.value) in (int, float):
            if not math.isfinite(n.value) or abs(n.value) > 1e12: raise Unsupported('Numeric bound exceeded.')
            return s.Rational(str(n.value))
        if isinstance(n, ast.Name) and re.fullmatch('[a-zA-Z][a-zA-Z0-9_]{0,15}', n.id):
            return {'pi': s.pi, 'E': s.E}.get(n.id, s.Symbol(n.id, real=True))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            return -visit(n.operand) if isinstance(n.op, ast.USub) else visit(n.operand)
        if isinstance(n, ast.BinOp):
            a, b = visit(n.left), visit(n.right)
            if isinstance(n.op, ast.Add): return a+b
            if isinstance(n.op, ast.Sub): return a-b
            if isinstance(n.op, ast.Mult): return a*b
            if isinstance(n.op, ast.Div):
                if b == 0: raise Unsupported('Division by zero.')
                return a/b
            if isinstance(n.op, ast.Pow):
                if b.free_symbols or abs(b) > 12: raise Unsupported('Power must be a constant with magnitude at most 12.')
                return a**b
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in funcs and len(n.args)==1 and not n.keywords:
            return funcs[n.func.id](visit(n.args[0]))
        raise Unsupported('Unsupported expression syntax.')
    result = visit(tree.body)
    if result.has(s.zoo, s.nan, s.oo, -s.oo):
        raise Unsupported('Nonfinite symbolic expression is outside the supported domain.')
    return result


def _math(spec):
    import sympy as s
    task = spec.get('task', '')
    var = _expr(spec.get('variable', 'x'))
    if not isinstance(var, s.Symbol): raise Unsupported('Variable must be a symbol.')
    if task in ('matrix', 'determinant'):
        rows = spec.get('matrix')
        if not isinstance(rows, list) or not 1 <= len(rows) <= 4 or not all(isinstance(r,list) and 1<=len(r)<=4 for r in rows) or len({len(r) for r in rows}) != 1:
            raise Unsupported('Matrix must be rectangular, at most 4 by 4.')
        m = s.Matrix([[_expr(x) for x in r] for r in rows])
        op = 'determinant' if task == 'determinant' else spec.get('operation', 'determinant')
        if op == 'determinant':
            if m.rows != m.cols: raise Unsupported('Determinant requires a square matrix.')
            answer = m.det()
            return _result(str(answer), [f'A = {m}', f'det(A) = {answer}'], 'symbolic determinant', {'passed':True,'check':'Exact symbolic matrix arithmetic.'})
        if op in ('add', 'multiply'):
            other=spec.get('matrix_b',spec.get('other_matrix'))
            if not isinstance(other,list) or not 1<=len(other)<=4 or not all(isinstance(r,list) and 1<=len(r)<=4 for r in other) or len({len(r) for r in other})!=1:
                raise Unsupported('The second matrix must be rectangular, at most 4 by 4.')
            b=s.Matrix([[_expr(x) for x in r] for r in other])
            if op=='add':
                if m.shape!=b.shape: raise Unsupported('Matrix addition requires equal dimensions.')
                answer=m+b
                check=all(s.simplify(answer[i,j]-m[i,j]-b[i,j])==0 for i in range(m.rows) for j in range(m.cols))
                rule='Add corresponding entries: C[i,j] = A[i,j] + B[i,j].'
            else:
                if m.cols!=b.rows: raise Unsupported('Matrix multiplication requires columns(A) = rows(B).')
                answer=m*b
                check=all(s.simplify(answer[i,j]-sum(m[i,k]*b[k,j] for k in range(m.cols)))==0 for i in range(m.rows) for j in range(b.cols))
                rule='Row-column product: C[i,j] = sum_k A[i,k] B[k,j].'
            return _result([[str(x) for x in row] for row in answer.tolist()], [f'A = {m}',f'B = {b}',rule,f'C = {answer}'], 'symbolic matrix algebra', {'passed':bool(check),'check':'Recompute every entry using the stated addition or row-column formula.'})
        if op == 'inverse':
            if m.rows != m.cols or m.det()==0: raise Unsupported('Inverse requires a nonsingular square matrix.')
            answer=m.inv(); passed=m*answer==s.eye(m.rows)
        elif op=='transpose': answer=m.T; passed=answer.T==m
        else: raise Unsupported('Supported matrix operations: add, multiply, determinant, inverse, transpose.')
        return _result([[str(x) for x in row] for row in answer.tolist()], [f'A = {m}', f'{op}(A) = {answer}'], 'symbolic matrix algebra', {'passed':bool(passed),'check':'Inverse product or transpose identity.'})
    raw = spec.get('expression', spec.get('equation', ''))
    if task in ('solve_equation','equation','solve'):
        parts=str(raw).split('=')
        if len(parts)>2: raise Unsupported('Provide a single equation.')
        e=_expr(parts[0]) - (_expr(parts[1]) if len(parts)==2 else 0)
        if e.free_symbols - {var}: raise Unsupported('Equation may contain only the selected variable.')
        if not e.is_polynomial(var) or s.degree(e,var)>4: raise Unsupported('Equation solver supports polynomials through degree four.')
        # Reject variable denominators before simplification can erase a hole.
        for part in parts:
            for node in ast.walk(ast.parse(part.strip().replace('^','**'), mode='eval')):
                if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div) and _expr(ast.unparse(node.right)).free_symbols:
                    raise Unsupported('Polynomial equation syntax cannot contain variable denominators.')
                if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow) and _expr(ast.unparse(node.right)).is_negative and _expr(ast.unparse(node.left)).free_symbols:
                    raise Unsupported('Polynomial equation syntax cannot contain negative variable powers.')
        if not e.free_symbols:
            if e == 0:
                return _result('all real numbers', ['Both sides simplify to the same expression.', f'Every real {var} satisfies this identity.'], 'symbolic polynomial equation solver', {'passed':True,'check':'The difference of both sides is identically zero.'}, domain='real')
            return _result([], [f'The equation simplifies to {e} = 0, which is false.'], 'symbolic polynomial equation solver', {'passed':True,'check':'A nonzero constant cannot equal zero.'}, domain='real')
        values=s.solve(e,var)
        passed=all(s.simplify(e.subs(var,v))==0 for v in values)
        return _result([str(v) for v in values], [f'Set {e} = 0.', f'Solutions for {var}: {values}'], 'symbolic polynomial equation solver', {'passed':bool(passed),'check':'Substitute every returned root into the equation.'})
    e=_expr(raw)
    if task in ('differentiate','derivative'):
        out=s.diff(e,var); check=s.simplify(out-s.diff(e,var))==0
        return _result(str(out), [f'f({var}) = {e}', f'df/d{var} = {out}', 'Real variable; derivative applies where the original expression is defined and differentiable.'], 'symbolic differentiation', {'passed':bool(check),'check':'Symbolic derivative identity (same engine, not independent). '})
    if task in ('integrate','integral'):
        out=s.integrate(e,var)
        if out.has(s.Integral): raise Unsupported('No supported closed-form antiderivative found.')
        check=s.simplify(s.diff(out,var)-e)==0
        if 'lower' in spec or 'upper' in spec:
            if 'lower' not in spec or 'upper' not in spec: raise Unsupported('Both integration bounds are required.')
            a,b=_expr(spec['lower']),_expr(spec['upper'])
            if a.free_symbols or b.free_symbols or a.is_real is not True or b.is_real is not True: raise Unsupported('Finite real constant bounds required.')
            from sympy.calculus.util import continuous_domain
            interval=s.Interval(min(a,b),max(a,b))
            domain=continuous_domain(e,var,s.S.Reals)
            if interval.is_subset(domain) is not True: raise Unsupported('Integrand is not certified continuous on the closed integration interval.')
            value=s.simplify(out.subs(var,b)-out.subs(var,a))
            return _result(str(value), [f'Antiderivative: {out}', f'F({b}) - F({a}) = {value}'], 'symbolic continuous-interval integration', {'passed':bool(check),'check':'Differentiate antiderivative; subtract endpoints.'})
        return _result(f'{out} + C', [f'Integrand: {e}', f'Antiderivative: {out} + C', 'Identity applies on intervals where the integrand and antiderivative are defined; no global continuity is asserted.'], 'symbolic integration', {'passed':bool(check),'check':'Differentiate antiderivative to recover integrand.'})
    raise Unsupported('Unsupported math task.')


_UNITS={'q1':'C','q2':'C','q':'C','r':'m','voltage':'V','current':'A','resistance':'ohm','f':'m','u':'m','emf':'V'}
def _physics(spec):
    task=spec.get('task','')
    supplied=spec.get('values',spec.get('parameters',{}))
    if not isinstance(supplied,dict): raise Unsupported('Physics values must be a mapping.')
    def value(k, positive=False):
        x=supplied.get(k,spec.get(k))
        if not isinstance(x,dict) or set(x)!={'value','unit'}:
            raise Unsupported(f'{k} requires explicit value and unit ({_UNITS.get(k,"ohm")}).')
        if x['unit']!=_UNITS.get(k,'ohm'): raise Unsupported(f'Expected {_UNITS.get(k,"ohm")} for {k}; unit conversion is not implicit.')
        v=x['value']
        if type(v) not in (int,float) or not math.isfinite(v) or abs(v)>1e12 or (positive and v<=0): raise Unsupported(f'Invalid value for {k}.')
        return float(v)
    def electrostatic_constant():
        if 'k' not in supplied and 'k' not in spec:
            return 8.9875517923e9, 'Vacuum point-charge approximation; k = 8.9875517923e9 N*m^2/C^2 (default because no k was supplied).'
        if 'k' in supplied and 'k' in spec and supplied['k']!=spec['k']:
            raise Unsupported('Conflicting electrostatic constants were supplied.')
        raw=supplied.get('k',spec.get('k'))
        if isinstance(raw,dict):
            if set(raw)!={'value','unit'} or not isinstance(raw['unit'],str): raise Unsupported('k requires a numeric value and electrostatic SI unit.')
            unit=raw['unit'].replace(' ','').replace('·','*').replace('²','^2').replace('**','^')
            if unit not in {'N*m^2/C^2','Nm^2/C^2','N*m^2*C^-2','Nm^2C^-2'}:
                raise Unsupported('k must use N*m^2/C^2 units.')
            raw=raw['value']
        if type(raw) not in (int,float) or not math.isfinite(raw) or not 0<raw<=1e12:
            raise Unsupported('k must be a finite positive electrostatic SI constant at most 1e12.')
        return float(raw), f'Using the supplied k = {raw:.12g} N*m^2/C^2.'
    def resistors():
        entries=supplied.get('resistors',spec.get('resistors'))
        if not isinstance(entries,list) or not 1<=len(entries)<=30: raise Unsupported('Supply 1–30 resistors.')
        vals=[]
        for x in entries:
            if not isinstance(x,dict) or set(x)!={'value','unit'} or x.get('unit')!='ohm' or type(x.get('value')) not in (int,float) or not 0<x['value']<=1e12:
                raise Unsupported('Every resistor requires a positive finite value in ohm.')
            vals.append(float(x['value']))
        return vals
    if task=='coulomb':
        q1,q2,r=value('q1'),value('q2'),value('r',True); k,k_step=electrostatic_constant(); answer=k*abs(q1*q2)/r**2
        return _result(answer, [k_step, 'F = k |q1 q2| / r²', f'F = {answer:.12g} N; '+('repulsive' if q1*q2>0 else 'attractive' if q1*q2<0 else 'zero force')], 'Coulomb law tool', {'passed':math.isfinite(answer),'check':'SI dimensions: (N m²/C²) C²/m² = N.'}, units='N')
    if task in ('electric_field','field'):
        q,r=value('q'),value('r',True); k,k_step=electrostatic_constant(); answer=k*q/r**2
        return _result(answer,[k_step,'E_r = k q/r²; positive is radially outward.',f'E_r = {answer:.12g} N/C'],'point-charge field tool',{'passed':math.isfinite(answer),'check':'Signed radial SI field.'},units='N/C')
    if task in ('ohm','ohms_law'):
        unknown=spec.get('unknown','current')
        if unknown=='current': a,b=value('voltage'),value('resistance',True); out=a/b; equation='I = V/R'; unit='A'
        elif unknown=='voltage': a,b=value('current'),value('resistance',True); out=a*b; equation='V = IR'; unit='V'
        elif unknown=='resistance': a,b=value('voltage'),value('current'); out=a/b if b else float('nan'); equation='R = V/I'; unit='ohm'
        else: raise Unsupported('Unknown must be current, voltage, or resistance.')
        if not math.isfinite(out) or (unknown=='resistance' and out<=0): raise Unsupported('Undefined or nonpositive resistance.')
        return _result(out,[equation,f'{unknown} = {out:.12g} {unit}'],'Ohm law tool',{'passed':True,'check':'Explicit SI units and nonzero denominator.'},units=unit)
    if task in ('resistance_series','resistance_parallel'):
        vals=resistors()
        out=sum(vals) if task.endswith('series') else 1/sum(1/v for v in vals)
        if out<=0: raise Unsupported('Equivalent resistance underflows the supported floating range.')
        return _result(out,['R_eq = Σ R_i' if task.endswith('series') else '1/R_eq = Σ 1/R_i',f'R_eq = {out:.12g} ohm'],'resistor network tool',{'passed':True,'check':'Series sum or parallel reciprocal identity.'},units='ohm')
    if task in ('kirchhoff_loop','kvl','series_loop'):
        vals=resistors()
        if 'emfs' in supplied or 'emfs' in spec:
            if 'emf' in supplied or 'emf' in spec: raise Unsupported('Specify emf or emfs, not both.')
            cells=supplied.get('emfs',spec.get('emfs'))
            if not isinstance(cells,list) or not 1<=len(cells)<=20: raise Unsupported('Supply 1–20 signed emfs in V.')
            emfs=[]
            for cell in cells:
                if not isinstance(cell,dict) or set(cell)!={'value','unit'} or cell.get('unit')!='V' or type(cell.get('value')) not in (int,float) or not math.isfinite(cell['value']) or abs(cell['value'])>1e12:
                    raise Unsupported('Each emf requires a finite signed value in V.')
                emfs.append(float(cell['value']))
            total_emf=sum(emfs)
        else: total_emf=value('emf')
        total_resistance=sum(vals); current=total_emf/total_resistance
        residual=total_emf-current*total_resistance
        return _result(current,['Single closed loop, all given resistors in series; signed emfs follow the chosen loop direction.',f'KVL: sum(emf) - I sum(R) = 0; {total_emf:.12g} - I*{total_resistance:.12g} = 0.',f'I = {current:.12g} A'], 'single-loop Kirchhoff voltage law tool', {'passed':math.isfinite(residual) and abs(residual)<=1e-10*max(1,abs(total_emf)),'check':'Substitute the current into the signed loop voltage sum.'}, units='A')
    if task in ('lens','mirror'):
        f,u=value('f'),value('u')
        if not f or not u: raise Unsupported('f and u must be nonzero signed distances.')
        denominator=1/f+(1/u if task=='lens' else -1/u)
        if denominator == 0: raise Unsupported('Image at infinity; finite answer unavailable.')
        v=1/denominator; mag=v/u if task=='lens' else -v/u
        residual=(1/v-1/u-1/f) if task=='lens' else (1/v+1/u-1/f)
        return _result({'image_distance':v,'magnification':mag},['Cartesian sign convention; distances signed along incident light.', '1/f = 1/v - 1/u' if task=='lens' else '1/f = 1/v + 1/u',f'v = {v:.12g} m; magnification = {mag:.12g}'],'thin optical element tool',{'passed':abs(residual)<=1e-10*max(1,abs(1/f),abs(1/u),abs(1/v)),'check':'Substitute image distance back into stated signed formula; relative floating tolerance 1e-10.'},units={'image_distance':'m','magnification':'dimensionless'})
    raise Unsupported('Unsupported physics task.')


def _trace(spec):
    code=spec.get('code','')
    if not isinstance(code,str) or len(code)>10000: raise Unsupported('Code must be at most 10000 characters.')
    tree=ast.parse(code); env={}; outputs=[]; events=[]; budget=[0]
    functions={}; scopes=[env]; local_names=[set()]
    class ReturnValue(Exception):
        def __init__(self,value): self.value=value
    def validate_tree(v, ancestors=(), depth=0):
        if depth>20: raise Unsupported('Nested value depth exceeds 20.')
        if isinstance(v,(list,tuple)):
            if id(v) in ancestors: raise Unsupported('Cyclic containers are unsupported.')
            if len(v)>500: raise Unsupported('Container length exceeds 500.')
            for item in v: validate_tree(item,ancestors+(id(v),),depth+1)
    def assign(target,value):
        validate_tree(value)
        if isinstance(target,ast.Name):
            if target.id in functions: raise Unsupported('Rebinding function names is unsupported.')
            scopes[-1][target.id]=value
        elif isinstance(target,ast.Subscript):
            container=expr(target.value); index=expr(target.slice)
            if not isinstance(container,list) or type(index)!=int: raise Unsupported('Indexed assignment requires a list and integer index.')
            previous=container[index]; container[index]=value
            try: validate_tree(container)
            except Unsupported:
                container[index]=previous; raise
        else: raise Unsupported('Assignment requires a name or list index.')

    def tick():
        budget[0]+=1
        if budget[0]>2000: raise Unsupported('Interpreter exceeded 2000 operations.')
    def bounded(v):
        if type(v) not in (int,float,bool,str,list,tuple,type(None)): raise Unsupported('Unsupported value type.')
        if isinstance(v,(str,list,tuple)) and len(v)>500: raise Unsupported('Value length exceeded 500.')
        if type(v) in (int,float) and (not math.isfinite(v) or abs(v)>1e12): raise Unsupported('Numeric bound exceeded.')
        return v
    def expr(n):
        tick()
        if isinstance(n,ast.Constant): return bounded(n.value)
        if isinstance(n,ast.Name):
            if n.id in scopes[-1]: return scopes[-1][n.id]
            if n.id in local_names[-1]: raise Unsupported(f'Local name read before assignment: {n.id}')
            if n.id in env: return env[n.id]
            raise Unsupported(f'Unbound name: {n.id}')
        if isinstance(n,(ast.List,ast.Tuple)):
            values=[expr(x) for x in n.elts]
            return bounded(tuple(values) if isinstance(n,ast.Tuple) else values)
        if isinstance(n,ast.UnaryOp) and isinstance(n.op,(ast.USub,ast.UAdd,ast.Not)):
            v=expr(n.operand); return bounded(-v if isinstance(n.op,ast.USub) else +v if isinstance(n.op,ast.UAdd) else not v)
        if isinstance(n,ast.BinOp):
            a,b=expr(n.left),expr(n.right)
            if isinstance(n.op,ast.Add): v=a+b
            elif isinstance(n.op,ast.Sub): v=a-b
            elif isinstance(n.op,ast.Mult):
                if isinstance(a,(str,list,tuple)) and isinstance(b,int) and len(a)*max(b,0)>500: raise Unsupported('Sequence allocation bound exceeded.')
                if isinstance(b,(str,list,tuple)) and isinstance(a,int) and len(b)*max(a,0)>500: raise Unsupported('Sequence allocation bound exceeded.')
                v=a*b
            elif isinstance(n.op,ast.Div): v=a/b
            elif isinstance(n.op,ast.FloorDiv): v=a//b
            elif isinstance(n.op,ast.Mod): v=a%b
            elif isinstance(n.op,ast.Pow):
                if type(b) not in (int,float) or abs(b)>12: raise Unsupported('Exponent bound exceeded.')
                v=a**b
            else: raise Unsupported('Unsupported operator.')
            return bounded(v)
        if isinstance(n,ast.Compare):
            a=expr(n.left)
            for op,right in zip(n.ops,n.comparators):
                b=expr(right)
                if isinstance(op,ast.Eq): ok=a==b
                elif isinstance(op,ast.NotEq): ok=a!=b
                elif isinstance(op,ast.Lt): ok=a<b
                elif isinstance(op,ast.LtE): ok=a<=b
                elif isinstance(op,ast.Gt): ok=a>b
                elif isinstance(op,ast.GtE): ok=a>=b
                else: raise Unsupported('Unsupported comparison.')
                if not ok:return False
                a=b
            return True
        if isinstance(n,ast.BoolOp):
            v=expr(n.values[0])
            for child in n.values[1:]:
                if (isinstance(n.op,ast.And) and not v) or (isinstance(n.op,ast.Or) and v): break
                v=expr(child)
            return v
        if isinstance(n,ast.Subscript): return bounded(expr(n.value)[expr(n.slice)])
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and not n.keywords and n.func.attr in ('append','pop'):
            container=expr(n.func.value)
            if not isinstance(container,list): raise Unsupported('append/pop require a list.')
            args=[expr(x) for x in n.args]
            if n.func.attr=='append' and len(args)==1:
                if len(container)>=500: raise Unsupported('List length bound exceeded.')
                container.append(args[0])
                try: validate_tree(container)
                except Unsupported:
                    container.pop(); raise
                return None
            if n.func.attr=='pop' and len(args)<=1 and (not args or type(args[0])==int): return container.pop(*args)
            raise Unsupported('Invalid append/pop arguments.')
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and not n.keywords:

            name=n.func.id; args=[expr(x) for x in n.args]
            if name=='range' and 1<=len(args)<=3 and all(type(x)==int for x in args):
                r=range(*args)
                if len(r)>500: raise Unsupported('Range bound exceeded.')
                return list(r)
            if name=='len' and len(args)==1:return len(args[0])
            if name=='abs' and len(args)==1:return bounded(abs(args[0]))
            if name in functions:
                if name in scopes[-1] or name in env: raise Unsupported('Calling a shadowed function name is unsupported.')
                node,params,defaults,names=functions[name]
                if not len(params)-len(defaults)<=len(args)<=len(params): raise Unsupported('Function positional argument count mismatch.')
                if len(scopes)>12: raise Unsupported('Function call depth exceeds 12.')
                bound=dict(zip(params,args))
                for key,value in zip(params[len(params)-len(defaults):],defaults):
                    if key not in bound: bound[key]=value
                scopes.append(bound);local_names.append(names)
                try:
                    try: block(node.body)
                    except ReturnValue as ret: return bounded(ret.value)
                    return None
                finally: scopes.pop();local_names.pop()

        raise Unsupported('Supported code: assignments, arithmetic, comparisons, if, bounded loops, top-level functions, return, print, len, abs, list indexing/append/pop. Imports and arbitrary attributes are unavailable.')
    def block(nodes):
        for n in nodes:
            tick()
            if isinstance(n,ast.FunctionDef):
                if len(scopes)!=1 or n.decorator_list or n.args.vararg or n.args.kwarg or n.args.kwonlyargs or n.returns or any(a.annotation for a in n.args.posonlyargs+n.args.args):
                    raise Unsupported('Functions must be top-level, undecorated, with positional parameters only.')
                params=[a.arg for a in n.args.posonlyargs+n.args.args]
                if len(params)>8 or n.name in env or n.name in ('range','len','abs','print'): raise Unsupported('Unsupported function name or parameter count.')
                defaults=[expr(x) for x in n.args.defaults]
                names=set(params)|{a.id for child in n.body for a in ast.walk(child) if isinstance(a,ast.Name) and isinstance(a.ctx,ast.Store)}
                functions[n.name]=(n,params,defaults,names)
            elif isinstance(n,ast.Return):
                if len(scopes)==1: raise Unsupported('Return outside a function.')
                raise ReturnValue(expr(n.value) if n.value else None)
            elif isinstance(n,ast.Assign) and len(n.targets)==1: assign(n.targets[0],expr(n.value))
            elif isinstance(n,ast.AugAssign): assign(n.target,expr(ast.BinOp(left=n.target,op=n.op,right=n.value)))
            elif isinstance(n,ast.If): block(n.body if expr(n.test) else n.orelse)
            elif isinstance(n,ast.For) and isinstance(n.target,ast.Name) and not n.orelse:
                for v in expr(n.iter): assign(n.target,v);block(n.body)
            elif isinstance(n,ast.While) and not n.orelse:
                while expr(n.test): tick();block(n.body)
            elif isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='print' and not n.value.keywords:
                outputs.append(' '.join(str(expr(x)) for x in n.value.args))
                if len(outputs)>100: raise Unsupported('Output bound exceeded.')
            elif isinstance(n,ast.Expr) and isinstance(n.value,ast.Call): expr(n.value)
            elif isinstance(n,ast.Try):
                allowed={'ZeroDivisionError':ZeroDivisionError,'IndexError':IndexError,'ValueError':ValueError,'TypeError':TypeError}
                if any(h.name or not isinstance(h.type,ast.Name) or h.type.id not in allowed for h in n.handlers):raise Unsupported('Only explicit supported exception types without aliases.')
                try:
                    try: block(n.body)
                    except (ZeroDivisionError,IndexError,ValueError,TypeError) as exc:
                        if isinstance(exc,Unsupported):raise
                        handler=next((h for h in n.handlers if isinstance(exc,allowed[h.type.id])),None)
                        if handler is None:raise
                        block(handler.body)
                    else:block(n.orelse)
                finally:block(n.finalbody)
            elif isinstance(n,ast.Pass): pass
            else: raise Unsupported('Unsupported Python statement.')
            if len(events)<80: events.append({'line':n.lineno,'variables':copy.deepcopy(scopes[-1]),'output_count':len(outputs),'call_depth':len(scopes)-1})
    block(tree.body)
    return _result('\n'.join(outputs),['Executed the supported AST subset in a bounded interpreter.'], 'bounded AST interpreter',{'passed':True,'check':'No exec/eval, imports, file access, network, or arbitrary attributes; only list append/pop methods. Functions have bounded call depth and local scopes.','operations':budget[0]}, variables=env, trace=events)


def _sql(spec):
    query=spec.get('query',''); tables=spec.get('tables',{})
    if not isinstance(query,str) or len(query)>4000 or not re.match(r'^\s*SELECT\b',query,re.I): raise Unsupported('Only a single SELECT query is supported.')
    if not isinstance(tables,dict) or not 1<=len(tables)<=5: raise Unsupported('Supply 1–5 in-memory tables.')
    conn=sqlite3.connect(':memory:')
    try:
        for name, rows in tables.items():
            if not re.fullmatch('[A-Za-z][A-Za-z0-9_]{0,30}',name) or not isinstance(rows,list) or not 1<=len(rows)<=500 or not all(isinstance(r,dict) for r in rows): raise Unsupported('Invalid table name or rows (1–500 required).')
            columns=list(rows[0])
            if not 1<=len(columns)<=20 or not all(re.fullmatch('[A-Za-z][A-Za-z0-9_]{0,30}',c) for c in columns): raise Unsupported('Invalid column identifiers.')
            if not all(set(r)==set(columns) for r in rows):raise Unsupported('Rows must share column names.')
            values=[[r[c] for c in columns] for r in rows]
            if any(type(v) not in (str,int,float,bool,type(None)) or (isinstance(v,str) and len(v)>1000) for row in values for v in row):raise Unsupported('Unsupported SQL cell value.')
            conn.execute(f'CREATE TABLE "{name}" ('+', '.join(f'"{c}"' for c in columns)+')')
            conn.executemany(f'INSERT INTO "{name}" VALUES ('+','.join('?' for _ in columns)+')',values)
        allowed={sqlite3.SQLITE_SELECT,sqlite3.SQLITE_READ,sqlite3.SQLITE_FUNCTION}
        def authorize(action,a,b,db,source):
            if action==sqlite3.SQLITE_FUNCTION and (b or a or '').lower() not in {'abs','lower','upper','count','sum','avg','min','max','round','length','substr','substring','coalesce','ifnull','nullif','trim','ltrim','rtrim','replace','like','glob','typeof','total','instr'}: return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY
        conn.set_authorizer(authorize)
        count=[0]
        def progress():
            count[0]+=1
            return int(count[0]>1000)
        conn.set_progress_handler(progress,100)
        cur=conn.execute(query); rows=cur.fetchmany(501)
        if len(rows)>500:raise Unsupported('Result exceeds 500 rows.')
        return _result({'columns':[x[0] for x in cur.description], 'rows':[list(x) for x in rows]},['Loaded provided rows into an isolated in-memory database.','Executed one read-only SELECT with instruction and row limits.'],'restricted in-memory SQLite tool',{'passed':True,'check':'SQLite authorizer blocks writes, attachment and PRAGMA; bound progress and rows.'})
    finally:conn.close()


def solve(spec:dict)->dict:
    """Return explicit unsupported status; never substitute an unrelated answer."""
    try:
        if not isinstance(spec,dict): raise Unsupported('Task must be a mapping.')
        from .advanced import advanced
        extra=advanced(spec)
        if extra is not None:return extra
        subject=spec.get('subject',spec.get('domain',''))
        if subject in ('math','mathematics'): return _math(spec)
        if subject=='physics': return _physics(spec)
        if subject in ('coding','python'): return _trace(spec)
        if subject=='sql': return _sql(spec)
        raise Unsupported('Supported subjects: math, physics, coding, sql.')
    except (Unsupported,ValueError,TypeError,SyntaxError,ZeroDivisionError,OverflowError,IndexError,KeyError,RecursionError,sqlite3.Error,NotImplementedError) as exc:
        return dict(supported=False,answer=None,steps=[],verification={'passed':False,'check':'Unsupported or invalid input.'},method='no solution',reason=str(exc))
