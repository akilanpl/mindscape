"""Candidate C bounded additions; explicit schemas, units and residual checks.

No filesystem or network execution. These are deterministic tools, not neural learning.
"""
import csv,io,math,re
from .solvers import Unsupported,_result,_expr

def advanced(spec):
 s=spec.get('subject');t=spec.get('task')
 if s=='math' and t in ('simultaneous','probability','linear_ode'):return math_tool(spec)
 if s=='physics' and t in ('field_vector','induction','photoelectric','dc_network'):return physics_tool(spec)
 if s=='coding' and t=='csv_summary':return csv_tool(spec)
 return None

def math_tool(p):
 import sympy as s
 t=p['task']
 if t=='probability':
  def prob(v):
   x=_expr(v)
   if x.free_symbols or x.is_real is not True or not 0<=x<=1:raise Unsupported('Probabilities must be real numbers in [0,1].')
   return x
  if p.get('operation')=='conditional':
   both,given=prob(p['intersection']),prob(p['given'])
   if given==0 or both>given:raise Unsupported('Require 0 <= intersection <= positive conditioning probability.')
   out=both/given;steps=[f'P(A|B) = P(A intersection B)/P(B) = ({both})/({given}) = {out}.']
  elif p.get('operation')=='bayes':
   pri=p['priors'];like=p['likelihoods'];i=p['index']
   if not isinstance(pri,list) or not isinstance(like,list) or not 1<=len(pri)==len(like)<=20 or type(i)!=int or not 0<=i<len(pri):raise Unsupported('Bounded equal probability arrays and valid index required.')
   pri=list(map(prob,pri));like=list(map(prob,like))
   if sum(pri)!=1:raise Unsupported('Exhaustive disjoint prior probabilities must sum to one.')
   den=sum(a*b for a,b in zip(pri,like))
   if den==0:raise Unsupported('Evidence probability must be positive.')
   out=pri[i]*like[i]/den;steps=[f'P(E) = sum(P(H_i)P(E|H_i)) = {den}.',f'P(H_{i}|E) = {pri[i]}*{like[i]}/{den} = {out}.']
  else:raise Unsupported('Probability supports conditional and Bayes with explicit probabilities.')
  return _result(str(out),steps,'exact probability tool',{'passed':bool(0<=out<=1),'check':'Exact probability normalization and bounds; supplied event assumptions only.'})
 if t=='simultaneous':
  names=p['variables'];eqs=p['equations']
  if not isinstance(names,list) or not 1<=len(names)<=4 or len(set(names))!=len(names) or not all(isinstance(v,str) and re.fullmatch('[a-z]',v) for v in names) or not isinstance(eqs,list) or len(eqs)!=len(names):raise Unsupported('Square linear system, 1–4 distinct single-letter variables required.')
  vs=[_expr(v) for v in names];es=[]
  for eq in eqs:
   if not isinstance(eq,str) or eq.count('=')!=1:raise Unsupported('Each equation requires one equality.')
   a,b=eq.split('=');e=_expr(a)-_expr(b)
   if e.free_symbols-set(vs):raise Unsupported('Undeclared system variable.')
   es.append(e)
  A,b=s.linear_eq_to_matrix(es,vs)
  if A.det()==0:raise Unsupported('System has no unique solution; singular systems are unsupported.')
  x=A.inv()*b;passed=all(s.simplify(e.subs(dict(zip(vs,x))))==0 for e in es)
  return _result(dict(zip(names,map(str,x))),[f'A = {A}; b = {b}',f'Unique solution: {dict(zip(names,map(str,x)))}','Substitute into every original equation.'],'linear system tool',{'passed':bool(passed),'check':'Exact residuals in every supplied equation.'})
 v=_expr(p.get('variable','x'));a=_expr(p['coefficient']);f=_expr(p['forcing']);x0=_expr(p['initial_x']);y0=_expr(p['initial_y'])
 if not isinstance(v,s.Symbol) or a.free_symbols or a.is_real is not True or x0.free_symbols or y0.free_symbols or x0.is_real is not True or y0.is_real is not True or not f.is_polynomial(v):raise Unsupported('ODE supports y prime + constant*y = polynomial with finite real initial data.')
 factor=s.exp(a*v);F=s.integrate(factor*f,v)
 if F.has(s.Integral):raise Unsupported('No elementary integrating factor result.')
 out=s.simplify((F+s.exp(a*x0)*y0-F.subs(v,x0))/factor)
 check=s.simplify(s.diff(out,v)+a*out-f)==0 and s.simplify(out.subs(v,x0)-y0)==0
 return _result(str(out),[f'y prime + ({a})*y = {f}; y({x0}) = {y0}.',f'Integrating factor = {factor}.',f'y = {out}.'],'linear initial-value ODE tool',{'passed':bool(check),'check':'Substitute into differential equation and initial condition.'})

SCALES={'C':('C',1),'uC':('C',1e-6),'nC':('C',1e-9),'m':('m',1),'cm':('m',.01),'mm':('m',.001),'V':('V',1),'ohm':('ohm',1),'s':('s',1),'ms':('s',.001),'Wb':('Wb',1),'mWb':('Wb',.001),'Hz':('Hz',1),'J':('J',1),'eV':('J',1.602176634e-19),'N*m^2/C^2':('k',1)}
def quantity(v,u,positive=False):
 if not isinstance(v,dict) or set(v)!={'value','unit'} or v['unit'] not in SCALES or SCALES[v['unit']][0]!=u or type(v['value']) not in (int,float):raise Unsupported(f'Explicit compatible {u} quantity required.')
 x=v['value']*SCALES[v['unit']][1]
 if not math.isfinite(x) or abs(x)>1e16 or (positive and x<=0):raise Unsupported('Quantity outside finite supported range.')
 return x

def physics_tool(p):
 t=p['task']
 if t=='field_vector':
  point=p['point'];charges=p['charges']
  if not isinstance(point,list) or len(point)!=2 or not isinstance(charges,list) or not 1<=len(charges)<=12:raise Unsupported('2D point and 1–12 point charges required.')
  x,y=[quantity(z,'m') for z in point];k=quantity(p.get('k',{'value':8.9875517923e9,'unit':'N*m^2/C^2'}),'k',True);out=[0.,0.];steps=[f'Point charges in vacuum; k = {k} N*m^2/C^2; components along supplied Cartesian axes.']
  for c in charges:
   if set(c)!={'q','position'} or not isinstance(c['position'],list) or len(c['position'])!=2:raise Unsupported('Each charge requires q and a 2D position.')
   q=quantity(c['q'],'C');cx,cy=[quantity(z,'m') for z in c['position']];dx,dy=x-cx,y-cy;r=math.hypot(dx,dy)
   if r==0:raise Unsupported('Field is singular at a source charge.')
   ex,ey=k*q*dx/r**3,k*q*dy/r**3;out[0]+=ex;out[1]+=ey;steps.append(f'E_i = k*q*(r-r_i)/|r-r_i|^3 = ({ex}, {ey}) N/C.')
  steps.append(f'Vector sum = {out} N/C.')
  return _result(out,steps,'point-charge vector superposition',{'passed':True,'check':'SI vector components summed; inverse-square radial point-charge law; no independent physical proof.'},units='N/C')
 if t=='induction':
  N=p['turns'];a=quantity(p['flux_initial'],'Wb');b=quantity(p['flux_final'],'Wb');dt=quantity(p['duration'],'s',True)
  if type(N)!=int or not 1<=N<=100000:raise Unsupported('Positive bounded integer turn count required.')
  emf=-N*(b-a)/dt
  return _result(emf,[f'Flux per turn follows the supplied positive normal: {a} -> {b} Wb in {dt} s.',f'Average emf = -N*(Phi_final-Phi_initial)/dt = {emf} V; minus sign is Lenz law.'],'Faraday average induction tool',{'passed':True,'check':'Flux per turn, time and turn count checked; average rather than instantaneous emf.'},units='V')
 if t=='photoelectric':
  nu=quantity(p['frequency'],'Hz',True);phi=quantity(p['work_function'],'J')
  if phi<0:raise Unsupported('Work function must be nonnegative.')
  energy=6.62607015e-34*nu-phi
  if energy<0:raise Unsupported('Below threshold: no photoemission, so no stopping potential is defined.')
  out=energy/1.602176634e-19
  return _result(out,[f'Photon energy h*nu = {6.62607015e-34*nu} J; work function = {phi} J.',f'K_max = h*nu - phi = {energy} J.',f'Stopping potential magnitude = K_max/e = {out} V.'],'Einstein photoelectric tool',{'passed':True,'check':'Exact SI h and elementary charge; nonnegative kinetic energy and J/e -> V.'},units='V')
 import sympy as s
 fixed=p['fixed_voltages'];edges=p['resistors'];target=p['unknown']
 if not isinstance(fixed,dict) or not 1<=len(fixed)<=6 or not isinstance(edges,list) or not 1<=len(edges)<=15:raise Unsupported('Bounded resistor network and explicit fixed voltage nodes required.')
 nodes=set(fixed);normalized=[]
 for e in edges:
  if set(e)!={'a','b','resistance'} or not all(isinstance(e[k],str) and re.fullmatch('[A-Za-z][A-Za-z0-9_]{0,15}',e[k]) for k in ('a','b')) or e['a']==e['b']:raise Unsupported('Distinct named endpoints required.')
  nodes.update([e['a'],e['b']]);normalized.append((e['a'],e['b'],quantity(e['resistance'],'ohm',True)))
 if len(nodes)>8 or target not in nodes:raise Unsupported('Network exceeds eight nodes or target absent.')
 fv={k:s.Rational(str(quantity(v,'V'))) for k,v in fixed.items()};unknown=sorted(nodes-set(fv));vs={k:s.Symbol('V_'+k) for k in unknown};volt={**fv,**vs};eq=[]
 for k in unknown:eq.append(sum((volt[k]-volt[b if a==k else a])/s.Rational(str(r)) for a,b,r in normalized if k in (a,b)))
 if unknown:
  A,b=s.linear_eq_to_matrix(eq,list(vs.values()))
  if A.det()==0:raise Unsupported('Floating or singular circuit; supply sufficient voltage boundaries.')
  vals=A.inv()*b;solution=dict(zip(vs.values(),vals));volt.update({k:solution[v] for k,v in vs.items()})
  if any(s.simplify(e.subs(solution))!=0 for e in eq):raise Unsupported('KCL residual failed.')
 out=float(volt[target])
 return _result(out,[f'Fixed node voltages: {fv}.','Apply KCL sum((V_node - V_neighbor)/R) = 0 at each unknown node.',f'Node potentials in V: {volt}; requested {target} = {out} V.'],'passive resistor nodal KCL tool',{'passed':True,'check':'Exact rational KCL residual at every unknown node; ideal supplied fixed potentials.'},units='V')

def csv_tool(p):
 text=p['text'];column=p['column'];op=p.get('operation','sum')
 if not isinstance(text,str) or len(text)>12000 or op not in ('sum','count','mean'):raise Unsupported('Inline CSV only, bounded length and sum/count/mean operations.')
 reader=csv.DictReader(io.StringIO(text));headers=reader.fieldnames
 if not headers or len(headers)>20 or len(set(headers))!=len(headers) or column not in headers:raise Unsupported('Unique header names and selected column required.')
 rows=list(reader)
 if not 1<=len(rows)<=200 or any(set(r)!=set(headers) or any(v is None or not isinstance(v,str) or len(v)>1000 for v in r.values()) for r in rows):raise Unsupported('CSV rows must match headers; 1–200 rows.')
 vals=[float(r[column]) for r in rows]
 if any(not math.isfinite(v) or abs(v)>1e12 for v in vals):raise Unsupported('Finite bounded numeric CSV column required.')
 out=len(vals) if op=='count' else math.fsum(vals)/(len(vals) if op=='mean' else 1)
 return _result(out,[f'Read {len(rows)} inline CSV records with explicit header {headers}.',f'{op}({column}) = {out}; no filesystem read occurred.'],'inline CSV aggregation tool',{'passed':True,'check':'CSV parser, header/row bounds, finite numeric column; no host file access.'})
