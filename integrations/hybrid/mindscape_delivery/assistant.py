"""Local neural-assisted orchestration with explicit method provenance."""
import json,os,subprocess,sys,threading,time
from pathlib import Path
from .parsing import supplied_parse,ground_neural
from .context import Contexts,validate,intent,explain,clarification

def tool(spec):
    env=dict(os.environ);env['PYTHONPATH']=str(Path(__file__).resolve().parents[1])
    try:
        p=subprocess.run([sys.executable,'-m','mindscape_delivery.tool_worker'],input=json.dumps(spec),text=True,capture_output=True,timeout=7,env=env)
        if p.returncode or len(p.stdout)>100000:raise ValueError('Tool worker failed')
        result=json.loads(p.stdout)
        if not isinstance(result,dict):raise ValueError('Tool output invalid')
        return result
    except (ValueError,subprocess.TimeoutExpired):return {'supported':False,'answer':'Tool timeout or unsupported complexity. Please simplify the problem.','steps':[],'method':'bounded verified solver','verification':{'passed':False}}

class Assistant:
    def __init__(self,neural=None):
        self.neural=neural;self.started=time.perf_counter();self._lock=threading.Lock();self._contexts=Contexts()
    def capabilities(self):
        return {'product':'Mindscape STEM delivery candidate','external_inference':False,'pretrained':{'model':'Qwen/Qwen3-0.6B','ready':bool(self.neural and self.neural.model is not None),'backend':getattr(self.neural,'stats',{})},'tools':{'math':['differentiate','standard integration','matrix addition/multiplication/inverse/transpose','determinants','polynomial equations'],'physics':['Coulomb force','point-charge electric field','Ohm law','series/parallel resistance','signed lens/mirror equations','single series-loop Kirchhoff current'],'coding':['bounded Python tracing: functions, conditionals, loops, lists','restricted SQLite SELECT with explicit tables']},'native':{'status':'Research demonstration only; R6 math/coding collapse and weak physics OOD','separate_from_pretrained_and_tools':True},'limits':['No broad CBSE mastery claim','Tool checks verify the displayed interpretation; verify it matches your question','No arbitrary code execution, filesystem access or network tools','Local loopback serving; no public hosting','Chat history is browser-local; up to 32 recent server-owned answers support follow-ups for 30 minutes, until restart or eviction']}
    def answer(self,question,mode='hybrid',cancel=None):
        if not isinstance(question,str) or not 1<=len(question.strip())<=6000:raise ValueError('Question must contain1–6000 characters')
        if mode not in ('hybrid','pretrained','tools','native'):raise ValueError('Unknown inference mode')
        started=time.perf_counter();trace=[];neural={};cancel=cancel or threading.Event()
        result={'status':'unsupported','domain':'unsupported','tools':[],'limitations':[],'native':None}
        if mode=='native':
            return {**result,'answer':'Method: Native Mindscape R6 research interface.\n\nR6 cannot accept general CBSE natural-language questions. Its verified research results show math/coding output collapse and weak physics OOD generalization. Open Model & research for the frozen-checkpoint demonstration.','method':'native R6 unsupported input','timing':{'total_seconds':time.perf_counter()-started},'trace':[{'kind':'native','summary':'No pretrained or symbolic answer is attributed to R6.'}],'provenance':{'native_neural_only':True,'optimizer_updates':0}}
        if mode!='tools':
            if self.neural is None:
                if mode=='pretrained':raise ValueError('Pretrained model unavailable; start model-backed server')
                neural={'status':'unavailable'};result['limitations'].append('Pretrained model unavailable: this response uses supplied grammar and verified tools only.')
            else:
                if mode=='pretrained':
                    raw=self.neural.generate(question+'\nGive a concise answer and short worked explanation. State units when relevant.',max_new_tokens=192)
                    return {**result,'status':'unverified','answer':'Method: Pretrained Qwen3-0.6B neural-only; not tool-verified.\n\n'+raw,'method':'pretrained neural-only','neural':{'raw_answer':raw},'timing':{'total_seconds':time.perf_counter()-started},'trace':[{'kind':'pretrained','summary':'Local Qwen3-0.6B generated this answer without solver verification.'}],'provenance':{'native_r6_used':False,'pretrained_model':'Qwen/Qwen3-0.6B','tool_verified':False,'optimizer_updates':0}}
                neural=self.neural.structured_parse(question)
                trace.append({'kind':'neural interpretation','summary':'Local pretrained Qwen3-0.6B produced an advisory structured interpretation; tool inputs require independent grounding.'})
        if cancel.is_set():return {**result,'status':'cancelled','answer':'Method: Cancelled local request.\n\nNo solver answer produced.','method':'cancelled','trace':trace,'provenance':{'optimizer_updates':0}}
        spec=supplied_parse(question);parser='supplied grammar'
        if spec is None and neural:
            spec=ground_neural(question,neural);parser='grounded pretrained interpretation'
        if spec is None:
            return {**result,'answer':'Method: Unsupported-question handling'+(' after pretrained interpretation' if neural else '')+'.\n\nI cannot verify this question with the supported tools. Try an explicit expression, a fenced Python program, labeled physical quantities with units, or SQL with explicit table data. For optics provide signed f and u. I will not invent a checked answer.','method':'unsupported; no verified answer','neural':neural,'trace':trace,'timing':{'total_seconds':time.perf_counter()-started},'provenance':{'native_r6_used':False,'pretrained_used':bool(neural and neural.get('raw')),'tool_verified':False,'optimizer_updates':0}}
        solver=tool(spec);verified=solver.get('supported') and solver.get('verification',{}).get('passed') is True
        method=(('Pretrained Qwen3-0.6B advisory interpretation + ' if neural.get('raw') else '')+parser+' + '+solver.get('method','verified computation'))
        text='Method: '+method+'.\n\n'
        text+='Interpreted problem:\n```json\n'+json.dumps(spec,indent=2)+'\n```\n\n'
        if verified:
            answer=solver.get('answer');rendered=answer if isinstance(answer,str) else json.dumps(answer,ensure_ascii=False)
            text+='Answer: '+str(rendered)+((' '+solver['units']) if isinstance(solver.get('units'),str) else '')+'\n\n'
            steps=list(solver.get('steps',[]))
            if spec.get('subject')=='coding':
                for event in solver.get('trace',[])[-8:]:
                    steps.append('Trace line '+str(event.get('line'))+' (call depth '+str(event.get('call_depth',0))+'): '+json.dumps(event.get('variables',{}),ensure_ascii=False))
            text+='Worked solution:\n'+'\n'.join(f'{i+1}. {step}' for i,step in enumerate(steps))+'\n\n'
            text+='Check: '+str(solver.get('verification',{}).get('check','Passed.'))
        else:text+=str(solver.get('answer') or solver.get('reason') or solver.get('error') or 'This problem is unsupported or failed verification.')
        trace += [{'kind':'structured problem','summary':parser+': '+json.dumps(spec)}, {'kind':'verified computation','summary':json.dumps(solver.get('verification',{}))}]
        return {**result,'status':'verified' if verified else 'unsupported','answer':text,'domain':spec.get('subject'),'method':method,'structured_problem':spec,'solver_result':solver,'neural':neural,'tools':[solver.get('method')],'trace':trace,'timing':{'total_seconds':time.perf_counter()-started},'provenance':{'native_r6_used':False,'pretrained_used':bool(neural.get('raw')),'interpretation':parser,'tool_verified':bool(verified),'verification_scope':'Displayed structured problem; some checks use the same symbolic engine','optimizer_updates':0}}
    def chat(self,data,cancel=None):
        if not isinstance(data,dict):raise ValueError('JSON object required')
        cid,history=validate(data)
        question=data.get('message','')
        mode='native' if data.get('native_demo') else data.get('inference_mode',data.get('mode','hybrid'))
        if not isinstance(question,str) or not 1<=len(question.strip())<=6000:raise ValueError('Question must contain 1–6000 characters')
        if mode not in ('hybrid','pretrained','tools','native'):raise ValueError('Unknown inference mode')
        started=time.perf_counter()
        with self._lock:
            context=self._contexts.previous(cid,history)
        if intent(question) and supplied_parse(question) is None:
            if mode in ('native','pretrained'):
                response=clarification('Grounded follow-up explanations are available in Hybrid or Tools only mode. Select one of those modes and resend the original problem if needed.')
            elif cancel is not None and cancel.is_set():
                response=clarification('The request was cancelled. Please resend the problem if you want an explanation.')
            else:response=explain(question,context)
        else:response=self.answer(question,mode=mode,cancel=cancel)
        with self._lock:
            self._contexts.remember(cid,question,response,context)
        response.setdefault('timing',{})['total_seconds']=time.perf_counter()-started
        return response
