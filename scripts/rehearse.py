"""Development demonstrations only; never the held-out acceptance set."""
import argparse,json,time,urllib.request
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--url',default='http://127.0.0.1:8765');p.add_argument('--output',default='rehearsal.json');a=p.parse_args();started=time.perf_counter()
 cases=[('math','Differentiate x^3 + sin(x) with respect to x','3*x**2 + cos(x)'),('physics','Find current through a 4 ohm resistor with 12 V across it',3.0),('coding','Predict the output:\n```python\ndef total(n):\n s=0\n for i in range(n):\n  s+=i\n return s\nprint(total(5))\n```','10')]
 health=json.load(urllib.request.urlopen(a.url+'/api/health'));results=[]
 for topic,q,expected in cases:
  req=urllib.request.Request(a.url+'/api/chat',data=json.dumps({'message':q,'inference_mode':'hybrid'}).encode(),headers={'Content-Type':'application/json'})
  st=time.perf_counter();r=json.load(urllib.request.urlopen(req,timeout=125));answer=r.get('solver_result',{}).get('answer');passed=r.get('status')=='verified' and answer==expected
  results.append({'topic':topic,'question':q,'expected':expected,'passed':passed,'latency_seconds':time.perf_counter()-st,'response':r})
 report={'status':'PASSED' if all(x['passed'] for x in results) and health['neural']['ready'] else 'FAILED_OR_SOLVER_ONLY','health':health,'results':results,'elapsed_seconds':time.perf_counter()-started,'scope':'Public development examples; not held-out competence evidence.'}
 Path(a.output).write_text(json.dumps(report,indent=2));print(json.dumps({'status':report['status'],'seconds':report['elapsed_seconds']}))
 if report['status']!='PASSED':raise SystemExit(1)
if __name__=='__main__':main()
