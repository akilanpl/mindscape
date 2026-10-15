"""Development examples only; independent held-out questions live elsewhere."""
import argparse
import json
import platform
import resource
import statistics
import time
from pathlib import Path
from mindscape_delivery.neural import NeuralBackend

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',default='auto')
    parser.add_argument('--output',required=True)
    parser.add_argument('--model-path')
    args=parser.parse_args()
    import psutil
    backend=NeuralBackend(args.model_path,args.device)
    before=psutil.Process().memory_info().rss
    backend.load()
    result={'purpose':'Development smoke benchmark; not held-out evaluation', 'host':platform.platform(), 'backend':dict(backend.stats), 'rss_before_bytes':before, 'rss_loaded_bytes':psutil.Process().memory_info().rss, 'predictions':[]}
    questions=[('math','Differentiate x^3 + 2*x with respect to x. Give a brief answer.'),('physics','A 12 V potential difference is applied across a 4 ohm resistor. Find the current in amperes. Give a brief answer.'),('python','What does this Python program print?\ns=0\nfor i in range(4):\n    s+=i\nprint(s)\nGive a brief answer.')]
    for domain,question in questions:
        raw=backend.generate(question,max_new_tokens=160)
        result['predictions'].append({'domain':domain,'question':question,'raw_prediction':raw, **backend.stats['last_inference']})
        Path(args.output).parent.mkdir(parents=True,exist_ok=True)
        Path(args.output).write_text(json.dumps(result,indent=2))
        print(json.dumps(result['predictions'][-1]),flush=True)
    result['rss_after_bytes']=psutil.Process().memory_info().rss
    result['peak_process_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result['latency_p50_seconds']=statistics.median(p['seconds'] for p in result['predictions'])
    result['optimizer_updates']=0
    Path(args.output).write_text(json.dumps(result,indent=2))
if __name__=='__main__':
    main()
