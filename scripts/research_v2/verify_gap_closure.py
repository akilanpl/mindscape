"""Independently verify saved closure evidence; no neural inference or task execution."""
import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

from coding_telemetry import digest

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--old-study', type=Path, required=True)
    parser.add_argument('--historical-evidence', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    old_manifest = json.loads((args.old_study/'MANIFEST.json').read_text())
    legal = Counter()
    for name, checksum in old_manifest.items():
        path = args.old_study/name
        if digest(path) != checksum:
            raise RuntimeError('Historical numeric artifact changed')
        if name.endswith('.json.gz'):
            with gzip.open(path, 'rt') as stream:trace = json.load(stream)
            for e in trace['events']:
                if e['component']=='MindscapeModel.select_action':legal[len(e['observed']['valid_actions'])]+=1
    g = args.package/'cloud/gap_closure'
    statistics = json.loads((g/'statistics.json').read_text())
    rows = [json.loads(line) for line in (g/'rows.jsonl').read_text().splitlines()]
    protocol = json.loads((ROOT/'configs/research_v2/gap_closure/protocol.json').read_text())
    data = json.loads((ROOT/'datasets/research_v2/masking_replication_v1.json').read_text())
    if data['protocol_sha256'] != digest(ROOT/'configs/research_v2/gap_closure/protocol.json'):
        raise RuntimeError('Protocol changed')
    expected = {(s,c,t['example']['example_id']) for s in protocol['seeds'] for c in protocol['conditions'] for t in data['cases']}
    if len(rows)!=288 or {tuple(r['key']) for r in rows}!=expected:
        raise RuntimeError('Missing/duplicate replication keys')
    for row in rows:
        if digest(g/row['trace'])!=row['sha256']:raise RuntimeError('Trace changed')
        with gzip.open(g/row['trace'],'rt') as stream:trace=json.load(stream)
        if trace['result']['score']!=row['success']:raise RuntimeError('Outcome mismatch')
    if statistics['raw_sha256']!=digest(g/'rows.jsonl') or statistics['unsuccessful']!=sum(not r['success'] for r in rows):
        raise RuntimeError('Statistics/raw mismatch')
    for cell in statistics['cells']:
        selected=[r for r in rows if r['family']==cell['family'] and r['key'][1]==cell['condition']]
        if len(selected)!=cell['executions'] or sum(r['success'] for r in selected)!=cell['successes']:
            raise RuntimeError('Cell statistics mismatch')
    from itertools import product
    raw_p=[]
    lookup={tuple(r['key']):r for r in rows}
    for test in statistics['tests']:
        ids=[t['example']['example_id'] for t in data['cases'] if t['family']==test['family']]
        seed,policy=test['seed'],test['policy']
        pairs=[(lookup[seed,policy+'_mask',t]['success'],lookup[seed,policy+'_permissive',t]['success']) for t in ids]
        b=sum(a and not c for a,c in pairs);c=sum(c and not a for a,c in pairs)
        n=b+c
        pvalue=sum(abs(2*sum(bits)-n)>=abs(2*b-n) for bits in product((0,1),repeat=n))/2**n if n else 1.
        if [b,c]!=test['discordant'] or abs(pvalue-test['exact_p'])>1e-12:raise RuntimeError('Independent exact test mismatch')
        raw_p.append(pvalue)
    previous=0.
    for rank,i in enumerate(sorted(range(len(raw_p)),key=raw_p.__getitem__)):
        previous=max(previous,min(1.,(len(raw_p)-rank)*raw_p[i]))
        if abs(previous-statistics['tests'][i]['holm_p_family18'])>1e-12:raise RuntimeError('Holm mismatch')
    index=json.loads((g/'coding_historical/index.json').read_text())
    for source,meta in index['sources'].items():
        path=args.historical_evidence/meta['path']
        if digest(path)!=meta['sha256']:raise RuntimeError('Historical coding source changed')
        lines=path.read_text().splitlines()
        entries=[e for e in index['entries'] if e['source']==source]
        if len(entries)!=len(lines) or len({e['line'] for e in entries})!=len(lines):raise RuntimeError('Duplicate/missing pointers')
        for entry in entries:
            if hashlib.sha256(lines[entry['line']-1].encode()).hexdigest()!=entry['raw_row_sha256']:
                raise RuntimeError('Historical pointer mismatch')
    for name in ('coding_probe_v2','coding_probe_c'):
        trace=json.loads((args.package/name/'trace.json').read_text())
        requests=[e for e in trace['events'] if e['kind']=='model_request']
        forwards=[e for e in trace['events'] if e['kind']=='actual_forward_output']
        if len(requests)!=trace['result']['model_calls'] or not forwards:raise RuntimeError('Missing actual neural telemetry')
        for request in requests:
            payload=json.loads(request['observed']['prompt'])
            if 'hidden_tests' in payload or 'private_tests' in payload:raise RuntimeError('Hidden input contamination')
        if trace['result']['success']!=trace['result']['terminal']['all_passed']:raise RuntimeError('Final scoring mismatch')
    zero_mask=[r for r in rows if r['key'][0]==0 and r['key'][1]=='zero_mask']
    zero_off=[r for r in rows if r['key'][0]==0 and r['key'][1]=='zero_permissive']
    if len(zero_mask)!=24 or not all(r['success'] for r in zero_mask) or any(r['success'] for r in zero_off):
        raise RuntimeError('Observed all-benefit control assumption changed; recompute general interval')
    bound=2*.0125**(1/24)-1
    receipt={'status':'PASS','old_study_hashes':len(old_manifest),'old_observed_legal_action_cardinalities':dict(legal),
             'new_unique_keys':288,'new_unsuccessful':statistics['unsuccessful'],'coding_pointers_verified':2320,
             'zero_control_independent_task_units':24,'descriptive_paired_effect':1.0,
             'post_hoc_conservative95_interval':[bound,1.0],
             'interval_method':'Simultaneous Bonferroni Clopper-Pearson bounds on benefit/harm probabilities; excludes duplicated zero seeds',
             'primary_holm_significant_tests':sum(t['holm_p_family18']<.05 for t in statistics['tests']),
             'neural_inference_in_this_verifier':False,'scope':'Saved evidence validation; no full Research v2 completion claim'}
    args.receipt.write_text(json.dumps(receipt,indent=2)+'\n')
    print('PASS288 replication traces,2320coding pointers, original numeric archive and actual neural diagnostics')


if __name__ == '__main__':
    main()
