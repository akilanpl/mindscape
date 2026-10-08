"""Separate post-evaluation hypothetical annotations; never alters original predictions."""
import argparse,json
from pathlib import Path
from mindscape.core.schema import State
from mindscape.core.serialization import decode
from mindscape.environments.multiplication.claims import expected_value

def main():
 p=argparse.ArgumentParser();p.add_argument('results');a=p.parse_args();root=Path(a.results);runs=[json.loads(x) for x in (root/'runs.jsonl').read_text().splitlines()]
 for r in runs:
  if r['variant']!='no_environment_interaction':continue
  folder=Path(r['directory']);out=folder/'hypothetical_judgments.json'
  if out.exists():raise ValueError('Refuse overwrite')
  judgments=[]
  for row in map(json.loads,(folder/'predictions.jsonl').read_text().splitlines()):
   for index,d in enumerate(row['diagnostics']['decisions']):
    value=expected_value(decode(State,d['state_before']));judgments.append(dict(example_id=row['example_id'],decision_index=index,correct_action=value,chosen_action=d['chosen_action'],hypothetical_judgment=d['selected_value']==value,hypothetical=True,transition_valid=None,trajectory_valid=None))
  out.write_text(json.dumps(dict(annotations=judgments,interpretation='Verifier-only posthoc values. Original predictions remain untouched. Hypothetical agreement does not establish real transition/trajectory validity.'),indent=2))
 print('Hypothetical posthoc sidecars saved')
if __name__=='__main__':main()
