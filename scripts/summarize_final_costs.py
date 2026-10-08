"""Disclose shared encoder work, training rows, feedback and prediction calls separately."""
import argparse,json
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('results');a=p.parse_args();root=Path(a.results);runs=[json.loads(x) for x in (root/'runs.jsonl').read_text().splitlines()];models={}
 for r in runs:
  if r['variant'] not in ('full','no_explicit_state','no_episodic_memory'):continue
  key=(r['variant'],r['condition'],r['budget'],r['seed']);models[key]=r
 modelcost=[]
 for key,r in models.items():
  s=r['resources'];modelcost.append(dict(variant=key[0],condition=key[1],budget=key[2],seed=key[3],distinct_training_problems=s['training_problem_count'],supervision_rows=s['training_rows'],gradient_steps=s['gradient_steps'],sampled_training_rows=s['minibatch_rows_processed'],feedback_queries=s['feedback_attempts'],training_head_calls=s['gradient_steps']+s['feedback_attempts'],training_transformer_calls=s['transformer_calls'],training_transformer_tokens=s['transformer_tokens'],training_seconds=s['training_time']))
 evalcost=[dict(condition=r['condition'],variant=r['variant'],budget=r['budget'],seed=r['seed'],split=r['split'],examples=r['metrics']['example_count'],prediction_head_calls=r['metrics']['model_calls'],uncached_transformer_calls=r['inference_transformer_calls'],uncached_transformer_tokens=r['inference_transformer_tokens'],seconds=r['metrics']['inference_time']) for r in runs]
 totals=dict(trained_models=len(models),evaluations=len(runs),training_seconds=sum(x['training_seconds'] for x in modelcost),evaluation_seconds=sum(x['seconds'] for x in evalcost),gradient_steps=sum(x['gradient_steps'] for x in modelcost),sampled_supervision_rows=sum(x['sampled_training_rows'] for x in modelcost),feedback_queries=sum(x['feedback_queries'] for x in modelcost),prediction_head_calls=sum(x['prediction_head_calls'] or 0 for x in evalcost),uncached_evaluation_transformer_tokens=sum(x['uncached_transformer_tokens'] for x in evalcost))
 payload=dict(totals=totals,models=modelcost,evaluations=evalcost,shared_encoder=json.loads((root/'resource_totals.json').read_text()),interpretation='Actual CPU research durations include warm/shared target-free feature caching. Distinct problems, labels, gradient minibatch rows and feedback queries are different resource units. Original examples_seen metadata means sampled supervision rows. Encoder token counts reflect actual cache misses, not hypothetical uncached deployment. Adam update calls and feedback forward calls are disclosed separately from frozen transformer batches. This does not establish compute efficiency superiority.')
 (root/'costs.json').write_text(json.dumps(payload,indent=2));print(totals)
if __name__=='__main__':main()
