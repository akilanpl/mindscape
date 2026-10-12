# Baseline and information matrix

| Condition | Training information | Representation / memory | Evaluation and compute |
| --- | --- | --- | --- |
| Mindscape | Approved trajectory budget, actual transition/contact labels | Four-frame history, relational/concept features, train-only residual memory | 400 updates, MPS; nine-candidate planning at final budget |
| Matched neural | Same approved IDs, transition/contact labels, batch and updates | Same38,409 parameter dimensions; extra feature/pooling channels zeroed, no memory | Same frozen scenes and planning candidates; MPS |
| No relationships | Same supervision | Relational features/pooling removed | Final250 budget, all seeds and scenes |
| No concepts | Same supervision | Derived concepts zeroed | Same final-budget schedule |
| No memory | Same supervision | Residual retrieval omitted | Same final-budget schedule |
| No history | Same supervision | Repeated current frame instead of four-frame history | Same final-budget schedule |
| No feedback | Permuted dynamics labels, actual contact labels | Full architecture | Negative supervision control; not an isolated architecture ablation |
| Persistence / constant velocity | No fitting | Current visible state / velocity | 400 scene forecasts per baseline; CPU nonneural |
| Analytical available | Legitimately visible parameters plus declared nominal hidden values | Free-flight equations, no contacts/friction/spring | 400 forecasts and80 goal plans; approximate, nonneural |
| Supplemental analytical contacts | Same model-visible state, physical equation priors | Approximate contacts, floor friction, spring equations | 400 extra forecasts; separately verified; no simulator answer queries |
| Random / simple | No training labels | Remove hypothetical candidate selection | 80 actual goal episodes each; no neural calls |
| Model-free policy |250 train scenes, extra labels from nine executed action alternatives | Four-frame flat policy input; no dynamics rollout | Six MPS fits,400 updates each;240 actual goal episodes |

Equal parameter counts do not imply equal effective capacity: zeroed feature/pooling channels remove useful input paths. The matched comparison isolates added mechanisms as scheduled; it does not exhaust strong unstructured multibody baselines. All forecast methods receive permitted observed states/actions/context without authoritative future states. Equation priors and extra model-free teachers are explicitly different information sources. Neural planning uses24 batched forward steps over nine candidates; model-free uses one policy forward; random/simple use none. All execute only one selected impulse for outcome scoring. Teacher queries occur exclusively on training scenes.

The no-planning investigation uses random/simple controls; no equal-compute isolated neural no-planning condition was scheduled. That limitation remains explicit. Evaluation feedback never enters persistent training memory. No LLM tokens, pretrained adapters or paid inference were used in this physics study.
