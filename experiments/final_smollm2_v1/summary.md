# Final measured study

210 evaluations; 84 primary training runs plus6 retrained ablations; seeds0/1/2.

| Condition | IID mean ± SD | OOD mean ± SD | Grounded IID | Goal IID | Valid trajectory IID |
|---|---:|---:|---:|---:|---:|
|answer_only|5.000% ± 0.866|0.000% ± 0.000|0.000% ± 0.000|0.000% ± 0.000|N/A|
|structured|5.333% ± 1.041|0.000% ± 0.000|0.000% ± 0.000|0.000% ± 0.000|N/A|
|trajectory|41.500% ± 2.179|6.111% ± 1.347|41.500% ± 2.179|41.500% ± 2.179|41.500% ± 2.179|
|experiential|19.167% ± 3.014|0.000% ± 0.000|19.167% ± 3.014|19.167% ± 3.014|19.167% ± 3.014|

## Claims

| Claim | Status | Evidence | Limitations |
|---|---|---|---|
|Mindscape improves accuracy|INCONCLUSIVE|See primary curves and matched-information control.|State/scaffold and supervision differ from answer baselines; wrapper causal benefit must exceed matched local policy.|
|Mindscape improves data efficiency|NOT SUPPORTED|Observed N* and DER only.|Unequal label/query counts; compute reported separately.|
|Mindscape improves structural OOD generalization|INCONCLUSIVE|Unseen structures evaluated separately.|One domain and hand-defined cursor.|
|Mindscape improves groundedness|INCONCLUSIVE|Verified executed evidence measured.|Answer-only protocol emits no trajectory; metric favors evidence emitters by definition.|
|Mindscape improves trajectory validity|INCONCLUSIVE|Complete verifier checks.|Baselines without trajectories have null validity; matched policy is required.|
|Explicit state contributes|INCONCLUSIVE|Retrained no-state intervention at n50.|One budget and changes representation. Support, if present, is limited to IID at n50; OOD is separately reported.|
|Episodic memory contributes|NOT SUPPORTED|Matched accepted rows/order/updates no-storage control.|Tests SQLite retrieval versus in-memory replay, not all reuse.|
|Dream simulation contributes|NOT SUPPORTED|Depth2/branch3 intervention n50.|Confidence scoring, not learned world model; extra inference calls.|
|Architecture transfers to another environment|NOT SUPPORTED|Goal-directed generic integer dream demo retained.|No second learned benchmark; extension deferred.|

The comparisons characterize conditional prediction under a manually designed multiplication scaffold. They do not establish general intelligence, discovery of arbitrary procedures, or superiority over GPT/Gemini. All inference claims are uncorrected. Raw failures and historical negative results remain available.
