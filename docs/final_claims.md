# Claims and evidence

|Claim|Status|
|---|---|
|high bounded-vertical capability|NOT SUPPORTED|
|OOD improvement|SUPPORTED|
|data efficiency|INCONCLUSIVE|
|grounded execution|SUPPORTED|
|trajectory validity|SUPPORTED|
|goal success|NOT SUPPORTED|
|recovery|INCONCLUSIVE|
|external benchmark competitiveness|INCONCLUSIVE|
|memory benefit|INCONCLUSIVE|
|dream benefit|INCONCLUSIVE|
|GPT reference exceedance|NOT SUPPORTED|
|Gemini reference exceedance|NOT SUPPORTED|

Capability/goal claims use a descriptive 90% threshold (reporting criterion, not a preregistered hypothesis test) on both locked splits. OOD improvement requires a positive task-paired 95% CI against A. Grounded execution requires all locked independent replay evidence to match. Grounding is established only for saved re-executed episodes. No architecture-only causal claim is supported.
