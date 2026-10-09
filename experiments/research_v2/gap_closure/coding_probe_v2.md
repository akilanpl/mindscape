# Observed coding-foundation diagnostic

One repeated historical task as a telemetry diagnostic, not new accuracy evidence

Task: 94f35b7ac3218cac61f74e9a05b1b2cde08883e97af9746a06d9048c20d89b29
Model: 2e1fd397ee46e1388853d2af2c993145b0f1098a:adapter:85b6cdc1d5667a9839c8d91c42deb5dd12dd0b3a4f407a56a9f0d51400b782f9
Backend: cpu float32; generation cache: False
Actual model calls: 1; independently scored success: False
Trace SHA-256: c7205d7d1f50a8326e8d6a3a6141eb4456e3b7961e182593fdfaf839d312e8f0

| Directly observed event/component | Count |
|---|---:|
| actual_forward_output | 39 |
| actual_token_input | 39 |
| deterministic_environment_transition | 1 |
| model_request | 1 |
| model_response | 1 |

Exact generate arguments and direct forward token IDs are in the JSON events. Logits are direct forward observations; their softmax summaries are derived, uncalibrated probabilities. State changes are actual environment transitions. Memory and planner events record actual invocations; hypothetical proposals are not executed outcomes or causal explanations.

Failures: ["Unrecognized source edit envelope"]

Historical prompts/logits are not retroactively recreated. These CPU float32 diagnostics are distinct from the original MPS/CPU benchmark measurements and are not new accuracy cases. Selected hidden activations and all-adapter live coverage remain unmeasured.
