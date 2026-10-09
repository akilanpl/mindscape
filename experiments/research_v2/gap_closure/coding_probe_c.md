# Observed coding-foundation diagnostic

One repeated historical task as a telemetry diagnostic, not new accuracy evidence

Task: 94f35b7ac3218cac61f74e9a05b1b2cde08883e97af9746a06d9048c20d89b29
Model: 2e1fd397ee46e1388853d2af2c993145b0f1098a:adapter:b0400c88517b542e7fbd112fb591869c4310d44be038424cd9462b7e4a276ac0
Backend: cpu float32; generation cache: False
Actual model calls: 8; independently scored success: True
Trace SHA-256: 22cf6951ad496c9a3391f844a1a3fe52cca535463c52cb770ea822f19541c63d

| Directly observed event/component | Count |
|---|---:|
| BoundedPlanner.plan | 8 |
| CodingMemory.dream | 8 |
| CodingMemory.remember | 3 |
| CodingMemory.reset_working | 1 |
| CodingMemory.retrieve | 8 |
| actual_forward_output | 352 |
| actual_token_input | 352 |
| deterministic_environment_transition | 3 |
| model_request | 8 |
| model_response | 8 |

Exact generate arguments and direct forward token IDs are in the JSON events. Logits are direct forward observations; their softmax summaries are derived, uncalibrated probabilities. State changes are actual environment transitions. Memory and planner events record actual invocations; hypothetical proposals are not executed outcomes or causal explanations.

Failures: [null, null, null, "Edit budget exhausted", "Edit budget exhausted", "Edit budget exhausted", "Edit budget exhausted", "Edit budget exhausted"]

Historical prompts/logits are not retroactively recreated. These CPU float32 diagnostics are distinct from the original MPS/CPU benchmark measurements and are not new accuracy cases. Selected hidden activations and all-adapter live coverage remain unmeasured.
