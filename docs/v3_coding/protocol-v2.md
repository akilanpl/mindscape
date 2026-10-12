# Transport V2, a fresh frozen study

V1's transport failures are retained. V2 keeps the exact generator, six OOD families, counts, 24 approved training tasks, learned checkpoints, two-call/two-edit/256-token limits, methods and60% target. Only the prompt's action schema changes: it enumerates allowed existing paths and removes the fictitious filename example. No model-proposed filename or source is silently normalized, and no reference patch enters a held-out prompt.

V2 uses seed840017 and18 fresh held-out tasks (6 IID/12 OOD), with no V1 held-out task identities. Numerical parameters are sampled from the same2..179 bounds while excluding all previously used parameters. This prevents an exact parameterized-task copy, rather than selecting tasks by model success. Train/validation tasks and their actual prior validation receipts are reused explicitly; every new holdout defect/oracle was executed on visible/private tests before freeze. There are72 new validation VM executions, in addition to V1's192.

The transport change has a separate training-only diagnostic; none of V2's private outcomes informs it. Both versioned datasets, protocols, failed V1 episodes and all V2 outcomes remain available. V2 is a newly predeclared follow-up, not a rewritten V1 result. Existing historical lockbox data is never opened, modified or tuned.

Primary evaluator seed11, secondary critic seeds23/37; greedy Qwen remains fixed. Primary success is private structural-OOD exact accuracy on all12 tasks. A mechanism advantage requires at least10 percentage points over the strongest new same-Qwen baseline and a positive paired task-bootstrap95% lower bound. With12 tasks, intervals and seed conclusions remain limited; passing60% alone does not establish superiority or general coding intelligence.

Batched execution changes only throughput. Its identity records the wrapper source hash, configuration, frozen core hash, dataset, protocol, backbone files and trained checkpoint files. Scalar V1 partial evidence and complete/further batched evidence remain separate; the study does not select a compute stratum by accuracy.
