# Completion recovery inventory

Recovered on main at `4689e6d`; multiplication commits/results remain intact. Four useful jobs survived: full expert prefix tuple collection, 5000-example actual teacher collection/retrieval sweep, five-seed LoRA prefix training (10/25/50/100), and the initial primary adapter evaluation. None was terminated. The adapter evaluation completed and is reused; the other jobs continue from their existing checkpoints.

Historical 0.5B HumanEval: 94/164 = 57.317073% under greedy full-module instruction generation, CPython 3.14.7 WASI, 512 output tokens, one sample/task. It is preserved, with its noncanonical runtime documented. All 5060 generated correct/buggy pairs were actually validated (40480 function invocations); model-generated programs are never batched or run in host Python.

Completed 10-example adapter evaluation: plain baseline 18/20 IID, 10/20 OOD; structured baseline 20/20 IID, 9/20 OOD. These 40 cases are the primary learning-curve pool, not the later one-shot bounded-perfection set. The completion protocol calls for a separate 100-task lockbox, frozen before final model selection.

The existing large sweep is retrieval adaptation, not gradient data efficiency. The existing gradient checkpoints are patch-supervised; they cannot alone substantiate trajectory learning or experiential replay. Completion adds matched structured/action training and actual accepted-experience replay, with supervision differences audited explicitly. No DER or architecture advantage is inferred from the preliminary numbers.

Completion lockbox: 100 fresh tasks (50 IID/50 OOD), generation seed 13001, all prior training/evaluation seeds excluded. Preselected final budget100/seed11; no score-based configuration selection on this set. The 40-case learning-curve pool remains distinct and its prior diagnostic exposure is disclosed.

The surviving teacher/retrieval job was checkpointed deliberately before any retrieval inference to prioritize the new gradient objective. Its accepted teacher prefix is reused in a collection-only continuation; original protocol/checkpoints remain intact. This avoids running the old retrieval-only controller for hours as if it answered gradient data efficiency. Its supplemental inference sweep can run separately with the revised frozen controller. No completed evaluation was discarded.
