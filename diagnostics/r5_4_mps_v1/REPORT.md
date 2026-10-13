# R5.4 nontraining native engineering diagnosis

**Original readiness: ENGINEERING_NO_GO. Original scientific freeze intact. No training authorized or executed.**

The installed session runtime reports MPS built=true, available=false, fallback=0. Therefore this session performed only a diagnostic capability preflight and no MPS objective/backward calls. Cold/warm native distributions, gradient residency checks, GPU allocations and native peak RSS are unrun/unknown. Parent preflight peak RSS was 204,570,624 bytes; this is not a native workload memory measurement. Do not treat it as a passed 4 GiB workload check.

The user's separate Terminal result is transcribed in `user-reported-native-engineering.json`: native MPS available/built=true, fallback=false, zero updates, Markov 0.6119346670166124 seconds (projected 734.3216004199348), recurrent 0.21027987499837764 seconds (projected 252.33584999805316), ENGINEERING_NO_GO with reason `Frozen resource headroom insufficient`. These are user-reported figures, not an independently verified native artifact or byte-for-byte original JSON. The earlier `experiments/r5_4/engineering.json` remains unchanged and still describes the earlier unavailable-MPS session.

Inspection confirms the frozen gate always times Markov first, measures one first objective-plus-backward call, and omits peak RSS on failure. The Markov timing is about 1.99% above the frozen 720-second projection limit. A cold-start or order effect is plausible but not established. The current evidence also cannot rule out a memory-limit failure. No favorable timing was selected and no official gate was rerun.

The standalone diagnostic under `diagnostics/r5_4_mps_v1` adds a fixed four-process MR/RM/MR/RM schedule: one first-call round, three warm-up rounds and eight measured rounds per model per process, alternating slots. It records all 96 objective/backward calls if native execution completes, complete distributions and order strata, synchronization boundaries, float32 parameter/gradient/unchanged-weight checks, per-process peak RSS and distinct sampled MPS tensor/driver allocation counters. It uses only the same first 32 training tasks and unmodified update-401 objective. No optimizer is instantiated, no fitting is invoked, no evaluation labels are decoded, and no frozen file is edited. The original 19-file SHA256 inventory and original engineering artifact are verified before/after; evaluation files are accessed as opaque integrity bytes only.

Six pure schedule/statistics/integrity regressions pass, including incomplete/order-mismatched evidence, slow warmed measurements, and RSS failure handling. Tests do not execute either neural model or an optimizer. The session's guard-only report retains the source fingerprint from its invocation; that exact source is separately preserved. Subsequent changes added explicit order-difference reporting, failed-RSS handling and automatic source snapshots for future native runs. No native timing observations were available for those changes. The native measurement schedule did not change.

Exact next action, once in the native Terminal and the same Python environment as the previous successful MPS detection, from the checkout root:

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 python diagnostics/r5_4_mps_v1/run.py --output results/r5_4/mps-diagnostic-v1-native
```

Execute that fixed nontraining schedule once; preserve its entire output directory and review `REPORT.json` and `RECEIPT.json`. Do not invoke `python -m mindscape.r5_4 engineering` again to seek a favorable result, and do not execute any training command.

No qualification amendment can currently be justified because the native distributions and memory measurements remain missing. If complete native evidence consistently supports first-call overhead, submit a separately versioned **R5.4 engineering qualification v2** proposal for human review, with a precommitted warm-up/repetition/order statistic, unchanged 720-second/4 GiB limits, startup amortization and explicit optimizer-cost limitations. Retain v1 ENGINEERING_NO_GO. If RSS exceeds 4 GiB, retain ENGINEERING_NO_GO and propose a separately frozen engineering v2 plan; do not relax thresholds. Reviewing or approving qualification v2 does not authorize the six-fit optimizer campaign.

**ENGINEERING_NO_GO**
