# R5.4 MPS diagnostic v1 — nontraining only

This is a new diagnostic, not an amendment to R5.4-v1 and not an official engineering-gate rerun. `schedule.json` fixes the measurement plan before execution. Existing scientific source, architecture, data, optimizer budget, threshold, lock and protocol remain untouched. The previous environment artifact `experiments/r5_4/engineering.json` remains unchanged. The user's native Terminal figures are separately transcribed in `user-reported-native-engineering.json`, explicitly not an independently verified artifact or a preserved original JSON.

## Why inspect cold timing

The official frozen engineering implementation measures one first objective-plus-backward call for Markov, then one for recurrent. Although its field says `backward_seconds`, the timed interval includes CPU training-reference construction, feature/loss creation, per-example exposure `.item()` transfers/synchronization, forward evaluation, backward and final MPS synchronization. Model construction, snapshots and gradient checks are outside the timer. There is no warm-up and no order control. Compilation, allocation, lazy dispatch and global first-use effects are plausible explanations, not yet established. The Markov reported projection exceeds 720 seconds by 14.3216004199348 seconds (1.9891%). This narrow miss does not authorize a new threshold.

The code only attaches `peak_process_rss_bytes` on the success path. A failed timing or RSS check stores the same general reason and no RSS field. Thus the user's native failure cannot distinguish a timing-only miss from an additional memory failure. Native RSS is unknown, not below-limit by assumption.

## Fixed schedule and execution boundary

Run four sequential fresh child processes, with initial model orders MR, RM, MR, RM. Each creates both frozen architectures with seed 11, float32 MPS tensors, and unchanged snapshots. For each process: one first-objective round per model; three discarded-from-summary-but-retained warm-up rounds per model; eight measured rounds per model. Alternate base/reversed measurement order every round within each phase. Both slots are represented equally in warmed measurements. Four processes × two models × twelve calls = 96 total objective/backward calls, zero optimizer instances/updates. No adaptive repetition, fastest-run choice or official gate execution. Initial-model order, within-round slot and all phase/order identifiers are recorded.

The batch is exactly the first 32 frozen training tasks, mode `objective(..., update=401)`, using the unmodified frozen function. No batch changes, vectorization, synthetic easier tasks or objective simplifications. Gradient buffers clear with `model.zero_grad(set_to_none=True)` before each call; gradients then flow through the full prescribed objective. Training labels alone are constructed. The diagnostic never invokes fitting, campaign, qualification or evaluation. An Optimizer constructor guard raises if accidental construction occurs. No CPU forward/backward fallback or dependency download.

MPS synchronizes immediately before and after each objective-plus-backward interval. `perf_counter_ns` supplies inclusive wall time; there is no extra barrier between forward and backward, preserving the original timed boundary. The optional objective-return/backward-submission/final-sync segments are CPU wall intervals, **not GPU phase timings**. Finite/nonzero float32 gradients, finite resident parameters, expected counts and exact unchanged weights are checked after every call, outside the timed interval. Loss metadata, training identities, checks, timing, memory and source/lock hashes are retained for every call, including warm-up. Both models stay resident (tiny additional tensors); this is disclosed rather than silently changed between trials.

First-objective means first call of this objective for that model in a new process. GPU initialization and model construction precede it. Metal driver/disk caches can persist across processes. This design tests first-objective and order effects; it cannot uniquely attribute overhead to GPU compilation rather than Python, allocator or driver effects. Validation calls between trials also warm the runtime. No cache flush or forced eviction is used.

Each worker has a 90-second subprocess timeout; abort the remaining schedule on timeout, validation failure or RSS exceedance. Partial JSONL traces remain evidence. Output directories and final JSON files use exclusive creation, so reruns cannot overwrite a run. Four workers and all planned records are required for a complete summary. No retry is built in.

## Memory and summaries

On native macOS, `resource.getrusage(RUSAGE_SELF).ru_maxrss` is a lifetime process peak in bytes. It includes startup and validation, not just the timed interval. Record it initially, after every timed call and after checks; report whether it exceeds **4,294,967,296 bytes**. Record per-worker final peaks and parent startup RSS separately. Workers run sequentially. The 4 GiB condition is the original per-process limit; no subtraction of startup memory and no threshold relaxation.

`torch.mps.current_allocated_memory()` samples live tensor allocation. `driver_allocated_memory()` samples driver-managed memory, including allocator caches and other allocations. Neither is process RSS or a continuous GPU peak. CPU/GPU unified-memory views can overlap: do not add these counters to derive physical memory. GPU sample maxima, if computed, are only sampled maxima; RSS is the OS-observed process high-water mark.

Retain every cold/warm-up/measured sample. Report mean, median, sample standard deviation, interpolated p90/p95, min/max, per-process distributions, pooled distributions and strata by measured slot and process initial order. Minimum is descriptive only and never selected as qualification. Include cold-minus-warm-median and cold/warm ratios. Diagnostic warm-mean/p95 ×800×1.5 projections and `(cold +799×warm mean)×1.5` are labeled comparisons only: they omit optimizer/clipping cost and cannot replace the original gate.

A predeclared cold-start-support flag requires cold greater than warmed median for each model in all four processes, all warmed per-process p95 projections <=720 seconds, complete finite gradient/unchanged-weight checks and no RSS exceedance. The flag is descriptive, not a statistical proof or a readiness pass. Report order-stratum differences even when this flag is true. If observations disagree, diagnosis stays unresolved; do not repeat to seek a favorable result. Any observed RSS exceedance retains ENGINEERING_NO_GO and warrants a separately frozen v2 engineering plan. Missing native measurements leave timing and memory unresolved.

Hash inspection reads all locked artifacts only as opaque bytes for SHA256; it never decodes development/OOD/outcome labels. Only training JSON is parsed. Verify unchanged lock/source/data and the earlier engineering artifact before and after. Outputs include a hash receipt. This script sits outside the locked package and does not modify it.

## Exact native action

From the checkout root in the native Terminal where MPS is available, using the same installed Python as the previous native command:

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 python diagnostics/r5_4_mps_v1/run.py --output results/r5_4/mps-diagnostic-v1-native
```

Run this schedule once. It is explicitly authorized nontraining diagnosis, not training authorization. Preserve the entire output directory; review `REPORT.json`, `RECEIPT.json`, four worker JSONs and progress files. Never execute the six-fit training command during this diagnostic.

No qualification amendment is adopted or justified in advance. If complete native measurements support cold-start overhead, submit a human-review proposal titled `R5.4 engineering qualification v2`, separately freeze its own implementation/schedule/hash, and explicitly retain v1's ENGINEERING_NO_GO. The proposal must specify a precommitted warm-up/repetition/order statistic, unchanged 720-second and 4 GiB limits, startup amortization and treatment of optimizer overhead. The current training entry point remains blocked under its original gate until an explicitly approved, independently versioned integration plan exists. If memory exceeds 4 GiB, propose a separately frozen engineering v2 plan with measured allocation diagnosis and an agreed scope; do not reclassify v1 or relax limits after seeing results. A v2 qualification approval is still not optimizer-campaign authorization.
