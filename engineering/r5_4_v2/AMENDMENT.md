# R5.4-v2 engineering qualification amendment — proposed for human review

Status: **R54_V2_FROZEN_AWAITING_QUALIFICATION_AUTHORIZATION**. Preparation is authorized; native execution and the six-fit training campaign are not authorized. This amendment changes only the proposed engineering qualification method. Original R5.4-v1 source, lock, data, protocol and ENGINEERING_NO_GO evidence remain byte-identical. It does not adopt a new training entry point or rewrite older findings.

## Audited basis and limits of causal explanation

The complete native diagnostic directory was located and audited read-only, not inferred from pasted summary text. Thirteen original artifacts are preserved byte-for-byte in `experiments/r5_4/engineering_v2/evidence/native-diagnostic-v1.tar.gz`, with a member/archive SHA256 receipt. The independent auditor imports neither PyTorch nor the diagnostic predictor/summary implementation. It verifies four workers, 24 calls each, MR/RM/MR/RM initial order, all measurement slots, first 32 training identities, frozen source/lock/schedule provenance, 96 synchronized objective/backward calls, all progress/report mirrors, 576 parameter/gradient assertions, finite positive recorded norms, unchanged-weight assertions, monotonically increasing RSS, memory attribution and all timing distributions/projections. Schedule snapshot JSON is verified against canonical serialization of the committed plan; its formatting hash legitimately differs from the source plan's formatting hash.

Recorded-native evidence audit: VERIFIED_RECORDED_NATIVE_DIAGNOSTIC. Markov warm mean 0.126500559875 seconds; warm P95 0.13017294375 seconds. Recurrent mean 0.12657036840625; P95 0.12816779565. All 32 warm timings per model are retained. Process peak 376,832,000 bytes (359.375 MiB), below 4 GiB. Sampled maximum live MPS tensors 54,784 bytes; sampled driver allocation 11,550,720 bytes. These GPU samples are not OS RSS or continuous GPU peaks and cannot be added to RSS as disjoint physical memory. Original diagnostic optimizer instances/updates=0; no evaluation tasks.

Cold-start overhead is supported: for both models in every worker, first-call timing exceeded warmed median; all warmed worker P95 comparisons were below 720 seconds. But the entire original 0.6119346670166124-second Markov event is **not causally explained**. The diagnostic's maximum Markov first-call time was 0.241437 seconds; global driver/disk caches, process/runtime initialization, Python exposure bookkeeping, workload scheduling and system activity were not isolated. Warm performance is not evidence of learned arithmetic or an explanation of every microsecond of the original event. The original native v1 NO_GO remains a user-reported Terminal transcription; the original complete diagnostic is independently hash/record-audited.

The v1 diagnostic saved norms/metadata/assertions, not raw weights/gradients or OS process attestations. The audit verifies recorded assertions against the exact preserved source; it cannot independently regenerate raw tensor equality, GPU residency or hardware execution. This limitation is explicit. V2 prospectively saves raw initial/restored tensors, clipped gradients and reset optimizer states for independent storage-only tensor inspection.

## Fixed proposed qualification, one scheduled invocation

Six fresh sequential worker processes, in the exact order: Markov 11, recurrent 11, recurrent 23, Markov 23, Markov 47, recurrent 47. These are disposable engineering microbenchmarks, **not six learning fits**. No extra scientific seeds are introduced. Each worker uses the frozen architecture and initialization seed, float32 MPS, the original unmodified `objective(...,401)`, and two fixed training-only batches: first 32 manifest tasks and first 32 tasks with both operands four digits. The latter covers the maximum training horizon and all active positions. Neither development nor OOD labels are decoded; integrity hashes read those files only as opaque bytes.

Per worker/batch: one first-call timing, three fixed warm-up calls and eight measured calls. Alternate baseline/stress and stress/baseline order each round within each phase. All samples, complete distributions, loss/dtype, residency, gradient metadata, memory and exact restoration evidence are saved. There are 24 full-update microbenchmarks per worker, 144 total, including 96 measured warm calls. No adaptive warm-up, stopping on convergence, fastest measurement, favorable worker selection, extra seed, retry or post-result statistic change.

## Optimizer overhead is explicit and isolated

To measure actual step overhead, construct AdamW at the frozen lr .003 and weight decay .0001 on **disposable parameters only**. Full timed interval includes zero-grad, objective, finite loss check, backward, finite gradient check, norm 1 clipping, actual AdamW step, MPS synchronization, host loss conversion and the same progress JSON serialization/write cost as frozen fitting. Synchronize immediately before and after. Initialization, first-step state allocation and cold dispatch are recorded; all process launch/import/setup/probe/restoration/validation wall time is charged separately in the projection below.

An actual scratch step must change at least one parameter; afterward restore every weight exactly to initialized values and zero all optimizer moments and counter, retaining allocated buffers. Warm calls therefore include allocated-state update kernels but no learning trajectory accumulates. The optimizer's scalar step counter may reside on CPU under native PyTorch, while parameter/gradient and moment computations must be MPS. CPU transfers only serialize evidence; they do not perform inference or backward fallback. Save initial/restored float32 tensors, all clipped gradients, final reset optimizer state and hashes. No optimized parameter vector is used in the next forward call or retained as a learned checkpoint. A failure after a scratch mutation may preserve failed forensic tensors, labeled as such; it never creates readiness or learning evidence.

If approved and completed, qualification performs **144 actual disposable optimizer steps**, while **campaign optimizer updates remain 0**. Calling this globally “zero optimizer updates” would be misleading. No optimizer construction, step, MPS objective call or v2 qualification executes during preparation or regression testing. Human review must explicitly approve this disposable-step workload. Qualification approval never implies approval to execute any learning fit.

## Unchanged limits and predeclared projection

For each configuration, compute both batch distributions from all eight measured values. Linear-interpolated P95 at (n−1)×.95 is fixed before execution. The chosen steady-state cost is the **maximum** of the two batch P95 values, never the minimum/fastest. Report mean, median, standard deviation, P95, range and all samples separately for cold/warm-up/measured phases. Require every one of the six configurations to pass; do not average away a slow seed/model.

Projected fit seconds = **1.5 × [entire parent-observed worker wall time + 800 × max(batch warm P95 full-update seconds)]**.

The entire probe worker duration charges Python startup, imports, data loading, model/optimizer initialization, all cold/warm-up/measured probes, state allocation, validation, gradient storage and restoration. Including all probe work is deliberately conservative; no cold event is silently replaced by a warm value. The 800 additional projected update costs include optimizer-step/clipping/progress overhead. This screen assumes warmed parameter-value-independent kernel costs; real training trajectories and thermal changes could differ. It is not an 800-update campaign benchmark. Failed campaign budgets would still produce negative evidence rather than a relaxed limit.

Limits remain **projected fit <=720 seconds**, **fit runtime 900 seconds**, and **per-process lifetime peak RSS <=4,294,967,296 bytes (4 GiB)**. Peak RSS includes initialization, snapshots, optimizer states, CPU storage copies and validation. No subtraction of startup memory. Parent process RSS is also screened. Each qualification worker has an additional 90-second subprocess timeout; first failure stops the scheduled invocation. OS RSS is a lifetime peak in bytes on native macOS. Live tensor/driver allocation counters are sampled separately and are not treated as RSS or continuous GPU peaks.

Architectures, parameter counts, training/dev/OOD manifests, seeds 11/23/47, original six-fit allocation, batch 32 learning policy, 800 scientific updates per fit, optimizer policy, update-400 exposure boundary, checkpoint selection, predictions-before-outcomes ordering, evaluation denominators, success gates and 900-second campaign runtime remain unchanged. The existing campaign still calls its original v1 engineering gate: v2 preparation does **not** silently wire a replacement into it. Adoption/integration needs an explicit reviewed follow-up; the scientific source freeze itself remains intact.

## Status, audit and preservation

Original v1 readiness: ENGINEERING_NO_GO, immutable.

V2 preparation: R54_V2_FROZEN_AWAITING_QUALIFICATION_AUTHORIZATION.

V2 completed and independently verified readiness: R54_V2_ENGINEERING_READY_TRAINING_NOT_AUTHORIZED. This is engineering readiness only, not a learned-result PASS.

V2 negative/partial work: R54_V2_ENGINEERING_NO_GO. Native unavailability and invalid evidence are separately recorded; failed or incomplete evidence cannot qualify. The result record remains unchanged when read-only audit produces a separate verdict. Initial and final receipts cover all artifacts. Exclusive output creation prevents overwrite/resumption. No automatic retries.

The independent v2 auditor separately reconstructs schedules, distributions, the projection and decisions; verifies hashes, identities, logs, true scratch/campaign counts, tensor shapes/dtype/finiteness, exact initial/restored equality, parameter hashes, all saved clipped gradients and zero final optimizer moments. Float32 gradient-norm comparisons between MPS-recorded scalars and CPU storage inspection use declared relative 1e−5/absolute1e−8 tolerance; integer counts, identity fields and restoration hashes remain exact. CPU storage inspection never runs a neural forward, backward or optimizer. Timing calculations use finite Python float64 with 1e−12 equality tolerance. Record hardware-attestation limitations rather than treating receipts as proof of hardware identity.

All amended source, this specification, plan, independent tests, audit, preserved native archive/receipt, original-engineering/diagnostic-source references and v1 lock digest are separately frozen under `engineering/r5_4_v2/LOCK.json` before any approved execution. A failed v2 result remains preserved and requires a newly reviewed version to alter the plan, not repeated runs to seek success.

## Single native command, approval required first

After explicit human approval of this amendment **and its 144 disposable optimizer steps**, update the native checkout to the reviewed commit. In that checkout:

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 /Users/akilan/Documents/Codex/2026-10-08/files-pasted-by-the-user-mindscape/outputs/mindscape/work/final-venv/bin/python engineering/r5_4_v2/qualify.py --approve-engineering-qualification-v2 --output results/r5_4/engineering-v2-native
```

This command is documented and unrun. It accepts no training command, never invokes `fit`/campaign/evaluation, requires committed clean v1/v2 hashes and creates a new evidence directory. It runs one bounded qualification schedule and read-only audit. Review its result rather than rerunning a miss. Independent later re-audit uses `audit_qualification.py` and never constructs a model or optimizer.

**Decision request:** approve or reject the frozen v2 engineering qualification method and its bounded disposable-step execution. A later successful qualification does not authorize amendment adoption into the learning entry point, nor the six-fit neural training campaign. Those require explicit subsequent human instructions. Current authorization ends at preparation and freeze.
