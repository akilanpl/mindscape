# Three-domain integration

One typed Request/Response API coordinates the existing specialists through the original V3 Kernel. The integration branch combines mathematics PR #2 and coding/physics/V3.1 dependencies, preserving their sources and evidence. No website changes or training are included.

Native execution:
    scripts/v3_integration/native.command --domain all
Selective execution uses --domain mathematics, coding or physics.

The compact mathematical checkpoint predicts local claims and signs on MPS; the preserved signed-magnitude/digit scheduling scaffold composes them. Actual execution writes neural claims without arithmetic correction. Independent arithmetic is consulted afterward.

Coding reuses local Qwen, approved structured memory, the learned proposal evaluator and existing path-schema transport. Both Qwen and evaluator execute on MPS. Initial visible tests inform repair; generated code and terminal private tests execute exclusively through the preserved WASI Executor. Private scorer results never feed generation or persistent memory.

Physics reuses the corrected V3.1 checkpoint and bounded nine-candidate planner. Its observation history is collected from the same initial zero-action trajectory used by execution/replay. Independent simulation executes the selected action; prediction MSE and actual goal success remain separate. Prior scientific NO-GO and uncertainty limitations remain unchanged.

Mathematics and physics macro routing always submit the specialist proposal rather than selecting a semantic answer. This routing is not learned reasoning or transfer; neural numerical claims/actions remain untouched. Coding candidate ranking uses the preserved learned evaluator. Shared memory is not pooled across domains and evaluation cannot update training memory.

The tiny frozen request suite compares direct specialists with unified access; it establishes engineering equivalence only. Learned cross-domain transfer is NOT YET ESTABLISHED. No shared-learning training or final-holdout experiment is run.

Verification:
    PYTHONPATH=src:work/physics-deps/site-packages python scripts/v3_integration/verify.py --run <native-directory> --output work/integration-replay.json

Existing reports: docs/v3_math/scientific-report.md, docs/v3_coding/scientific-report.md, docs/v3_1_physics/diagnostic-review.md. Assets remain external and must be available at the pinned paths in assets.py; no model downloads or CPU neural fallback occur.
