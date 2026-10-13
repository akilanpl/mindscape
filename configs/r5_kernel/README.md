# Request/Response contract: mindscape.kernel/1

Run from the repository root with `PYTHONPATH=src:work/physics-deps/site-packages` and the existing project Python.

- `python -m mindscape.r5_kernel --request configs/r5_kernel/example-math.json`
- `python -m mindscape.r5_kernel --request configs/r5_kernel/example-coding.json`
- `python -m mindscape.r5_kernel --request configs/r5_kernel/example-physics.json --verifier configs/r5_kernel/example-physics-verifier.json`
- `python -m mindscape.r5_kernel --capabilities`
- `scripts/r5_kernel/native.command` runs the one fixed native engineering smoke. Existing output prevents an automatic rerun.
- `python scripts/r5_kernel/verify_smoke.py results/r5_kernel/native-smoke` checks completed receipts without neural inference.

Request fields: `version`, `domain` (mathematics/coding/physics), `operation`, `mode` (validated/experimental_neural/experimental_shared), `observation`, `action`. Defaults are action={},mode=validated,version=mindscape.kernel/1; domain,operation,observation are required. Unknown fields reject. Errors return REJECTED with no fallback. Public physics observation has history[4,3,10],masks[4,3,10],contexts[T,4]; action contains schedule[T,3,4],1≤T≤64. Hidden fields must be zero and masked. Private scene and prior actions go only in the separate verifier file.

Math validated operations: integer,division,fractions,algebra,equation,matrix,geometry,calculus,statistics,probability. Shapes/bounds follow R5.2 capabilities. Math action is empty. Neural integer mode never silently substitutes symbolic output. Coding validated operations:execute/repair. Execute requires repository+visible_tests;repair adds contract and a separate private_tests verifier context. The original WASI and Qwen policy remain intact.

Shared `transition` uses domain-qualified entities,observed masks,relations in observation and an8-number encoding in action. Use `mindscape.r5_kernel.shared.encode(domain,public)` for existing V5 register,AST-edit or impulse adapters. Unknown/hidden target fields reject. This operates on bounded microtask graphs,not arbitrary Python repositories or a universal physical ontology.

Response includes request_id,domain,state,action,prediction,execution,verification,provenance,trace and version. Prediction kinds distinguish neural,symbolic,procedural and submitted programs. `execution_verified` means actual execution/replay consistency; `success` means the stated verifier criterion. `success=null` means untested. Original prediction values are never overwritten. Semantic provenance includes `shared_parameters`, `learned_transfer=false`, tool/checkpoint identity and zero optimizer steps.
