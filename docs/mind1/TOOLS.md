# Mind1 tool contract

`mindscape.mind1.engines.run_tool(name, args)` returns a JSON-compatible envelope:
`{ok: true, tool, result}` or `{ok: false, tool, error: {code, message}}`.
`tool_specs()` publishes OpenAI function schemas. `capabilities()` reports optional
runtime availability, not benchmark performance. External tools must remain distinct
from learned model capability and from held-out research evaluators.

- `math_calculate`: expression, operation (`evaluate`, `differentiate`, `integrate`,
  `solve`, `simplify`), variable, optional numeric values. Uses optional SymPy;
  constructs objects from an allowlisted AST, never evaluates raw text. Variables
  x/y/z/t; constants pi/E; functions sin/cos/tan/exp/log/sqrt/Abs. Expressions are
  capped at 512 characters, 100 nodes and nesting 16; literal powers at magnitude
  12 with numeric growth bounded. Integration accepts polynomials to degree 12;
  solve accepts univariate polynomials through degree four.
- `math_matrix`: numeric matrix up to 6×6 and operation determinant/inverse/rank.
- `physics_simulate`: model, parameters, steps (1–1000). Projectile parameters:
  speed, angle_degrees, gravity, height, duration. Oscillator: mass, spring,
  damping, position, velocity, duration. RC: resistance, capacitance, voltage,
  initial_voltage, duration. Numeric values are finite and bounded. Models use SI
  units. Projectile has constant gravity/no drag; oscillator uses fixed-step RK4
  with a conservative stability guard; RC is ideal analytic charging. Returned
  trajectories are simulations, not observed evidence or learned predictions.
- `coding_analyze`: source (64KB maximum), language. Python uses AST syntax
  validation; other languages receive explicitly heuristic lexical risk checks,
  never a claim of compilation or comprehensive security validation.
- `coding_execute`: Python source (64KB maximum), using the existing fuel/memory/
  file/output/wall-time bounded WASI sandbox and capability probe. Requires
  optional wasmtime and a configured Python WASI runtime. Set `MIND1_WASI_RUNTIME`
  (or legacy `MINDSCAPE_WASI_RUNTIME`). Missing runtimes fail closed. Guest source
  never runs under host Python; no host-execution fallback exists.

The API caller must impose a process wall deadline on each tool request, especially
symbolic operations. Parser bounds reduce resource exposure but do not replace a
wall deadline for computer algebra. Optional package installation is explicit;
missing SymPy does not disable physics or static analysis.

`math_numerical` adds two bounded numerical methods using the same expression parser
and optional SymPy. `operation=ode` treats the expression as `dx/dt=f(t,x)` with
initial (default 1), start/end (0/1), steps (1–400), and returns a fixed-step RK4
trajectory. There is no adaptive accuracy certificate or stiff-equation guarantee.
`operation=minimize` takes an expression in x and lower/upper bounds and runs at
most 100 golden-section iterations plus endpoint checks. This assumes a unimodal
objective; it does not certify a global minimum for multimodal objectives.

WASI compiled `.cwasm` caches are runtime/platform dependent. When migrating a
runtime, copy its `python.wasm` and `lib` into a fresh local directory and let the
current wasmtime build its own cache. Do not reuse an unverified historical cache.

Verification metadata reports actual checks: polynomial integration differentiates
the result against the original input; equation solving substitutes every returned
root; matrix inversion checks the product against identity using exact rationals
from validated original numeric literals. Differentiation includes central finite
difference spot checks explicitly labeled as evidence rather than a proof.
Symbolic simplification uses SymPy simplification under the caller's wall deadline.
Physics reports projectile energy/impact residuals, oscillator energy conservation
or dissipation checks, and RC initial-condition/bounded-limit checks. These checks
can detect internal inconsistencies; they do not validate model assumptions against
the physical world or certify numerical accuracy for every input.
