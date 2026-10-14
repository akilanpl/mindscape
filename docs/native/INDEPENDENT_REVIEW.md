# Independent engineering review of native correction

Review date: 2026-10-14. Scope: the current native integration sources and actual
bounded-worker requests. This is an engineering/security review only. It issues
no scientific qualification, training authorization, signatures, benchmark result
or generalization claim. No optimizer campaign was run.

## Path inspected and exercised

`interpret` produces typed operands through a supplied deterministic grammar.
`Mind1Orchestrator` supplies bounded operation schedules and explicit concept IDs,
then `NativeMindscapeRuntime` invokes `native_transition` in the bounded trusted
worker. The worker constructs the original randomly initialized `R56Model`, applies
its graph encoder/message/recurrent core and learned-parameter output heads, and
returns the resulting predictions. The real path was exercised for addition,
pure-function rename candidate selection, and a two-component constant-force
velocity task. Each returned `engineering_demo`, a parameter count, checkpoint/
weight identity, original prediction and separately labeled post-hoc verification.
Invalid legacy external-model environment settings had no effect and appeared in
no response. These observations establish wiring, not competence.

The graph does not consume `target_name` or arbitrary label dictionaries. The
explicit user-requested rename identifier is legitimate task input and is separate
from the protected candidate target/verifier answer. Addition traversal uses the
model's own predicted carry at the next digit; it does not feed back the exact
arithmetic reference. A separate test replaced the verifier with one that clears
its received prediction: the returned original prediction remained intact. Model
state tensors stayed exactly unchanged, and backward/optimizer constructor calls
were prohibited during that test.

Normal native requests have no qualified checkpoint and explicitly abstain or use
a separately attributed deterministic tool. Engineering demonstration requires
explicit opt-in. A loaded checkpoint remains scientifically unqualified: a saved
trained flag cannot issue scientific acceptance. Stage B language interpretation
is unavailable; its random original encoder remains untrained and `qualified=False`.
The product still has supplied parsing, concepts and scheduling. It does not provide
general free-form native language reasoning.

## Security and research boundaries

Production inference no longer imports a cloud SDK or the archived provider
implementation. The provider module is a disabled tombstone. The native worker
adds a network-connect denial hook and receives a restricted environment without
legacy external API credentials. Tool assistance is explicitly separated.
Historical `sandbox.py` and `wasi_worker.py` are byte-identical to PR27 head; no
host-execution fallback was introduced by this correction. This review does not
replace sandbox correctness or historical preservation audits.

The training entry point calls source/data/signature gate validation before its
Torch import, neural model construction or optimizer construction. The source
freeze covers both complete Mind1 and R5.6 Python source trees, including language,
inference, model, graph packer, training/curriculum and benchmark sources. Gate
validation binds the source inventory, data hash and manifest hash, then requires
externally trusted Ed25519 signatures from distinct authorization and qualification
issuers. Unissued templates and self-asserted serialized flags are not approval.
This review neither created signing keys nor issued either gate artifact.

Checkpoint loading uses JSON tensors rather than pickle. It checks the reviewed
SHA-256 identity, a 64 MB file ceiling, exact state keys, declared tensor shapes and
dtypes, actual reconstructed shapes and finite tensor values before applying the
restored state. Roundtrip and nonfinite/oversize rejection were exercised.

## Remaining gaps and limits

- No trained, independently qualified native checkpoint exists. Random predictions
  can be wrong; original failed predictions must remain visible. Narrow grammar,
  candidate classification, supplied concepts and deterministic scheduling do not
  establish learned reasoning or transfer.
- The reviewed checkpoint loader did not yet validate the top-level/metadata JSON
  mapping shape or the type/range of `optimizer_steps` before applying weights.
  Malformed metadata can cause a sanitized worker failure rather than a controlled
  checkpoint error. Require bounded metadata mapping/types and reject nonfinite
  JSON constants before model state application. This is a robustness gap, not an
  executable deserialization finding; JSON never invokes Python object loading.
- A static no-cloud scan and subprocess demonstrations cannot prove all future
  dynamic/transitive code is network-free. Keep the native worker network denial
  and explicit inference/tool separation intact.
- No scientific evaluation, authorized optimizer campaign, language qualification,
  broad native task mastery, public hosting, or green remote CI result is claimed.

## Executed checks

`tests_native/test_independent.py` covers three full worker demonstrations with
invalid external settings, graph label exclusion, prediction immutability and no
inference updates, gate ordering, original WASI source identity, JSON checkpoint
roundtrip/finiteness/size boundaries, and learned-language unavailability. Results
are engineering evidence only; a passing verifier on one case is not scientific
qualification. The pre-merge run selected the parent integration source tree via
`MIND1_AUDIT_ROOT` and pytest's Python path override.

Parent follow-up: checkpoint payload/metadata mapping and bounded optimizer-step validation, plus nonfinite JSON rejection, were added with five rejection-before-weight-application regressions. The prior robustness finding is repaired; this is engineering validation, not scientific qualification.
