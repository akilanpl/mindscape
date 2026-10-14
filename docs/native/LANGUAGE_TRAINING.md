# Stage B language learning preparation

`language_learning.py` adds actual intent cross entropy and argument start/end
pointer cross entropy for the original `language_model` encoder. Random forward
and backward checks connect embedding, recurrent encoder, intent and both pointer
heads. No optimizer steps, fitted checkpoint, qualified interpreter or generated
free-form answer is produced by this work.

The fixed dataset builder renders annotated templates for add, local borrowing,
identifier rename, constant-force update, tool requests and unsupported requests.
Arguments are annotated during rendering, without solving downstream problems or
consulting transition targets. Text alone enters the encoder; protected intent and
span labels enter only the loss. `requested_name` is an observable instruction
argument; protected transition `target_name` is not a language slot. Missing slots
are supervised at BOS index zero, so the model learns an explicit null location.

Dataset partitions separate training forms, unseen paraphrase templates, reordered
clause/bracket structures, and new lexical/operand combinations. Surface forms and
template IDs are disjoint; training and held-out numeric sequences differ in the
structure/lexical partitions. The lexical partition includes unseen identifiers
and words. `FrozenVocabulary` is created exclusively from training text, so unseen
words map to unknown IDs. Unknown identifiers cannot be recovered from token IDs;
a future qualified pointer interpreter would need original-text boundary tracking.
These are synthetic, template-labelled diagnostics. They are not a human-audited
language benchmark or proof of semantic, task or open-language generalization.

`prepare_language_package(root, directory)` writes frozen data, train-only
vocabulary and source/protocol hashes. It never constructs a neural model or
optimizer. `train_language(...)` is future infrastructure only: it first calls the
existing `learning.validate_gate`, requiring valid external authorization and an
independent qualification signature over the exact source/data freeze. It further
requires the language stage, both language source files, vocabulary hash and
bounded optimization budget. Only after these checks can it construct the encoder,
Adam optimizer and actual loss/backward/update loop. Nothing invokes that path by
default. No signature or qualification is self-issued.

Checkpoint serialization reuses safe JSON tensor storage, never executable pickle.
`replay_language_checkpoint(path, expected_hash, rows)` checks the supplied hash,
restores the frozen vocabulary and original encoder dimensions, and returns raw
intent/span predictions, actual loss, intent counts and exact-span counts. Replay
is read-only and always unqualified. The engineering fixture serializes random
weights with `trained=false` and `optimizer_steps=0`; it is not learned evidence.

Tests run random forward/backward connectivity, split/vocabulary isolation,
protected-label separation, pre-optimizer gate rejection and deterministic random
checkpoint replay with tamper rejection. No optimizer is constructed during these
tests and no training campaign is run.

The frozen manifest also binds deterministic algorithm use, each evaluation split
hash, fixed final-checkpoint selection, exact intent/span counting metrics, and the
limited synthetic-language scientific scope. Preparation creates unsigned
`authorization.UNISSUED.json` and `qualification.UNISSUED.json` templates with
`issued=false`, empty issuers/signatures, and the exact freeze hash. Existing
`native-r56-training` signature scope is reused; the signed manifest separately
binds the language stage. Templates grant no permission.

The guarded future fitting loop rejects nonfinite losses, missing/nonfinite
gradients, and nonfinite post-update parameters. Failures record the exact number
of completed optimizer steps, freeze identity and error type in a failure sidecar.
No failure is reported as success. Read-only replay reports checkpoint and evaluated
split hashes and scores both all slots and present slots separately, so absent-slot
null predictions cannot obscure argument extraction errors. These facilities were
tested using random backward and replay fixtures only; the optimization loop and
its failure evidence remain unexecuted because training has not been authorized.
