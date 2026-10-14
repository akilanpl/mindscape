# Native language boundary: two explicit stages

Stage A is a deterministic typed grammar, **not learned language understanding**.
`mindscape.mind1.language.interpret(message, history=None)` never produces an answer.
It returns `supported`, `kind` (`native`, `tool`, `unsupported`), typed `problem`,
`interpretation: deterministic-grammar`, and limitations. Tool routes additionally
return `tool: {name, args}`. History is not concatenated or used to guess missing
operands; context-dependent pronouns abstain.

Supported native forms include:

- `What is 27 plus 18?`, `add 27 and 18`, `27 + 18`: nonnegative base-10 integer
  addition, operands preserved as decimal strings, at most 18 digits each.
- `7 minus 9`: a **digit borrow transition**, not a signed integer answer. Inputs
  are `left`, `right` and `borrow_in=0`; result interpretation belongs to the
  domain engine. General multi-digit subtraction is not supported here.
- `mass 2 kg; force (4, 6) N; time 3 s; initial velocity (1, -2) m/s`: typed
  constant-force Newtonian vector problem. All four fields must occur exactly
  once. Mass is positive, duration nonnegative, magnitude at most 1e6. Scalar
  force/velocity use the x axis. Recognized non-SI units and additional numerical
  constraints abstain. This is a narrow grammar, not a general unit parser.
- `rename variable x to y in this function:` followed by one fenced Python
  function: bounded pure arithmetic function only. No calls, imports, attributes,
  control flow, nested functions, decorators or annotations. No execution occurs.
  Existing requested target names are rejected to avoid capture. Candidate names
  are bounded at sixteen. `requested_name` is the desired identifier explicitly
  supplied in the user instruction and is legitimate task input; it is not a
  computed renamed program or a verifier result. `target_name` is a **protected verifier-only field**;
  it must not enter neural proposal scoring features or training input tensors.

Tool assistance is explicitly distinct: calculus (`differentiate`, `integrate`,
`simplify`, `solve`) produces `math_calculate` arguments; fenced inspection produces
`coding_analyze`; explicit Python execution produces `coding_execute`, whose WASI
availability is checked by the tool; `simulate projectile|oscillator|rc_circuit`
followed by a JSON parameter object produces `physics_simulate`. Interpretation
alone does not execute tools or certify their arguments/answers. Malformed, unsafe,
unrecognized, ambiguous or open-ended requests return unsupported with no invented
answer. The bounded grammar does not meet general assistant language coverage.

## Stage B: original trainable candidate, unavailable for product inference

`mindscape.r56.language_model` defines an original compact vocabulary/tokenizer and
a randomly initialized embedding + bidirectional GRU encoder. It has an intent head
and independent argument start/end pointer heads. These produce logits, not solved
problems or natural-language answers. No pretrained weights, hosted model, external
tokenizer or pretrained language representation are used. Torch is optional; missing
torch produces an explicit unavailable error.

`FrozenVocabulary.from_training_forms` must receive **training forms only**. The
vocabulary is immutable, includes PAD/UNK/BOS, records training-form and vocabulary
hashes, and never adapts while encoding validation or held-out paraphrases. Tokens
are bounded ASCII English word/punctuation/digit pieces; unseen pieces map to UNK.
Truncation and unknown-token counts are explicit. Argument slots cover operand and
physics fields plus the old and explicitly requested variable names. Protected
verifier labels and transformed-source answers have no argument head.
Padding masks are validated and packed recurrence prevents padding length from
changing contextual features.

`validate_split` rejects identical normalized surface forms across training and
held-out sets. It does **not** establish independent operand/template/semantic
splits. A future authorized protocol must freeze training-only forms, operands,
paraphrase families, seeds, tokenizer and qualification gates before training;
held-out targets must stay outside encoding, vocabulary fitting, checkpoint choice
and model inputs. Evaluate intent and exact argument extraction separately, with
abstention and adversarial malformed/out-of-scope sets. No campaign, optimizer
steps, dataset generation, checkpoint qualification or held-out tuning is run by
these modules. A unit-test gradient confirms trainability only.

The candidate advertises `UNTRAINED — LEARNED LANGUAGE INTERPRETATION UNAVAILABLE`,
`qualified=False`; its `interpret` method always raises an unavailable error.
A future training and independent qualification process is required before learned
interpretation can become a product route. Existing Stage A acceptance tests and
random encoder shape/gradient tests cannot demonstrate R5.6 language generalization.
