# Candidate C release result

**Select Candidate C for the documented bounded hybrid scope; keep Candidate B as the venue fallback.** No native architecture is replaced. Release code commit: `a23954a`; later commits add documentation/model hashes only.

- Matched hard36: B **14/36**, preformatter C **36/36** answers, gain **22 correct / +61.11 percentage points**. Consistent worked responses were B14/36, C35/36; C's Bayes equality formatter was corrected separately.
- New protected release12: B **3/12**, C **12/12**, pretrained **1/12**; C worked text12/12. This is the fresh final evidence for the corrected formatter version.
- Reused hard regression after correction: C36/36. Reused historical supported regression: B31/36 -> C36/36; both preserve4/4 refusals. Additional safety B8/10 -> C10/10.
- Original historical B acceptance remains31/36 hybrid vs16/36 pretrained,4/4 refusals,p953.46s. It is a separate campaign and cannot be subtracted from C's new final scores.
- Release tests: **173 passed**; original native suite **146 passed /22 skipped**,6 subtests; final extension-removal suite matches native results. No core source/config/checkpoint/protocol changes.
- Real offline model-backed proof passed; zero fitting/optimizer updates. Nine official model files, delivered B,1,306 R6 and90 R6.1 evidence files match hashes.

The replacement gate passes for measured answers, matched regression, refusal, isolated architecture, startup, offline operation and predeclared latency bounds. C's hard warm p95 **8.80s** versus B **5.85s** is slower, within the declared bound; no speedup is claimed. Complete timing/memory limits are in HYBRID_HARD_CBSE_BENCHMARK.md.

C's additional capability is mainly deterministic, schema-driven computation—not new neural learning. The model weights, advisory prompt and decoding settings did not change. No full Class12 mastery, independent human validation, general Python execution, unrestricted natural-language circuit analysis or public-hosting security claim is made. Arbitrary imports/files/network/attributes, active circuits, general ODEs and unspecified databases remain excluded. The inherited immediate-next-request HTTP429 race requires brief post-response spacing and remains documented.

Source is optional under `integrations/hybrid/`; delete that subtree to remove it. Frozen B remains available at immutable SHA `78c5771490491a2d1dc58e7baebdaf2ee5efd22a` on baseline branch `baseline/hybrid-candidate-b-20261015`; original offline B delivery is unchanged. Do not merge PR31 automatically.
