# R5.4 implementation and freeze report

Readiness: **ENGINEERING_NO_GO**. Data qualification passed and the experiment is frozen and reviewable. Native MPS is built into the installed PyTorch 2.14.1 but reports unavailable in this execution session. Native gradient placement and resource feasibility therefore remain unrun. This is an environment readiness result, not evidence against either learning hypothesis. No neural experiment is labeled PASS.

## Repository preservation

Inspected existing local checkout: branch `chore/remove-mindscape-vercel`, clean, HEAD `9162dcc`. Left that checkout untouched and created a separate workspace clone. Fetched all remote branch history. Remote integration head `development/r5.3-robust-physics` is `6c558caa3c3a1a09ec15e286a408cbb6d5076b03`, including merged unified-kernel PR15, README PR16 and deployment-removal PR17. Remote main is `f9955c7b8043a00ebea36fffd41c646d1d54f2bd`; their common ancestor is `aafa49e8d2fadaf46ded12cd686c72e739b217d9`. Main is not the latest research integration snapshot. Created `research/r5.4-compositional-rule-learning` from verified integration head. Rechecked remote integration SHA before publication.

Only new R5.4 source, tests, documentation and evidence are added. No existing experiment, evidence archive, physics model, kernel, Qwen/WASI specialist, website source or tag is modified. No force push, history deletion, unrelated merge, Vercel deployment or Alpha launch. Source/data/protocol freeze commit: `8082e99`; SHA256 lock includes every R5.4 implementation file, test, protocol and data/outcome artifact. Later readiness reporting does not alter that freeze.

## Hypotheses and limitations

H1: local supervised learning can predict a finite table of digit/carry transitions. H2: known transitions can compose into exact answers at lengths outside training. H3: transition prediction can transfer to held digit/carry combinations. H4: recurrent hidden state improves over a similarly sized stateless predictor. These are distinct hypotheses. H4 is exploratory; the Markov model can legitimately win. The R5.2 recovered recurrent 8.33% OOD, reset-hidden25%, memory21.53%, and intermediate81.75% results motivate a conservative design, with no presumption that recurrence helps.

The scaffold supplies LSD traversal, input padding, stopping and serialization; the learned models supply output digits and outgoing carries. Free inference feeds the model's own carry and supplies no masks, answer features, verifier feedback, oracle correction, retrieved evaluation answers or symbolic fallback. Exact arithmetic occurs only in training supervision, dataset qualification, reference ceiling and post-prediction scoring. The source/API boundary is tested; it is not an OS security sandbox.

Markov has2,188 parameters; recurrent has2,124 (2.93% fewer). Both share22 one-hot observable input features and12 output logits, seeds11/23/47, batch32, AdamW and800-update budget. Final checkpoint selection is fixed at update800. The bounded exposure objective combines teacher forcing with local supervision on each model's own visited carry states in updates401–800. The hard carry choice is detached; recurrent hidden-state gradients span training columns. This tests a concrete objective, but six fits cannot establish a causal exposure-objective advantage over a TF-only ablation. No extra ablation is authorized.

## Data and controls

Allocation is unchanged from the proposed1,024/64/192. Four balanced OOD families distinguish longer known-context operands, long carry chains, mixed carry patterns, and short held-context operands. Independent identities exclude operand commutations. Training has184 unique contexts, development112, OOD192. OOD family unique contexts are175/35/77/77; unseen-short naturally encounters12 of16 held contexts. Reset-state200-context diagnostic probes include all16 held contexts. Excluding both incoming carries for every reserved pair makes indirect held-context training exposure zero even under incorrect predicted carries.

Training-context memory solves48/48 in each known-context family and0/48 in unseen-short:144/192 overall. Whole-problem memory misses every evaluation identity. Zero-weight controls are specified for both architectures. The independently checked procedural reference is an oracle ceiling only. The protocol requires stringent per-seed exact-answer, long-carry, unseen-transition, interval, memory-gain and paired-comparison gates. Full numerical thresholds and their rationale are in `protocol.md` and `experiments/r5_4/frozen/protocol.json`. Repeated seeds do not inflate independent task denominators.

## Executed validation

- Dataset generation and qualification: DATA_READY. Deterministic regeneration, manifest identities, held exclusions, known-family context coverage, memory headroom and independent reference checks pass.
- CPU regression command: `PYTHONPATH=src python -m pytest --import-mode=importlib tests/r5_4 tests/r5_2_recovery -q`:27 passed (23 new R5.4 tests,4 historical R5.2 nonneural regressions). CPU backward checks use only training operands; no optimizer is constructed. Synthetic checkpoint fixtures test evidence schema without training or making learned claims.
- Numerical regression reproduces R5.3's float32/JSON float64 score discrepancy, verifies the explicit precision repair and rejects material tampering. No historical physics model executes.
- Source compilation and Git whitespace checks pass.
- Optional full R5.3 recovery module collection is blocked by absent `pymunk`; its collection failure is retained in `optional-historical-test-collection.txt`. No download was attempted. Shared `test_recovery.py` basenames also require pytest importlib mode.
- Native engineering command executes its capability guard and records ENGINEERING_NO_GO: arm64/macOS, PyTorch2.14.1, MPS built true, available false. Native backward, MPS residency and timing/RSS qualification are unrun. Optimizer updates:0; CPU fallback:false.

## Audit and preservation

Predictions are committed and fsynced before outcomes are read. Condition/seed/model/task/horizon metadata is explicit. Tensor checkpoints, optimizer states,800-step logs, exposure sets, free trajectories, answers and diagnostic outputs are saved. Independent read-only audit rejects incomplete attribution, bad carry chains, incorrect metadata, tampered hashes, nonfinite tensors and optimizer/update mismatches. It independently reconstructs references, replays controls, recomputes integer-count/float64 metrics, Wilson intervals and exact paired tests, and compares saved metrics. Native checkpoint replay is a separate optional read-only command after completion. Preservation archives source/data/lock and campaign files with member hashes and verifies without extraction.

Unrun: all six optimizer fits, neural OOD inference, neural teacher-forced evaluation, learned checkpoint generation, campaign metrics, trained-checkpoint replay and completed-campaign preservation. Tools for these stages are implemented, not claimed as executed or verified on real neural campaign artifacts. Existing regression tests validate components; native end-to-end campaign behavior remains unqualified until MPS is available.

## One approved native training command

From a local checkout of the frozen branch, using the existing native Python environment with installed torch:

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=src python -m mindscape.r5_4 train --authorize-training --output results/r5_4/campaign-v1
```

This is documentation, not execution authorization. Stop before running it. The command first validates committed frozen hashes, data qualification and native backward/resource readiness; it fails before optimizer construction if MPS remains unavailable. Its destination must not already exist. Explicit human authorization is required even after engineering passes. The accompanying output command file supplies this session's exact checkout and installed Python paths.

**ENGINEERING_NO_GO**
