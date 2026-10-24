# Independent CPU evaluation

`python -m cloud.kaggle.evaluation.worker --manifest evaluation-manifest.json --output /kaggle/working/evaluation-v1`

The worker and Kaggle runtime-isolation adapter are implemented and locally tested. The generic CLI fails closed without a provider verifier. The trusted controller can submit a separate private offline CPU Kaggle execution that calls `run(..., trusted_control_plane_verifier=kaggle_verifier(...))`. The verifier digest must be pinned after authenticated remote metadata/source retrieval and package inventory checks. To avoid circular hashes, evaluator `source_sha256` means the stable core entry file (`source_identity_scope=evaluator-core-file`, `source_path=cloud/kaggle/evaluation/worker.py`), and `core_source_files` pins the extracted source inventory. Pass `core_root` to `kaggle_verifier`; it verifies those bytes itself. The generated launcher is excluded from that inventory, and its complete remote source hash belongs in a separate postrun controller receipt. JSON flags alone are insufficient. See `isolation.py` and the `controller_evidence()` test fixture for the exact evidence contract.

The protection boundary is the **frozen trainer runtime**, with the account owner retained as trusted controller. Distinct Kaggle kernel IDs/versions, separate VM execution, no evaluator attachments in the trainer, offline execution and no training credentials in the evaluator provide a practical scoped separation. The adapter verifies the controller's pinned evidence, Kaggle Batch runtime, observed absence of known credential environment variables/files, and effective read-only checkpoint access under `/kaggle/input`. It attempts an `O_WRONLY` open without `O_TRUNC`, `O_CREAT` or writing bytes, and rejects any mount that permits it. The controller must verify remotely pulled source/package inventories to prove no final cases or credential/secret-access code were supplied to the trainer. Provider metadata alone does not inventory all secret grants; observed credential checks do not prove the absence of every possible secret or covert access route.

Separate account ownership is not required for this scoped runtime threat model. The trusted owner can change future notebook versions, attach evaluation data later, or retrieve protected data. This pipeline does not claim resistance to an adversarial owner or independently governed research evaluation. Hashes pin finalized checkpoint bytes, not authorship or secrecy. Repeating an evaluation after owner-directed training changes must be reported as adaptive historical reuse.

The exact previous `data-v1/evaluator-only/final-{17,29,43}.json` files are **historical reused evaluation data**. Their manifest's `EXPLORATORY_UNTOUCHED_AT_GENERATION` label describes their original generation, not their present status. They were evaluated in the prior focused-correction experiment. Zero train overlap does not turn them into a new untouched distribution. Reusing these bytes permits a distribution-matched CUDA comparison only. A genuinely new protected test distribution needs independent preparation, protocol review and controller custody before further fitting or model selection.

Manifest schema (`mindscape-independent-evaluation-v1`):

```json
{
  "schema": "mindscape-independent-evaluation-v1",
  "source_files": [{"path": "src/mindscape/r57c/ordinal_model.py", "sha256": "<actual digest>"}],
  "data_files": [
    {"path": "train/17.json", "sha256": "<actual digest>"},
    {"path": "evaluator-only/final-17.json", "sha256": "<actual digest>"}
  ],
  "checkpoints": [{"path": "checkpoints/17-graph-concept-on/final.json", "sha256": "<actual digest>", "version": "R57c", "seed": 17, "condition": "graph-concept-on"}],
  "evaluations": [{"seed": 17, "training_path": "train/17.json", "evaluation_path": "evaluator-only/final-17.json", "distribution_id": "r57c-historical-focused-correction-data-v1", "historical_evaluation_reuse": true}],
  "isolation_evidence": {"path": "controller-attestation.json", "sha256": "<actual digest>"}
}
```

Paths resolve against the manifest directory. Include **all** source files imported from `mindscape` (package `__init__.py` included), all nine finalized checkpoint pins, and three evaluation jobs. The example is schematic; placeholder hashes are rejected. Checkpoints must be safe `r56-json-tensors-v1` JSON, R57c, assigned seed/condition, exactly 2,000 recorded updates, exact expected tensors and finite values. Hashes are checked before model imports/data decoding, and again after inference. No pickle or dynamic user-supplied predictor is loaded. The evaluator imports the frozen R57c architecture, which defines a training-loss helper, but never imports or calls a training runner, learning/checkpoint helper, optimizer constructor, backward operation or that loss helper. PyTorch itself may load optimizer library code internally; this is not a sandbox guarantee. No optimizer executes.

Development APIs (`evaluate`, `perturbations`, `overlap_check`, `baseline_results`) can be unit tested without isolation claims. The production worker uses CPU tensors, disabled parameter gradients and `torch.inference_mode`. It preserves raw output heads alongside predictions/references and emits per-domain metrics, coding position/scope metrics, force/mass/time probes, operand/carry probes, candidate rotations and requested-name changes. Derived probes never repair primary predictions. Analytic-rule and training-majority/constant results are explicitly non-neural controls. Duplicate checks normalize incoming/carry aliases and ignore annotation fields. Final duplicates and train/final overlap are rejected. Repeated training contexts are reported without changing the frozen training protocol (the actual seed-17 training file has 128 rows and 121 normalized unique contexts). Cross-seed training union is compared to each final set. Identical math contexts across final seeds are not independent samples; seed scores remain separate.

R56/R57 comparisons require independently reviewed source-pinned adapters and identical distribution IDs; this worker rejects cross-version checkpoint entries instead of importing the historical scoreboard (which imports training helpers). Raw historical reports can be compared externally only with verified matching evaluation hashes. No superiority, new-distribution generalization or CBSE competence claim is justified by this infrastructure.
