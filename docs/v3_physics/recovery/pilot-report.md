# Corrected native MPS pilot qualification

Run: `results/v3_physics/native-pilot-1791811351750339000`. Model source identity: `1b74f4654ab1a6b1ea7f57abe5e66f8427f1a69250cd44ca5f1b4b546249b86a`. Independent pilot dataset seed20261013; frozen primary seed20261012; no task-identity overlap. Previous diagnostic and failed pilot evidence remain unchanged.

Engineering decision: **GO for the frozen full simulated study**. Scientific conclusion: **NOT ESTABLISHED**. Full study is pending; no target attainment or reasoning breakthrough is claimed. REAL-WORLD VALIDATION: NOT RUN. PHYRE: NOT RUN.

Native Apple M5 execution used PyTorch2.14.1, float32 and selected batch128. Actual allocation, forward/backward, optimizer updates and placement verification passed. No CPU neural fallback. Models have38,409 parameters. Two-trajectory pilot training comprised12 updates per dynamics model; tiny measured six-update invocation times were0.028–0.069 seconds for Mindscape, and should not be extrapolated into full-study throughput estimates.

Independent verification recomputed40 forecast episodes across four scenes, replayed44 actual selected-action planning episodes and verified16 study checkpoint indices. Separate serialization inspection safely loaded20 genuine checkpoint generations with `weights_only=True`. Both families' native interrupted6+resumed6 versus uninterrupted12 runs had exactly identical parameters, Adam optimizer state, memory tensors, training step and experiment provenance. CPU tensor loading in that inspection serialized/compared evidence only; it executed no neural model.

| Family / split | Mindscape normalized rollout MSE | Matched neural MSE |
| --- | ---: | ---: |
| Motion IID | 7.4769 | 165.3479 |
| Motion OOD | 5.9834 | 156.8993 |
| Interaction IID | 24.6201 | 7.4335 |
| Interaction OOD | 44.2457 | 19.9043 |

These are one scene per cell and are engineering smoke results, not credible population estimates. The interaction mechanism performs worse than its matched model here. Mindscape one-step MSE was0.6316/1.0961 for motion IID/OOD and0.7121/1.1875 for interaction. Long rollout error remains large; learning elementary local dynamics does not establish stable multi-step composition.

The original failure was `torch.torch_version.TorchVersion` embedded in checkpoint provenance. Version metadata now uses `str(torch.__version__)`, and the checkpoint serialization boundary normalizes string subclasses. No unsafe pickle loading or checkpoint validation bypass was introduced. All current genuine checkpoint identities contain plain Python metadata and load through the restricted loader. A fresh pilot was used because source identity changed; failed checkpoints were not forced into the corrected run.

Remaining research limitations include small simulated circles/floor environments, at most three objects, limited hidden-parameter inference, heuristic concept features, approximate accumulated uncertainty, and separate teacher information for the model-free policy. The supplemental analytical contact/spring baseline is approximate and is explicitly nonneural. Native hardware audit throughput is a synthetic MLP microbenchmark, not world-model throughput.

The frozen full study requires seeds11/23/37 and budgets10/25/50/100/250:6,000 primary forecasts,6,000 matched forecasts,6,000 ablation forecasts,1,200 nonneural forecasts and2,160 planning episodes. Independent exact-cell coverage, raw metric recomputation, selected-action replay and paired scene-level statistical analysis must finish before the study is declared complete.
