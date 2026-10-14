# Trainer/evaluator isolation: fail-closed deployment candidate

Actual host result: macOS sandbox-exec fails `sandbox_apply: Operation not permitted`
(exit71); Docker and Podman are unavailable. **OS enforcement is NOT demonstrated.**
No contract test, separate file or staged directory counts as an isolation pass.
Scientific decision remains NO-GO. This package is deployment-ready for review and
probe execution, NOT deployment-qualified by an actual container run on this host.

`isolation.stage_training` stages only hash-verified512 training rows and a receipt.
It copies no evaluator targets, source generators, verifiers, credentials or results.
It rejects traversal, file/directory symlinks, changed data and unexpected staging
contents. Read-only world-readable training files permit UID10001 container access;
the isolated mount, not host directory permissions, is the intended boundary.

`container_command` requires an operator-reviewed immutable image digest, nonroot
UID/GID10001, no network, read-only root and training mount, no capabilities,
no-new-privileges, bounded CPU/RAM/PIDs/FDs, noexec tmpfs. Trainer cannot request
an evaluator or checkpoint mount. Evaluator has distinct readonly target/checkpoint
mounts; no output/feedback mount is returned to trainer. Vault symlinks are rejected.
Subprocess launch uses a clean environment, shell=False/default and close_fds=True.
The image must be independently inspected; arbitrary images are not trusted by a
regex. Docker daemon/socket privileges are outside the container and must remain
with the independently controlled evaluator coordinator, never the trainer.

The adversarial `probe.py` checks actual training readability, absence of heldout
canaries (including traversal/symlink attempts), readonly data, nonroot UID, no
extra descriptors, clean env and network denial. Host launcher tests validate clean
env/closedFD contract but do not claim kernel denial. Run on a qualified Linux Docker
host with separately sealed data and checkpoint provenance, store all original
probe stdout/exit status/runtime version/image digest, and independently verify the
canary was accessible to evaluator but inaccessible to trainer. This final positive
control and evaluator/checkpoint execution must be qualified before scientific GO.

Example engineering-only invocation (no fitting entry point):

```python
from mindscape.r56.isolation import stage_training, run_probe
stage_training('/reviewed/proposal', '/new/trainer-stage', 17, '<independently-reviewed-package-sha256>')
# image must be independently reviewed and pinned; no mutable tag permitted.
receipt = run_probe('python@sha256:<actual-reviewed-digest>',
                    '/new/trainer-stage', '/reviewed/probe.py')
```

Do not invent an image digest, run a host-interpreter fallback, install permissive
Docker privileges into a trainer, or promote a probe receipt into authorization.
The existing scientific trainer does not silently migrate to this boundary. Until
an isolated fitting implementation and independent evaluator are actually qualified,
the current NO-GO gate blocks it. Original72fits/144000updates remain reserved.

Independent review found and repaired missing evaluator-positive control and a weak
network timeout check. run_isolation_pair now verifies target/checkpoint SHA256
readability in the evaluator plus trainer denial; probe.py checks the Linux network
namespace exposes only loopback and no nonloopback route, not a TEST-NET timeout.
Probe source is hash-bound before/after. All receipts retain qualified=False: a
paired engineering pass would still need external staging/daemon and actual fitting
qualification. Same-UID malicious host mutation/ancestor TOCTOU is not prevented by
path checks; independently owned immutable staging is a mandatory unmet requirement.
