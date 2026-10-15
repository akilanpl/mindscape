# October 21 demonstration and offline operation

Use the final selection in HYBRID_CANDIDATE_C_RESULTS.md. Always retain the frozen Candidate B commit and original working offline delivery as fallback. The native research architecture and pretrained/tool extension are separate systems.

## Reproducible installation

Use Python 3.12 in the extension's independent environment. From the repository feature branch:

```sh
cd integrations/hybrid
./setup.sh
# Check downloaded official revision/file identity:
(cd models/Qwen3-0.6B && shasum -a 256 -c ../../model.SHA256SUMS)
./start.sh
```

The single model is official `Qwen/Qwen3-0.6B`, revision `c1899de289a04d12100db370d81485cdf75e47ca`, Apache-2.0, 596,049,920 parameters. Download and dependency installation require network once. Keep its downloaded LICENSE. Weights are about 1.50 GB and are excluded from Git. The launcher uses local files, Hugging Face offline flags, CPU by default, and loopback port 8765. Stop another local service first or set `MINDSCAPE_PORT=8772`.

Existing installed resources can be mounted without copying or downloading:

```sh
MINDSCAPE_PYTHON=/path/to/environment/bin/python \
MINDSCAPE_MODEL_PATH=/path/to/Qwen3-0.6B ./start.sh
```

Open `http://127.0.0.1:8765/` (or the selected port). Confirm `/api/health` reports pretrained ready. For the backup use `MINDSCAPE_TOOLS_ONLY=1 ./start.sh` and explicitly disclose deterministic solver-only operation. No hybrid launcher runs during native research execution.

Optional external R6 research demonstration requires `MINDSCAPE_R6_REPOSITORY` and `MINDSCAPE_R6_ARTIFACTS`. With no mounts it correctly reports unavailable. It is never needed for verified CBSE answers. The original offline delivery retains its separately packaged native demo.

## Five minutes

1. **0:00–0:40:** State that Qwen supplies advisory interpretation and deterministic tools verify a bounded structured problem. Original native research remains independent; no training was done.
2. **0:40–1:30:** Ask `Differentiate exp(sin(2*x))*cos(x^3) with respect to x`. Show the literal interpretation, product/chain-rule answer, worked steps and tool provenance. Neural plans are advisory, not proofs.
3. **1:30–2:20:** Submit the explicit induction schema below. Explain SI conversion, flux normal, average emf and Lenz sign.
4. **2:20–3:15:** Submit the Python program below; show exception handling, final output and bounded trace. No CPython exec or host file/network access occurs.
5. **3:15–4:00:** Ask for current without circuit measurements; show a safe refusal. State complex pathways need explicit schemas and no full syllabus mastery is claimed.
6. **4:00–5:00:** Show matched hard benchmark subject totals and remaining limitations. Explain that native R6 cannot accept comparable general CBSE inputs. Demonstrate native independence using the original repository's `PYTHONPATH=src python -S scripts/run_episode.py 19 3` if needed; this verifies a deterministic foundation episode, not R6 neural learning.

```json
{"subject":"physics","task":"induction","turns":100,"flux_initial":{"value":2,"unit":"mWb"},"flux_final":{"value":-1,"unit":"mWb"},"duration":{"value":20,"unit":"ms"}}
```

Expected average emf: **+15 V** with respect to the supplied flux normal.

```python
s=0
for x in [3,0,-2]:
 try:
  s+=18//x
 except ZeroDivisionError:
  s+=4
 finally:
  s+=1
print(s)
```

Expected stdout: **4**. These are public development examples, separate from the protected benchmark.

Pause briefly between programmatic requests. The inherited loopback server may reject an immediately following request before its previous request slot is released; the measured hard campaign uses 150 ms post-response spacing and never retries. Do not expose this classroom service publicly as a hostile-code execution service.

Rollback/removal commands are in HYBRID_REMOVAL_GUIDE.md. Before the venue, install everything, verify hashes, rehearse with network disabled, retain B, and avoid last-minute core changes.
