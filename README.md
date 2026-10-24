# Mindscape STEM delivery candidate

Local pretrained neural interpretation plus bounded verified STEM tools, connected to the existing Mind1 web workspace. Native R6 research remains a separate, honestly labeled demonstration. No neural fitting is performed. No Kaggle or network connection is required after dependencies and model weights are installed.

## Start

Use Python3.12+ on macOS arm64. Install requirements into a fresh virtual environment. Download the single official Qwen/Qwen3-0.6B model with scripts/download_model.py; its exact revision and file hashes are recorded in the download receipt. For a frozen reproduction use the delivery's recorded revision, not a future floating main revision. Model loading uses local files and trust_remote_code=False.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
.venv/bin/python scripts/download_model.py --destination models/Qwen3-0.6B --receipt model-receipt.json
export MINDSCAPE_MODEL_PATH="$PWD/models/Qwen3-0.6B"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
.venv/bin/python -m mindscape_delivery.server --device cpu --port 8765
```

Open http://127.0.0.1:8765. MPS is optional; CPU is the venue fallback and is tested on the actual Mac. Run --tools-only as an explicit deterministic backup; it does not claim neural inference. Only loopback serving is enabled; no public accounts or paid services.

## Submit

Natural-language examples supported by the supplied grammar:

- Differentiate x^3 + sin(x) with respect to x
- Integrate x^2 from 0 to 3
- Solve x^2 - 5*x + 6 = 0 for x
- Find the determinant of [[1,2],[3,4]]
- Find the inverse of [[1,2],[3,4]]
- Find current through a 4 ohm resistor with 12 V across it (spaces around quantities recommended)
- Coulomb force between charges 2 uC and 3 uC separated by 0.1 m
- Find the equivalent resistance of 3 ohm and 6 ohm in parallel
- Lens equation, f=20 cm and u=-30 cm
- Predict the output, followed by exact fenced Python code
- SELECT name FROM marks WHERE score>80 tables: {"marks":[{"name":"A","score":90}]}

Explicit JSON specs are also accepted; see solver tests for the contract. Numeric tool verification applies to the displayed structured interpretation; verify it matches your question. A pretrained plan is advisory and unverified. Literal code is never replaced by a neural rewrite. Unsupported or unsafe requests are rejected explicitly. Every response identifies the actual method.

## Architecture

Question → local Qwen3-0.6B advisory JSON → supplied grammar or literal-grounded interpretation → bounded subprocess tool → checked answer and deterministic worked steps → existing Mindscape UI with method/provenance. Pretrained-only and tools-only modes are separately labeled. The pretrained model is never represented as native R6 learning.

Python is interpreted through a small AST whitelist, not exec/eval. SQLite receives provided tables only in a private in-memory database; writes and unsafe functions are denied. Math expressions are converted from a restricted AST to SymPy without evaluation of arbitrary Python. Physics requires explicit units and signed optics. Tool subprocesses have CPU/time limits; mathematical checks may use the same engine rather than independent proof. This is a bounded classroom assistant, not full CBSE mastery or a sandbox for hostile public users.

## Testing

Install pytest, then `python -m pytest -q tests`. The separate acceptance protocol/report and raw predictions are distributed under evaluation/. Its held-out questions are excluded from model/trainer inputs and should not be reused as a tuning set. Candidate guarantees apply only to topics that passed the locked threshold. No optimizer-based training is authorized by this application.
