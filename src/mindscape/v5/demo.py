"""Show saved public graphs and independently executed effects; no neural-success claim."""

import json
from pathlib import Path


def main():
    root = Path("experiments/v5/qualification_v1")
    rows = json.loads((root / "rows.json").read_text())
    print(
        json.dumps(
            {
                "demonstration": "structured graph -> independently executed target",
                "learned_prediction": None,
                "neural_execution": False,
                "qualification": json.loads((root / "qualification.json").read_text())["decision"],
            }
        )
    )
    for domain in ("mathematics", "coding", "physics"):
        row = next(r for r in rows if r["domain"] == domain)
        print(
            json.dumps(
                {
                    "domain": domain,
                    "identity": row["identity"],
                    "public_graph": row["graph"],
                    "executed_validity": row["target"]["valid"],
                    "executed_delta": row["target"]["delta"],
                    "executor": row["target"]["provenance"]["executor"],
                }
            )
        )


if __name__ == "__main__":
    main()
