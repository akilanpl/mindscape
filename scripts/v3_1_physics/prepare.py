import json
from pathlib import Path

from mindscape.v3.physics_world.storage import atomic_json, file_sha
from mindscape.v3_1.physics.data import create

if __name__ == "__main__":
    p = json.loads(Path("configs/v3_1_physics/pilot.json").read_text())
    root = Path("results/v3_1_physics/dataset-pilot-v1")
    m = create(root, p)
    old = json.loads(Path("results/v3_physics/dataset-full-v1/manifest.json").read_text())
    assert not {r["identity"] for r in old["records"]} & {r["identity"] for r in m["records"]}
    atomic_json(
        "experiments/v3_1_physics/freeze-v2.json",
        {
            "dataset": m["hash"],
            "protocol": file_sha("configs/v3_1_physics/pilot.json"),
            "files": {
                str(p): file_sha(p)
                for folder in ["src/mindscape/v3_1", "scripts/v3_1_physics"]
                for p in sorted(Path(folder).rglob("*.py"))
            },
            "sealed_testing": "NOT RUN",
            "v3_final_holdout_used_for_tuning": False,
        },
    )
    print(json.dumps({"dataset": m["hash"], "scenes": len(m["records"])}))
