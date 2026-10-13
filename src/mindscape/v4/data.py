"""Fresh development identities. No historical benchmark or sealed holdout reads."""

from mindscape.v3.schema import digest

DOMAINS = ("mathematics", "coding", "physics")


def tasks():
    rows = []
    for split, count, offset in [("train", 8, 0), ("iid", 4, 8), ("ood", 8, 12)]:
        for i in range(count):
            n = offset + i
            # Disjoint local states; OOD changes incoming carry/borrow context.
            rows.append(
                {
                    "domain": "mathematics",
                    "split": split,
                    "public": {
                        "operation": ("add", "sub")[n % 2],
                        "left": (n // 2) % 10,
                        "right": (n * 3 + 2) % 10,
                        "incoming": int(split == "ood"),
                    },
                }
            )
            rows.append(
                {
                    "domain": "physics",
                    "split": split,
                    "public": {
                        "family": "motion",
                        "seed": 2026101304,
                        "index": 1000 + n,
                        "horizon": 16 if split == "ood" else 8,
                        "target": [(-1.0, 0.0, 1.0)[n % 3], 1.0 + (n % 4) * 0.25],
                    },
                }
            )
            threshold = 21 + n
            entry = f"compute_v4_{n}"
            ood = split == "ood"
            expression = (
                (
                    "sum(1 for item in values if item > "
                    if ood
                    else "sum(item for item in values if item > "
                )
                + str(threshold)
                + ")"
            )

            def suite(values, entry=entry, threshold=threshold, ood=ood):
                return {
                    "entry": "solution." + entry,
                    "cases": [
                        {
                            "args": [v],
                            "expected": (
                                sum(x >= threshold for x in v)
                                if ood
                                else sum(x for x in v if x >= threshold)
                            ),
                            "expected_type": "int",
                            "unchanged": True,
                        }
                        for v in values
                    ],
                }

            public = {
                "repository": {"solution.py": f"def {entry}(values):\n    return {expression}\n"},
                "contract": {
                    "entry": "solution." + entry,
                    "description": f"Return the {'count' if ood else 'sum'} of values at least {threshold}, inclusive. Do not mutate input.",
                    "inputs": ["list of integers"],
                    "outputs": "int",
                    "invariants": ["Inputs remain unchanged."],
                },
                "visible_tests": suite([[threshold], [threshold - 1, threshold, threshold + 2]]),
            }
            rows.append(
                {
                    "domain": "coding",
                    "split": split,
                    "public": public,
                    "private": suite(
                        [
                            [],
                            [-3, threshold, threshold],
                            [threshold - 1],
                            [threshold + 1, threshold + 3],
                            [threshold, -9, threshold + 4],
                        ]
                    ),
                }
            )
    for row in rows:
        row["identity"] = digest({"domain": row["domain"], "public": row["public"]})
    if len({r["identity"] for r in rows}) != len(rows):
        raise ValueError("Overlapping identities")
    return rows
