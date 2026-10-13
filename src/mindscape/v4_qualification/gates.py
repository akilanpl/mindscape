"""Exact frozen thresholds; incomplete cohorts remain pending, never silently pass."""


def evaluate(rows, proposal):
    gates = {}
    limits = proposal["qualification_gates"]
    for domain in ("mathematics", "physics", "coding"):
        train = [
            r for r in rows if r["domain"] == domain and r["split"] == "train" and r["index"] < 4
        ]
        qual = [r for r in rows if r["domain"] == domain and r["split"] == "qualification"]
        for kind, group in (("target_training", train), ("qualification", qual)):
            key = domain + "." + kind
            values = {
                "tasks_observed": len(group),
                "positive": sum(sum(r["labels"]) for r in group),
                "negative": sum(len(r["labels"]) - sum(r["labels"]) for r in group),
                "mixed_tasks": sum(0 < sum(r["labels"]) < len(r["labels"]) for r in group),
                "solvable_tasks": sum(any(r["labels"]) for r in group),
            }
            passed = None
            if len(group) == 4:
                if kind == "target_training":
                    t = limits["every_domain_first_four_training_tasks"]
                    passed = (
                        values["positive"] >= t["minimum_positive_candidates"]
                        and values["negative"] >= t["minimum_negative_candidates"]
                        and values["mixed_tasks"] >= t["minimum_tasks_with_both_labels"]
                    )
                else:
                    t = limits["every_domain_four_qualification_tasks"]
                    passed = (
                        values["solvable_tasks"] >= t["minimum_solvable_tasks"]
                        and values["mixed_tasks"] >= t["minimum_tasks_with_both_labels"]
                    )
            gates[key] = {"pass": passed, **values}
        if domain == "coding":
            oracle = sum(any(r["labels"]) for r in qual)
            specialist = sum(
                r["labels"][max(range(len(r["scores"])), key=r["scores"].__getitem__)] for r in qual
            )
            first = sum(r["labels"][0] for r in qual)
            t = limits["coding_four_qualification_tasks"]
            passed = (
                (
                    oracle - specialist >= t["minimum_oracle_minus_specialist_correct_tasks"]
                    and oracle - first >= t["minimum_oracle_minus_first_candidate_correct_tasks"]
                )
                if len(qual) == 4
                else None
            )
            gates["coding.headroom"] = {
                "pass": passed,
                "oracle_correct": oracle,
                "specialist_correct": specialist,
                "first_candidate_correct": first,
                "oracle_minus_specialist": oracle - specialist,
                "oracle_minus_first": oracle - first,
                "tasks_observed": len(qual),
            }
    return gates


def decision(gates, completed):
    if any(g["pass"] is False for g in gates.values()):
        return "DATA_NO_GO"
    if completed == 36 and all(g["pass"] is True for g in gates.values()):
        return "DATA_READY"
    return "PENDING"
