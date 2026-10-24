"""Deterministic pure-data construction and verification for bounded R6 P1.

The final files and their seeds belong to the controller/evaluator. Trainer
packaging must take only train.json, dev.json and normalization.json.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path

from research.r6.datasets import coding, mathematics, physics

SPLIT_SEEDS = {"train": 61017, "dev": 62029, "final_iid": 63043, "final_ood": 64071}
SPLIT_COUNTS = {"train": 512, "dev": 64, "final_iid": 96, "final_ood": 96}
GENERATORS = {"mathematics": mathematics, "physics": physics, "coding": coding}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def verify_row(row):
    required = {"record_id", "domain", "topic", "difficulty", "input", "structured_input", "target",
                "verification_method", "source", "split", "generator_version", "template_id", "canonical_key"}
    if not required <= row.keys() or row["split"] not in SPLIT_SEEDS:
        raise ValueError("Missing schema fields or unknown split")
    if row["source"] != "generated" or row["generator_version"] != "r6-v1":
        raise ValueError("Unexpected generator provenance")
    if not row["input"].isascii() or len(row["input"].encode("ascii")) > 192:
        raise ValueError("Input must be ASCII and at most 192 bytes")
    domain = row["domain"]
    if domain == "mathematics":
        mathematics.verify_record(row)
        if row.get("features", []) != []:
            raise ValueError("Arithmetic features must be empty")
    elif domain == "physics":
        system = row["structured_input"]
        reference = physics.verify(system)
        target = row["target"].get("state")
        if not isinstance(target, list) or len(target) != 4 or any(
            not math.isfinite(x) or abs(x-y) > 1e-9 for x,y in zip(target, reference)
        ):
            raise ValueError("Physics independent verification failed")
        if row.get("features") != physics.features(system):
            raise ValueError("Physics features differ from observed raw system")
        if row["canonical_key"] != json.dumps(system, sort_keys=True, separators=(",", ":")):
            raise ValueError("Physics canonical key differs")
    elif domain == "coding":
        if row["target"] != {"answer": str(coding.interpret(row["input"]))}:
            raise ValueError("Program target differs from bounded interpreter")
        observed = row["structured_input"]
        if observed["ast"] != ast.dump(ast.parse(row["input"]), include_attributes=False):
            raise ValueError("Structured AST differs from program text")
        tree = ast.parse(row["input"])
        if observed["output_binding"] != tree.body[-1].targets[0].id:
            raise ValueError("Output binding differs")
        template = row["template_id"].removeprefix("python-")
        if row["target"]["answer"] != str(coding._oracle(template, **observed["parameters"])):
            raise ValueError("Program template oracle disagrees")
        if row["canonical_key"] != hashlib.sha256(coding.canonicalize(row["input"]).encode()).hexdigest():
            raise ValueError("Program canonical key differs")
        if row.get("features", []) != []:
            raise ValueError("Coding features must be empty")
    else:
        raise ValueError("Unsupported domain")
    return True


def validate_splits(splits):
    seen_keys, seen_inputs, seen_ids = {}, {}, {}
    for split, rows in splits.items():
        for row in rows:
            verify_row(row)
            if row["split"] != split:
                raise ValueError("Record assigned to wrong split")
            for seen, key, label in ((seen_keys, (row["domain"], row["canonical_key"]), "canonical"),
                                     (seen_inputs, (row["domain"], row["input"]), "exact input"),
                                     (seen_ids, row["record_id"], "record ID")):
                if key in seen:
                    raise ValueError(f"Duplicate {label} in {seen[key]} and {split}")
                seen[key] = split
    return {"records_verified": sum(map(len, splits.values())), "canonical_duplicates": 0,
            "exact_input_duplicates": 0, "record_id_duplicates": 0,
            "scope": "global across all four splits, domain-aware canonical/input identity"}


def normalization(train, train_hash):
    states = [row["target"]["state"] for row in train if row["domain"] == "physics"]
    if not states:
        raise ValueError("Training physics targets required")
    axes = list(zip(*states))
    means = [statistics.mean(axis) for axis in axes]
    scales = [max(statistics.pstdev(axis), 1e-6) for axis in axes]
    majority = {}
    majority_counts = {}
    for domain in ("mathematics", "coding"):
        counts = Counter(row["target"]["answer"] for row in train if row["domain"] == domain)
        if not counts:
            raise ValueError("Training categorical targets required")
        answer = sorted(counts, key=lambda key: (-counts[key], key))[0]
        majority[domain], majority_counts[domain] = answer, counts[answer]
    return {"computed_from": "train-only", "train_sha256": train_hash,
            "physics_state_axes": ["vx", "vy", "x", "y"], "physics_state_mean": means,
            "physics_state_std": scales, "physics_mean_state": means,
            "scale_floor": 1e-6, "std_convention": "population",
            "training_majority_answers": majority, "training_majority_counts": majority_counts,
            "majority_tie_break": "lexicographic canonical answer string"}


def _distribution(rows, field):
    return dict(sorted(Counter(row[field] for row in rows).items()))


def _lengths(rows):
    lengths = [len(row["input"].encode("ascii")) for row in rows]
    words = [len(row["input"].split()) for row in rows]
    return {"ascii_byte_tokens": {"count": sum(lengths), "min": min(lengths), "max": max(lengths),
                                  "mean": statistics.mean(lengths)},
            "whitespace_tokens": {"count": sum(words), "min": min(words), "max": max(words),
                                  "mean": statistics.mean(words)},
            "tokenization_note": "ASCII byte count is exact for the planned byte tokenizer; whitespace is descriptive only"}


def _observed_ranges(rows):
    def limits(values):
        values = list(values)
        return {"min": min(values), "max": max(values)}
    by_domain = {domain: [row for row in rows if row["domain"] == domain] for domain in GENERATORS}
    math_rows, physical, programs = (by_domain[domain] for domain in GENERATORS)
    return {"mathematics": {"absolute_operand": limits(abs(value) for row in math_rows for value in row["structured_input"]["operands"]),
                            "operand_arities": sorted({len(row["structured_input"]["operands"]) for row in math_rows})},
            "physics": {"mass": limits(row["structured_input"]["mass"] for row in physical),
                         "force_component": limits(value for row in physical for stage in row["structured_input"]["stages"] for value in stage["force"]),
                         "time_interval": limits(stage["dt"] for row in physical for stage in row["structured_input"]["stages"]),
                         "stage_counts": sorted({len(row["structured_input"]["stages"]) for row in physical})},
            "coding": {key: limits(row["structured_input"]["parameters"][key] for row in programs) for key in ("a", "b", "c")}}


def build(output, counts=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Dataset output must be a new directory")
    counts = dict(SPLIT_COUNTS if counts is None else counts)
    if set(counts) != set(SPLIT_SEEDS) or any(type(n) is not int or n <= 0 for n in counts.values()):
        raise ValueError("Positive counts required for all four splits")
    splits, generation = {}, {}
    for split, seed in SPLIT_SEEDS.items():
        rows = []
        generation[split] = {}
        for domain, module in GENERATORS.items():
            if hasattr(module, "generate_with_metadata"):
                batch, metadata = module.generate_with_metadata(split, seed, counts[split])
            else:
                batch = module.generate(split, seed, counts[split])
                if hasattr(module, "generation_statistics"):
                    metadata = module.generation_statistics(split, seed, counts[split])
                else:
                    metadata = {"accepted": len(batch), "requested": counts[split], "attempts": None,
                                "rejected": None, "invalid": None,
                                "count_limitation": "Generator exposes accepted records but not draw/rejection counters; unknown is not zero"}
            generation[split][domain] = metadata
            rows.extend(batch)
        splits[split] = rows
    validation = validate_splits(splits)
    # All independent data checks pass before the new directory is created.
    output.mkdir(parents=True)
    paths = {split: ("protected/" if split.startswith("final") else "") + split + ".json" for split in splits}
    for split, path in paths.items():
        _write(output / path, splits[split])
    _write(output / "normalization.json", normalization(splits["train"], sha256(output / "train.json")))
    templates = {domain: {split: sorted({row["template_id"] for row in rows if row["domain"] == domain})
                          for split, rows in splits.items()} for domain in GENERATORS}
    overlap = {}
    for domain, groups in templates.items():
        train = set(groups["train"])
        overlap[domain] = {"templates_by_split": groups,
                           "train_dev_shared": sorted(train & set(groups["dev"])),
                           "train_iid_shared": sorted(train & set(groups["final_iid"])),
                           "train_ood_shared": sorted(train & set(groups["final_ood"])),
                           "iid_template_overlap_expected": True,
                           "ood_note": {"mathematics": "Range OOD shares binary syntax but uses unseen magnitude ranges; composition OOD has three operands.",
                                        "physics": "Three stages with held-out mass/force/time ranges versus two training stages.",
                                        "coding": "Nested loops/functions and branching inside loops absent from training templates."}[domain]}
        if overlap[domain]["train_ood_shared"]:
            raise ValueError("Unexpected OOD template identifier overlap")
    files = {path: {"sha256": sha256(output / path), "bytes": (output / path).stat().st_size,
                    "exposure": "protected-final-controller-and-evaluator-only" if path.startswith("protected/") else "trainer-allowed"}
             for path in [*paths.values(), "normalization.json"]}
    source_files = [Path(__file__), *(Path(module.__file__) for module in GENERATORS.values())]
    manifest = {"dataset_version": "r6-v1", "status": "GENERATED_AND_INDEPENDENTLY_VERIFIED",
                "split_seeds": SPLIT_SEEDS, "requested_count_per_domain": counts, "files": files,
                "splits": {split: {"records": len(rows), "domains": _distribution(rows, "domain"),
                                    "topics": _distribution(rows, "topic"), "difficulties": _distribution(rows, "difficulty"),
                                    "templates": _distribution(rows, "template_id"), "lengths": _lengths(rows),
                                    "observed_ranges": _observed_ranges(rows)}
                           for split, rows in splits.items()}, "generation_counters": generation,
                "validation": validation, "structural_overlap": overlap,
                "generator_sources_sha256": {"datasets/" + path.name: sha256(path) for path in source_files},
                "trainer_allowlist": ["train.json", "dev.json", "normalization.json"],
                "isolation_scope": "Controller protects final attachments; same-owner account can access future runs. This manifest includes final hashes/seeds and is excluded from trainer payload.",
                "coverage": "Limited original foundation arithmetic, SI multistage motion, and bounded Python tracing; not Class 12 competency."}
    _write(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = build(args.output)
    print(json.dumps({"status": result["status"], "records": result["validation"]["records_verified"],
                      "output": str(Path(args.output).resolve())}))


if __name__ == "__main__":
    main()
