# Datasets

Generated JSONL splits are authoritative evaluation records, including full targets.
Do not train on these directly: export regime-specific views, restricted to train.
Experiential views contain no full target trace or answer. `development_v1` is a
seed-42 infrastructure dataset, not a final untouched research benchmark. Its manifest
records hashes, configuration/version and nested subset IDs. Shared factors are allowed;
canonical operand pairs, examples and full traces cannot overlap across splits.
