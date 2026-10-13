# V4 status

The completed native pilot is **NO_GO**. Start with [scientific-report.md](scientific-report.md).

- Original pretraining protocol: `configs/v4/protocol.json` (unchanged).
- Complete original run and frozen sources: `experiments/v4/native_v1`, with per-file SHA-256 and verified restoration.
- Independent replay, 27-checkpoint inspection, 54 metric recalculations, label/source tables and corruption rejection: `experiments/v4/review_v1`.
- Fresh-data qualification proposal, not executed: `configs/v4/fresh-development-proposal.json`.
- Original pre-native handoff: `handoff.md` (historical; its pending status is superseded by the completed report).

Read-only post-run diagnosis (existing local pinned assets required):

```zsh
PYTHONPATH=src:work/physics-deps/site-packages /Users/akilan/Documents/Codex/2026-10-08/files-pasted-by-the-user-mindscape/outputs/mindscape/work/final-venv/bin/python scripts/v4/review_completed.py --run results/v4/native-1791854687819867000 --output work/v4-new-review
```

Use a new output directory. This checks tensors, saved metrics and metadata; it does not execute models. Full candidate execution replay uses `python -m mindscape.v4.audit <copy-of-run>` with the same Python/PYTHONPATH. Audit writes replay receipts, so always give it a copy to preserve original bytes. Native checkpoint forward replay and bitwise continuation are explicitly not claimed. Frozen sources are archived to support reconstruction; new files added after the run must not be mistaken for its original source identity.

## Final qualification and sprint stop

The fresh-data screen completed **DATA_NO_GO**, stopping after 4 requests (3 positive/13 negative mathematics candidates; minimum positives 4). The remaining 32 requests were not executed. See [qualification-result.md](qualification-result.md) and [website-update-handoff.md](website-update-handoff.md). No further research or website changes are authorized.
