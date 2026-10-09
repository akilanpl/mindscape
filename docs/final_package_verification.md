# Final coding package verification

Release acceptance requires exactly 400 locked rows, 1920 learning rows, 400 independent trajectory/terminal replays and 16 dedicated latency cases, passing complete tests, scoped lint, compile and wheel build. The publication gate checks exact task/key sets, byte-preserved CPU evidence, frozen source hashes and agreement between raw scores and reports.

The immutable snapshot is `results/final/coding_research_v2`. Its manifest binds each file to SHA-256 and records the source commit. Clean-destination restoration, inventory/hash verification and the restored full test suite are required before the completion tag. The post-snapshot verification receipt is distributed alongside the snapshot at `results/final/coding_research_v2_verification.json` and committed in `experiments/coding_completion_v2/research_package_verified.json`; it necessarily postdates the immutable snapshot. Foundation weights are exact pinned external downloads.

The historical partial snapshot `coding_emergency_partial_v1` retains its original 6178 files and verification receipt. It is historical evidence, not the current completion scope.
