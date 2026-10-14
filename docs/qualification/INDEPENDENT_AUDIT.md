# Independent read-only scientific audit

A separate reviewer inspected the actual working-tree trainer, graph/control code
and qualification package builder without edits, optimizer construction, signatures
or training. Their decision was **NO-GO**.

They independently confirmed all three seeds have zero encoded train/dev/test
intersection, heldout transition inputs avoid training overlap, all72 unique run
identities sum to144000updates, and matched controls use their real architectures.
The reviewer retracted an initial stale-baseline-dispatch concern after rereading
the current source; no such defect is claimed for this revision.

Their final findings confirm the amended observation-hash concept corruption uses
no target labels, while retaining unresolved concept redundancy, non-episodic memory,
unsupervised action head, unequal compute and unenforced evaluator process isolation.
These remain NO-GO reasons in SCIENTIFIC_REPORT.md.

Reviewed snapshot hashes (read-only audit before subsequent additive paired-report
and manifest-detail additions; these are NOT claimed as final-package hashes):

- qualification.py: f205af4cf48d4a1cadb2276591f60e8fbbed0eac3c397bbe683b784b58c23784
- learning.py: db3e0c7fad80ce1b7159e71851f74fea343d978e146867f1f46c378425cfc4e3
- benchmarks.py: 4e4fa17d4830211fbc5c224fbcdd6f3344f7e1c3be2cfa9daacc8aa6e9c200a9

Final-package hashes are generated after all changes and bound separately. An
engineering agent's review is not the required externally signed scientific
qualification or human authorization. Neither is issued by this document.
