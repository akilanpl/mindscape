# Preservation recovery gates

Preservation start: 2026-10-24. Remote default branch main was independently
queried at `5ad2de4313aba6a7468638ed5709876e0c406668`.

Recovery has three separate gates:

1. Every important source state has an immutable commit and a durable remote ref.
2. Every essential scientific file has verified off-device bytes or an explicitly
   justified reproducible origin with pinned inputs, code, seeds and dependencies.
3. A fresh remote-only clone, clean dependency installation and restoration test
   pass with receipts. Local copies and local-only manifests are not backups.

This document is a procedure, not evidence that all gates passed. Consult the
preservation report for actual branch SHAs, archive validation receipts, unresolved
working states and deletion eligibility. No path is authorized for removal by this
document. Public source preservation does not preserve ignored results or work.

Never publish machine credentials, environment secrets, private keys, personal
logs or the private local inventory database. No paid storage may be provisioned
without explicit authorization. No force pushes or history rewrites are needed.
