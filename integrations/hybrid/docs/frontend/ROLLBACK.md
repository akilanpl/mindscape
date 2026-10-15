# Frontend-only rollback

Baseline Candidate C before any UI edits: `a026659e69a0a55ed441c3bf1e773ecce309f9e9`. Remote branch: `baseline/frontend-before-redesign`. Original B baseline remains `78c5771490491a2d1dc58e7baebdaf2ee5efd22a`.

Export important new-UI conversations first. Old and new browser storage keys differ; rollback does not delete the new store, but the old interface does not display it. Legacy `mind1.chats` was preserved. This rollback does not alter backend inference, model files, research or checkpoint formats, and does not rewrite history.

From a clean checkout of the feature branch:

```sh
git fetch origin baseline/frontend-before-redesign
git restore --source=a026659e69a0a55ed441c3bf1e773ecce309f9e9 -- integrations/hybrid/web
git rm -r -- integrations/hybrid/tests/frontend   integrations/hybrid/tests/serve_ui_faults.py   integrations/hybrid/tests/offline_startup.py
git commit -m "Rollback frontend only; preserve Candidate C inference"
```

The removed tests/fixtures were added only by this UI task and depend on its renderer. Existing backend/acceptance tests are retained. The redesign docs may remain as historical evidence. Reload the browser; the same running backend can continue serving the restored assets. Review any unrelated local modifications before restoring files. Alternatively create a separate worktree at the baseline SHA to inspect the full preserved C release.

The initial visual commit and subsequent interaction/responsive/test fixes are separate from all solver commits. COMMITS.txt records the complete UI series; revert that explicit series if a whole-change Git revert is preferred. Never revert the B/C scientific commits to undo this frontend.
