# Removal and rollback

The only added repository subtree is `integrations/hybrid/`. Core files, the root dependency manifest, native scripts, research protocols, checkpoints and CI are unchanged.

To remove the extension in your own clean feature checkout:

```sh
git rm -r integrations/hybrid
git commit -m "Remove optional hybrid service"
```

No native entrypoint or root configuration needs restoration. Stop an explicitly running hybrid server with Ctrl-C. External model caches/virtual environments may be retained or deleted independently; they are never tracked. Do not remove research archives or checkpoints.

Recover the known-good Candidate B without rewriting history:

```sh
git fetch origin baseline/hybrid-candidate-b-20261015
git worktree add --detach ../mindscape-candidate-b 78c5771490491a2d1dc58e7baebdaf2ee5efd22a
cd ../mindscape-candidate-b/integrations/hybrid
./setup.sh
./start.sh
```

Baseline: `78c5771490491a2d1dc58e7baebdaf2ee5efd22a`. The original delivered offline B bundle remains untouched as the venue fallback. Use a separate local environment/model mount when network is unavailable.

Removal proof was performed in a disposable Git worktree: extension deleted, original pytest suite 146 passed / 22 skipped, 6 subtests passed; native `python -S scripts/run_episode.py 19 3` still verified 57. No hybrid dependency or initialization is required by native imports.
