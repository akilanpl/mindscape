# Mindscape research website

Six static routes, built with HTML, CSS and browser JavaScript. No production dependencies, analytics or external fonts. Published through Sites; original research remains in the Mindscape repository.

## Build and validate

Node 25.9.0 and npm 11.12.1 were used. Run `npm ci`, `npm run build`, `npm test`, `npm run lint`, and `npm run typecheck`. Serve `dist` with any static HTTP server. The site uses root-relative paths and requires directory index support. `SITE_ORIGIN` optionally sets the canonical production origin during build.

`assets/evidence.json` and `assets/traces.json` are bounded exports from research commit `c530faa20aee2339679a5f2cfe9bea28509d4fb1`. In the Mindscape checkout, regenerate them with `python3 website/scripts/prepare_evidence.py`. The website renders observed data, retains failures and explicitly marks historical GPT/Gemini comparisons as unmeasured. Selected trace events are labelled as selections; they are not causal explanations.

The committed MP4 is original procedural artwork, not a computational trace. Regeneration uses `scripts/hero.py` with Pillow and imageio-ffmpeg 0.6.0; these are optional asset-authoring tools and not needed to build or serve the site. Video is desktop opt-in/automatic where permitted; reduced-motion, mobile and data-saving visitors receive the poster fallback. There is a pause control.

Metadata, sitemap, robots and a meta CSP are generated. `_headers` is a portable hosting hint; actual response headers depend on the host. No claims are made that these headers are enforced until checked on the deployed host.

Research v1 is preserved. Research v2 has unresolved generalization and measurement gaps; this website does not imply a v2 completion release or broad model superiority.
