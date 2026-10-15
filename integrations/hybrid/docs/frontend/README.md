# Mindscape frontend redesign

The actual Candidate C interface at http://127.0.0.1:8772/ now has a monochrome dark workspace, responsive sidebar/drawer, minimal Chat/Work header, centered “Ready when you are.” welcome, rounded expanding composer, useful suggested questions, real conversations and understated provenance. Vanilla HTML/CSS/JavaScript and the existing backend/API remain in place.

**The two reference images were not attached.** The supplied file contained text only; no reference image files were found, and their paths were requested. Design follows the written brief and tokens. Pixel-level matching to Reference A/B cannot be claimed. No OpenAI logos, browser chrome, account badges or proprietary assets were copied.

## Working functionality

- Real local Qwen hybrid, pretrained, tools-only and native-limit modes through the existing API. Native-limit mode is explicitly unsupported for general CBSE chat; it is not a fake neural answer.
- Versioned local chats, meaningful titles, active selection, search, actual local project folders and a Library of saved answers. Initial Research/CBSE Prep/Personal folders are real persisted empty local categories, not fabricated conversations.
- Confirmed conversation removal moves to Recently deleted and supports restore. Nothing is represented as cloud-synced. Legacy `mind1.chats` stays intact; unreadable new-format storage is backed up under `.recovery` before reset, or writes stop if that backup cannot be saved.
- Enter/Shift+Enter, growing textarea, empty/duplicate submission guards, loading/stop state, preserved text on recoverable failures, HTTP429 messaging and explicit retry without duplicate user messages. A newer draft is retained on failure/retry. The client gives the inherited server a brief post-response cooldown.
- Exact stdout/code copy, local syntax colors, tables, conservative local MathML, raw answer preservation and Activity with actual solver/interpretation/provenance metadata. No neural reasoning trace is invented.
- Work and Scheduled are unavailable. Assistants describes real modes, not autonomous agents. Voice and attachments are explicitly unavailable; no inert microphone/upload button is presented.

## Scientific preservation

`src`, native scripts/configuration, root dependencies, all `mindscape_delivery` backend modules, model lock and dependency lock retain identical Git objects to baseline `a026659e69a0a55ed441c3bf1e773ecce309f9e9`. Models, inference algorithms, verification flags, checkpoint formats and scientific results were not changed. Literal solver expressions and Python output bypass Markdown interpretation, so powers and printed markup cannot be silently reformatted. Original response strings/results remain unchanged for copying/export; rich MathML/table views are additional presentations, never new computations.

The reported prior C accuracy belongs to its earlier evaluation; these UI smoke checks are not a new CBSE accuracy campaign. No fitting or weight updates were performed.

## Evidence and source

- TEST_RESULTS.md: automated, real-browser, native and offline checks.
- VISUAL_REVIEW.md: all six measured viewports and genuine screenshots.
- ROLLBACK.md: frontend-only restoration while keeping Candidate C inference.
- The local delivery’s COMMITS.txt and PUSH_VERIFICATION.json record exact commits and remote/PR verification. GitHub’s commit history records the published series.

Frontend baseline: `a026659e69a0a55ed441c3bf1e773ecce309f9e9`, preserved in local branch `baseline/frontend-before-redesign` and `frontend-before-redesign.zip`; the baseline branch is pushed alongside the feature branch. Candidate B's existing baseline remains unchanged. The redesign is separate from all B/C solver commits and can be rolled back independently.

Final web source sizes: app.js 34,004 bytes; style.css 20,899 bytes. No runtime library/framework/font/CDN dependency was added. Existing local MathML helper and system fonts keep startup offline. Git changes remain under `integrations/hybrid/`. PR31 remains open and unmerged.

## Limitations

This is a local classroom/research assistant. Browser data has no sync, encryption vault or account system; don't enter credentials or private checkpoint data. Deleted chats remain in local storage until site data is cleared. Export important conversations first. Full TeX, arbitrary CPython, Work agents, scheduling, attachments and voice are not implemented. Unsupported math syntax remains raw rather than guessed. Cancellation cannot instantly stop an in-flight model call, and the inherited server can still return429 to rapid/concurrent clients; the UI handles it without pretending success. Tests use the actual in-app browser and emulated dimensions; physical mobile keyboards, all browser engines and formal WCAG certification were not tested.
