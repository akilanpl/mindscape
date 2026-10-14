# Mind 1 local security review

Review date: 2026-10-14. Scope: local Mind 1 provider, orchestration, tools and browser UI. This is a focused engineering review, not a penetration-test certification.

## Verified controls

- Provider URLs reject embedded credentials, query secrets and fragments. Remote HTTP is forbidden; explicit loopback HTTP is permitted. Redirects are disabled so authorization cannot follow a redirect. Public capability responses omit API keys.
- Provider HTTP failures expose fixed status descriptions rather than remote bodies. Requests enforce context limits; responses enforce a 2 MB limit.
- User history accepts only user/assistant text; system-role injection and oversized history are rejected.
- Symbolic inputs use an allowlisted AST builder rather than Python eval or textual sympify. Attribute access, arbitrary calls, comprehensions, lambdas and excessive powers are rejected.
- Code inspection parses source as data. Source execution is WASI-only and unavailable execution fails closed. Workers receive a minimal environment without model API credentials. Tool execution has process time/output limits.
- Browser answer rendering builds DOM nodes and textContent, never innerHTML. Links require HTTP(S) and use noopener. Code blocks are plain text. No remote fonts, scripts or runtime assets are required.
- Native MathML renders a bounded LaTeX subset: fractions, square roots, subscripts/superscripts, common Greek symbols/operators. Unsupported commands remain text. No HTML, macros, eval, external URLs or TeX execution are supported. This is not a complete KaTeX implementation.

## Findings repaired during integration

1. Provider model discovery needs shape validation for non-object JSON and a null/non-list `data` field. Usage metadata needs type validation before price arithmetic. Otherwise malformed remote output can become an uncontrolled internal exception.
2. Provider endpoint validation should also require a hostname.
3. Mode, domain and history role must be strings before set/dict membership tests; malformed arrays/objects should yield controlled request errors.
4. Validate structured tool call IDs and argument types before running the selected tool.

These findings were repaired in the integrated sources. Server review confirmed exact static-file allowlisting, strict loopback Host and Origin checks, cross-site fetch rejection, JSON-only POSTs, bounded request bodies, rejection of non-finite JSON, no-store caching, nosniff and a self-only content security policy. Error logging uses a fixed event rather than exception content. Cancellation is explicit: the frontend sends a request ID and POSTs /api/cancel before discarding the browser request. An in-flight provider request can continue until its configured socket timeout; the worker slot remains occupied until that call finishes.

## Validation

`tests_mind1/test_security.py`: 41 tests passed against the integrated source tree, covering provider secret/error/size boundaries, malformed completion, AST injection, exponent resource abuse, non-executing inspection, unavailable WASI, history role injection and history size; malformed selectors/discovery payloads; HTTP host/origin/fetch-site rejection; path traversal; CSP and response headers; malformed JSON, content type and size; explicit cancellation. HTTP tests exercise the real handler in memory. A live socket test was blocked by sandbox permissions, so these tests do not claim live-network verification.

`node --check mind1_web/app.js` and `node --check mind1_web/math.js` passed. A headless browser smoke test could not run because the bundled Playwright browser executable is absent; no browser was downloaded. Native MathML rendering still needs visual browser verification.

## Residual limits

This application is intended for a loopback, single-user workspace. Browser-local history is readable by anyone with the same browser profile. A configured provider receives submitted conversation text. Static code findings and successful symbolic computations do not establish general code safety or mathematical correctness. No public deployment or authentication security claim is made.
