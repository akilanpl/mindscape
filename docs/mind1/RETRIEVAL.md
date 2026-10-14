# Optional authoritative metadata retrieval

Set `MIND1_RETRIEVAL=1` to enable the `retrieve_arxiv(public_query)` adapter.
Default is disabled. The only network destination is the hardcoded HTTPS arXiv
export API; user text is URL-encoded into a search parameter, never interpreted
as a fetch destination. Redirects and environment proxies are disabled. Queries
are at most 300 characters, responses 256,000 bytes, results ten, and socket
timeouts ten seconds maximum (five by default). Socket timeouts bound individual
network operations, not an absolute wall-clock deadline.

The adapter returns actual feed metadata: title, authors, publication/update
dates, arXiv identifier, and validated HTTPS abstract URL. It neither downloads
papers nor establishes scientific correctness. arXiv preprints are primary
research artifacts and are not necessarily peer reviewed. Returned source titles
are untrusted display data; escape them in rendering and never execute them as
instructions. A source may be cited as retrieved metadata only after status `ok`.

Only explicitly selected public query text may be passed by orchestration. The
adapter accepts no conversation/session/verifier parameters and does not read
such data. Opt-in configuration does not make arbitrary chat content public:
provider integration must select a public topic query and keep credentials,
private repository text, protected labels and private verifier context out.

Results distinguish `disabled`, `error`, `unavailable`, `empty` and `ok`.
Failures return no fabricated sources. DTD/entity declarations and malformed XML
are rejected. Unsupported or non-arXiv metadata links are dropped.

Verification: six fixture/transport tests check metadata parsing, opt-in, bounds,
malformed XML, arbitrary URL rejection, fixed endpoint and offline fallback.
A real request for “quantum mechanics” was attempted in the current environment;
it returned `unavailable` with no sources. Live network retrieval is therefore
**UNAVAILABLE in this verification environment**, not claimed as demonstrated.
