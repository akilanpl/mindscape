"""Opt-in arXiv metadata retrieval. Never fetch model-selected URLs or private context."""
from __future__ import annotations

import os
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass

ENDPOINT = 'https://export.arxiv.org/api/query'
MAX_RESPONSE_BYTES = 256_000
ATOM = {'a': 'http://www.w3.org/2005/Atom'}


@dataclass(frozen=True)
class Source:
    title: str
    authors: tuple[str, ...]
    published: str
    updated: str
    url: str
    identifier: str
    source_type: str = 'arxiv-preprint'


@dataclass(frozen=True)
class RetrievalResult:
    status: str
    sources: tuple[Source, ...] = ()
    detail: str = ''
    provider: str = 'arxiv-export-api'

    def as_dict(self) -> dict:
        return asdict(self)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, 'Redirects disabled', headers, fp)


def _text(entry, name: str, limit: int = 2000) -> str:
    return ' '.join((entry.findtext(f'a:{name}', default='', namespaces=ATOM)).split())[:limit]


def parse_arxiv_metadata(payload: bytes, max_results: int = 5) -> RetrievalResult:
    """Parse bounded actual API metadata; untrusted XML never supplies instructions."""
    if not 1 <= max_results <= 10:
        raise ValueError('max_results must be between 1 and 10')
    if len(payload) > MAX_RESPONSE_BYTES:
        return RetrievalResult('error', detail='Response exceeds metadata byte limit')
    if b'<!DOCTYPE' in payload.upper() or b'<!ENTITY' in payload.upper():
        return RetrievalResult('error', detail='XML declarations are unsupported')
    try:
        root = ET.fromstring(payload)
    except (ET.ParseError, ValueError):
        return RetrievalResult('error', detail='Invalid arXiv metadata XML')
    if root.tag != '{http://www.w3.org/2005/Atom}feed':
        return RetrievalResult('error', detail='Unexpected arXiv response format')
    sources = []
    for entry in root.findall('a:entry', ATOM):
        identifier = _text(entry, 'id', 256)
        parsed = urllib.parse.urlsplit(identifier)
        # API may return http canonical identifiers; expose https only after validation.
        if parsed.scheme not in ('http', 'https') or parsed.netloc != 'arxiv.org':
            continue
        if not re.fullmatch(r'/abs/(?:\d{4}\.\d{4,5}|[a-zA-Z-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?', parsed.path):
            continue
        if parsed.query or parsed.fragment:
            continue
        title = _text(entry, 'title')
        published, updated = _text(entry, 'published', 40), _text(entry, 'updated', 40)
        authors = tuple(_text(author, 'name', 200) for author in entry.findall('a:author', ATOM))
        if not title or not authors or not all(authors) or not published:
            continue
        sources.append(Source(title, authors, published, updated,
                              f'https://arxiv.org{parsed.path}', parsed.path.removeprefix('/abs/')))
        if len(sources) == max_results:
            break
    return RetrievalResult('ok' if sources else 'empty', tuple(sources),
                           'Retrieved metadata only; papers are not fetched or verified.')


def retrieval_capability() -> dict:
    enabled = os.environ.get('MIND1_RETRIEVAL') == '1'
    return {'enabled': enabled, 'provider': 'arxiv-export-api',
            'scope': 'Public arXiv title/author/date metadata; no full-text retrieval',
            'configuration': 'Set MIND1_RETRIEVAL=1 to permit public query transmission.'}


def retrieve_arxiv(query: str, max_results: int = 5, timeout: float = 5.0) -> RetrievalResult:
    """Send only an explicitly selected public search query, never session/verifier context."""
    if os.environ.get('MIND1_RETRIEVAL') != '1':
        return RetrievalResult('disabled', detail='Optional public retrieval is disabled')
    if not isinstance(query, str) or not 1 <= len(query.strip()) <= 300:
        return RetrievalResult('error', detail='Public query must contain 1–300 characters')
    if any(ord(char) < 32 for char in query):
        return RetrievalResult('error', detail='Query control characters are unsupported')
    if type(max_results) is not int or not 1 <= max_results <= 10:
        return RetrievalResult('error', detail='Result limit must be between 1 and 10')
    if type(timeout) not in (int, float) or not 0 < timeout <= 10:
        return RetrievalResult('error', detail='Timeout must be positive and at most ten seconds')
    params = urllib.parse.urlencode({'search_query': f'all:{query.strip()}',
                                    'start': 0, 'max_results': max_results})
    request = urllib.request.Request(f'{ENDPOINT}?{params}', headers={
        'Accept': 'application/atom+xml', 'User-Agent': 'Mindscape-Mind1/0.1 (metadata retrieval)'})
    try:
        # Disable environment proxies: endpoint and TLS trust are not selected by user input.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
        with opener.open(request, timeout=timeout) as response:
            payload = response.read(MAX_RESPONSE_BYTES + 1)
        return parse_arxiv_metadata(payload, max_results)
    except (OSError, urllib.error.URLError, TimeoutError):
        return RetrievalResult('unavailable', detail='arXiv metadata retrieval failed or timed out; no sources returned')
