from mindscape.mind1.retrieval import (
    MAX_RESPONSE_BYTES, parse_arxiv_metadata, retrieve_arxiv, retrieval_capability)

FEED = b'''<feed xmlns="http://www.w3.org/2005/Atom"><entry>
<id>http://arxiv.org/abs/2401.12345v2</id><title>A real fixture title</title>
<author><name>A. Author</name></author><published>2024-01-01T00:00:00Z</published>
<updated>2024-01-03T00:00:00Z</updated></entry></feed>'''


def test_metadata_contract():
    result = parse_arxiv_metadata(FEED)
    assert result.status == 'ok'
    assert result.sources[0].url == 'https://arxiv.org/abs/2401.12345v2'
    assert result.sources[0].authors == ('A. Author',)
    assert result.sources[0].published == '2024-01-01T00:00:00Z'
    assert result.as_dict()['provider'] == 'arxiv-export-api'


def test_opt_in_and_validation(monkeypatch):
    monkeypatch.delenv('MIND1_RETRIEVAL', raising=False)
    assert not retrieval_capability()['enabled']
    assert retrieve_arxiv('quantum field theory').status == 'disabled'
    monkeypatch.setenv('MIND1_RETRIEVAL', '1')
    for query in ('', 'a' * 301, 'private\ncontext'):
        assert retrieve_arxiv(query).status == 'error'
    assert retrieve_arxiv('physics', max_results=11).status == 'error'
    assert retrieve_arxiv('physics', timeout=11).status == 'error'


def test_malformed_xml_and_resource_limit():
    assert parse_arxiv_metadata(b'not XML').status == 'error'
    assert parse_arxiv_metadata(b'<html/>').status == 'error'
    assert parse_arxiv_metadata(b'<!DOCTYPE x><x/>').status == 'error'
    assert parse_arxiv_metadata(b'x' * (MAX_RESPONSE_BYTES + 1)).status == 'error'


def test_no_arbitrary_urls_or_fabricated_citations():
    assert parse_arxiv_metadata(FEED.replace(b'arxiv.org', b'127.0.0.1')).sources == ()
    assert parse_arxiv_metadata(FEED.replace(b'/abs/2401.12345v2', b'/admin')).sources == ()
    assert parse_arxiv_metadata(FEED.replace(b'<title>A real fixture title</title>', b'')).sources == ()


def test_network_failure_is_graceful(monkeypatch):
    import urllib.request
    monkeypatch.setenv('MIND1_RETRIEVAL', '1')
    class Offline:
        def open(self, *args, **kwargs):
            raise OSError('offline')
    monkeypatch.setattr(urllib.request, 'build_opener', lambda *args: Offline())
    result = retrieve_arxiv('quantum mechanics')
    assert result.status == 'unavailable'
    assert result.sources == ()


def test_transport_contract_has_fixed_host_and_bounded_read(monkeypatch):
    import urllib.request
    import urllib.parse
    monkeypatch.setenv('MIND1_RETRIEVAL', '1')
    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self, limit):
            assert limit == MAX_RESPONSE_BYTES + 1
            return FEED
    class Transport:
        def open(self, request, timeout):
            parsed = urllib.parse.urlsplit(request.full_url)
            assert parsed.scheme == 'https' and parsed.netloc == 'export.arxiv.org'
            assert parsed.path == '/api/query'
            assert urllib.parse.parse_qs(parsed.query)['search_query'] == ['all:https://127.0.0.1/private']
            assert timeout == 5.0
            return Response()
    monkeypatch.setattr(urllib.request, 'build_opener', lambda *args: Transport())
    assert retrieve_arxiv('https://127.0.0.1/private').status == 'ok'
