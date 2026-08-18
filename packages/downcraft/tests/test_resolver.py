"""Tests for downcraft.resolver — link extraction from ad-heavy pages."""

import pytest

from downcraft.resolver import (
    _extract_js_redirects,
    _extract_meta_urls,
    _get_extension,
    _is_same_domain,
    _resolve_relative,
    _score_link,
    _LinkExtractor,
    resolve_page,
    ResolvedLink,
)
from conftest import _range_url


# ---------------------------------------------------------------------------
# Utility function tests
# ---------------------------------------------------------------------------

class TestGetExtension:
    def test_simple_zip(self):
        assert _get_extension("https://example.com/file.zip") == ".zip"

    def test_tar_gz(self):
        assert _get_extension("https://example.com/file.tar.gz") == ".tar.gz"

    def test_no_extension(self):
        assert _get_extension("https://example.com/file") == ""

    def test_query_string(self):
        assert _get_extension("https://example.com/file.zip?token=abc") == ".zip"

    def test_fragment(self):
        assert _get_extension("https://example.com/file.zip#section") == ".zip"

    def test_case_insensitive(self):
        assert _get_extension("https://example.com/file.ZIP") == ".zip"

    def test_tar_bz2(self):
        assert _get_extension("https://example.com/archive.tar.bz2") == ".tar.bz2"


class TestIsSameDomain:
    def test_same_domain(self):
        assert _is_same_domain("https://example.com/a", "https://example.com/b")

    def test_subdomain(self):
        assert _is_same_domain("https://cdn.example.com/a", "https://example.com/b")

    def test_different_domain(self):
        assert not _is_same_domain("https://other.com/a", "https://example.com/b")

    def test_invalid_url(self):
        assert not _is_same_domain("not-a-url", "https://example.com/b")


class TestResolveRelative:
    def test_absolute_url_unchanged(self):
        assert _resolve_relative("https://example.com/file.zip", "https://other.com") == "https://example.com/file.zip"

    def test_relative_path(self):
        result = _resolve_relative("files/archive.zip", "https://example.com/page.html")
        assert result == "https://example.com/files/archive.zip"

    def test_root_relative(self):
        result = _resolve_relative("/download/file.zip", "https://example.com/deep/page")
        assert result == "https://example.com/download/file.zip"


class TestExtractJsRedirects:
    def test_window_location(self):
        html = '<script>window.location = "https://example.com/download.zip";</script>'
        urls = _extract_js_redirects(html)
        assert "https://example.com/download.zip" in urls

    def test_window_location_href(self):
        html = "window.location.href = 'https://example.com/file.exe'"
        urls = _extract_js_redirects(html)
        assert "https://example.com/file.exe" in urls

    def test_document_location(self):
        html = 'document.location = "https://example.com/data.tar.gz"'
        urls = _extract_js_redirects(html)
        assert "https://example.com/data.tar.gz" in urls

    def test_meta_refresh(self):
        html = '<meta http-equiv="refresh" content="0;url=https://example.com/next">'
        urls = _extract_js_redirects(html)
        assert "https://example.com/next" in urls

    def test_window_open(self):
        html = 'window.open("https://example.com/popup.zip")'
        urls = _extract_js_redirects(html)
        assert "https://example.com/popup.zip" in urls

    def test_no_redirects(self):
        html = '<html><body>No JS here</body></html>'
        urls = _extract_js_redirects(html)
        assert urls == []

    def test_skips_javascript_void(self):
        html = 'href="javascript:void(0)"'
        urls = _extract_js_redirects(html)
        assert urls == []

    def test_skips_fragments(self):
        html = 'href="#section"'
        urls = _extract_js_redirects(html)
        assert urls == []


class TestExtractMetaUrls:
    def test_og_url(self):
        html = '<meta property="og:url" content="https://example.com/page">'
        urls = _extract_meta_urls(html)
        assert "https://example.com/page" in urls

    def test_canonical(self):
        html = '<link rel="canonical" href="https://example.com/canonical">'
        urls = _extract_meta_urls(html)
        assert "https://example.com/canonical" in urls

    def test_no_meta(self):
        html = '<html><body>Nothing</body></html>'
        urls = _extract_meta_urls(html)
        assert urls == []


class TestScoreLink:
    def test_download_extension_high(self):
        score = _score_link(
            "https://example.com/file.zip", "Download", {}, "https://example.com"
        )
        assert score > 0.3

    def test_ad_signal_penalized(self):
        score = _score_link(
            "https://tracking.example.com/ad?ref=abc", "Click here", {}, "https://example.com"
        )
        assert score < 0.2

    def test_download_class_bonus(self):
        score = _score_link(
            "https://example.com/file.zip", "Get it", {"class": "btn-download"}, "https://example.com"
        )
        assert score > 0.4

    def test_sponsor_class_penalized(self):
        score = _score_link(
            "https://example.com/file.zip", "Sponsor", {"class": "sponsor-link"}, "https://example.com"
        )
        assert score < 0.3

    def test_data_href_bonus(self):
        score = _score_link(
            "#", "Download", {"data-href": "https://example.com/file.zip"}, "https://example.com"
        )
        assert score > 0.1

    def test_empty_href_low(self):
        score = _score_link("", "", {}, "https://example.com")
        assert score <= 0.0


class TestLinkExtractor:
    def test_basic_link(self):
        parser = _LinkExtractor()
        parser.feed('<a href="https://example.com/file.zip">Download</a>')
        assert len(parser.links) == 1
        href, text, attrs = parser.links[0]
        assert href == "https://example.com/file.zip"
        assert text == "Download"

    def test_multiple_links(self):
        html = """
        <a href="/a">Link A</a>
        <a href="/b">Link B</a>
        <a href="/c">Link C</a>
        """
        parser = _LinkExtractor()
        parser.feed(html)
        assert len(parser.links) == 3

    def test_data_attribute(self):
        html = '<div data-url="https://example.com/hidden.zip">Content</div>'
        parser = _LinkExtractor()
        parser.feed(html)
        urls = [href for href, _, _ in parser.links]
        assert "https://example.com/hidden.zip" in urls


# ---------------------------------------------------------------------------
# Integration tests with local server
# ---------------------------------------------------------------------------

class TestResolvePage:
    """Tests for resolve_page using a local HTTP server."""

    def _make_page(self, html: str) -> str:
        """Register an HTML page on the range server."""
        from conftest import RangeHandler
        RangeHandler.payloads["/page"] = html.encode()
        return "/page"

    def test_finds_download_link(self, range_server):
        from conftest import RangeHandler
        RangeHandler.payloads["/page"] = b'''
        <html>
        <a href="/files/app.zip" class="btn-download">Download App</a>
        <a href="/ads/sponsor">Sponsor</a>
        </html>
        '''
        url = _range_url(range_server, "/page")
        links = resolve_page(url)
        assert len(links) >= 1
        best = links[0]
        assert "app.zip" in best.url
        assert best.confidence > 0.3

    def test_ranks_extension_above_non_extension(self, range_server):
        from conftest import RangeHandler
        RangeHandler.payloads["/page"] = b'''
        <html>
        <a href="/about">About</a>
        <a href="/download/file.tar.gz">Download</a>
        </html>
        '''
        url = _range_url(range_server, "/page")
        links = resolve_page(url)
        best = links[0]
        assert best.extension in (".tar.gz", ".zip")

    def test_follows_js_redirect(self, range_server):
        from conftest import RangeHandler
        RangeHandler.payloads["/page"] = b'''
        <html>
        <script>window.location = "/real-download/setup.exe";</script>
        <a href="/ad/tracking">Ad</a>
        </html>
        '''
        url = _range_url(range_server, "/page")
        links = resolve_page(url)
        exe_links = [l for l in links if ".exe" in l.url]
        assert len(exe_links) >= 1
        assert exe_links[0].source == "js_redirect"

    def test_empty_page(self, range_server):
        from conftest import RangeHandler
        RangeHandler.payloads["/page"] = b"<html><body></body></html>"
        url = _range_url(range_server, "/page")
        links = resolve_page(url)
        assert links == []

    def test_on_progress_called(self, range_server):
        from conftest import RangeHandler
        RangeHandler.payloads["/page"] = b'''
        <a href="/file.zip">Download</a>
        '''
        messages = []
        url = _range_url(range_server, "/page")
        resolve_page(url, on_progress=lambda m: messages.append(m))
        assert len(messages) >= 1

    def test_deduplicates_same_url(self, range_server):
        from conftest import RangeHandler
        RangeHandler.payloads["/page"] = b'''
        <html>
        <a href="/file.zip">Download 1</a>
        <a href="/file.zip">Download 2</a>
        <a href="/file.zip">Download 3</a>
        </html>
        '''
        url = _range_url(range_server, "/page")
        links = resolve_page(url)
        zip_links = [l for l in links if "file.zip" in l.url]
        assert len(zip_links) == 1

    def test_penalizes_ad_signals(self, range_server):
        from conftest import RangeHandler
        RangeHandler.payloads["/page"] = b'''
        <html>
        <a href="/ad/sponsor-click" class="sponsor">Sponsor</a>
        <a href="/real/model.safetensors" class="download-btn">Download Model</a>
        </html>
        '''
        url = _range_url(range_server, "/page")
        links = resolve_page(url)
        # The .safetensors link should rank higher than the ad
        safetensors_idx = next(
            (i for i, l in enumerate(links) if "safetensors" in l.url), -1
        )
        ad_idx = next(
            (i for i, l in enumerate(links) if "sponsor" in l.url), -1
        )
        assert safetensors_idx < ad_idx

    def test_redirects_tracked(self, range_server):
        from conftest import RangeHandler
        RangeHandler.payloads["/page"] = b'''
        <html>
        <script>window.location = "/target.zip";</script>
        </html>
        '''
        url = _range_url(range_server, "/page")
        links = resolve_page(url)
        assert len(links) >= 1

    def test_resolves_relative_urls(self, range_server):
        from conftest import RangeHandler
        RangeHandler.payloads["/downloads/page"] = b'''
        <a href="file.zip">Download</a>
        '''
        url = _range_url(range_server, "/downloads/page")
        links = resolve_page(url)
        zip_links = [l for l in links if "file.zip" in l.url]
        assert len(zip_links) == 1
        assert zip_links[0].url.endswith("/downloads/file.zip")

    def test_max_links_limit(self, range_server):
        from conftest import RangeHandler
        many_links = "".join(
            f'<a href="/file{i}.zip">Link {i}</a>' for i in range(200)
        )
        RangeHandler.payloads["/page"] = f"<html>{many_links}</html>".encode()
        url = _range_url(range_server, "/page")
        links = resolve_page(url, max_links=50)
        assert len(links) <= 50
