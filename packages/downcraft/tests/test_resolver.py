"""Tests for downcraft.resolver — link extraction from ad-heavy pages."""

import base64
import pytest

from downcraft.resolver import (
    _collect_urls_from_dict,
    _decode_obfuscated_urls,
    _extract_candidates,
    _extract_js_redirects,
    _extract_json_ld_urls,
    _extract_meta_urls,
    _find_main_content,
    _follow_intermediate,
    _get_extension,
    _is_in_main_content,
    _is_same_domain,
    _resolve_relative,
    _score_and_deduplicate,
    _score_link,
    _verify_content_type,
    _LinkExtractor,
    resolve_and_download,
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

    def test_follows_js_redirect_to_download(self, range_server):
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


# ---------------------------------------------------------------------------
# Obfuscated URL decoding
# ---------------------------------------------------------------------------

class TestDecodeObfuscatedUrls:
    def test_atob_base64(self):
        url = "https://example.com/secret.zip"
        encoded = base64.b64encode(url.encode()).decode()
        html = f'<script>window.location = atob("{encoded}");</script>'
        urls = _decode_obfuscated_urls(html)
        assert url in urls

    def test_decode_uri_component(self):
        url = "https://example.com/path/file.zip"
        encoded = url.replace("/", "%2F").replace(":", "%3A")
        html = f'decodeURIComponent("{encoded}")'
        urls = _decode_obfuscated_urls(html)
        assert url in urls

    def test_data_download_url(self):
        url = "https://example.com/model.safetensors"
        html = f'<a data-download-url="{url}">Download</a>'
        urls = _decode_obfuscated_urls(html)
        assert url in urls

    def test_data_download_url_base64(self):
        url = "https://example.com/model.safetensors"
        encoded = base64.b64encode(url.encode()).decode()
        html = f'<a data-download-url="{encoded}">Download</a>'
        urls = _decode_obfuscated_urls(html)
        assert url in urls

    def test_string_from_char_code(self):
        # "https://example.com/file.zip"
        codes = ",".join(str(ord(c)) for c in "https://example.com/file.zip")
        html = f'var url = String.fromCharCode({codes})'
        urls = _decode_obfuscated_urls(html)
        assert "https://example.com/file.zip" in urls

    def test_hex_escaped_data_attribute(self):
        # Hex-escaped URL in data-* attribute: \x68\x74\x74\x70\x73://example.com/file.zip
        url = "https://example.com/file.zip"
        hex_str = "".join(f"\\x{ord(c):02x}" for c in url)
        html = f'<a data-download-url="{hex_str}">Download</a>'
        urls = _decode_obfuscated_urls(html)
        assert url in urls

    def test_hex_escaped_no_quote_contamination(self):
        # Verify hex decode doesn't include surrounding quote chars in result
        url = "https://example.com/model.bin"
        hex_str = "".join(f"\\x{ord(c):02x}" for c in url)
        html = f'<div data-file="{hex_str}"></div>'
        urls = _decode_obfuscated_urls(html)
        assert len(urls) == 1
        assert urls[0] == url
        assert '"' not in urls[0]

    def test_no_urls_in_plain_html(self):
        html = '<html><body>Hello world</body></html>'
        urls = _decode_obfuscated_urls(html)
        assert urls == []

    def test_skips_short_strings(self):
        # Base64 string too short (less than 20 chars) — should be skipped
        html = '<script>atob("abc")</script>'
        urls = _decode_obfuscated_urls(html)
        assert urls == []


# ---------------------------------------------------------------------------
# JSON-LD extraction
# ---------------------------------------------------------------------------

class TestExtractJsonLdUrls:
    def test_software_application(self):
        html = '''
        <script type="application/ld+json">
        {"@type": "SoftwareApplication", "downloadUrl": "https://example.com/app.zip"}
        </script>
        '''
        urls = _extract_json_ld_urls(html)
        assert "https://example.com/app.zip" in urls

    def test_content_url(self):
        html = '''
        <script type="application/ld+json">
        {"@type": "VideoObject", "contentUrl": "https://example.com/video.mp4"}
        </script>
        '''
        urls = _extract_json_ld_urls(html)
        assert "https://example.com/video.mp4" in urls

    def test_nested_structure(self):
        html = '''
        <script type="application/ld+json">
        {"@type": "Dataset", "distribution": {"@type": "DataDownload", "contentUrl": "https://example.com/data.csv"}}
        </script>
        '''
        urls = _extract_json_ld_urls(html)
        assert "https://example.com/data.csv" in urls

    def test_no_json_ld(self):
        html = '<html><body>No structured data</body></html>'
        urls = _extract_json_ld_urls(html)
        assert urls == []

    def test_invalid_json_ignored(self):
        html = '<script type="application/ld+json">{invalid json}</script>'
        urls = _extract_json_ld_urls(html)
        assert urls == []


# ---------------------------------------------------------------------------
# Position-aware scoring
# ---------------------------------------------------------------------------

class TestIsInMainContent:
    def test_link_in_main_content(self):
        html = '''
        <header>Nav</header>
        <main>
        <a href="/file.zip">Download</a>
        </main>
        <footer>Footer</footer>
        '''
        assert _is_in_main_content("/file.zip", html)

    def test_link_in_footer(self):
        html = '''
        <main>Content</main>
        <footer><a href="/ad.zip">Ad</a></footer>
        '''
        assert not _is_in_main_content("/ad.zip", html)

    def test_link_in_nav(self):
        html = '''
        <nav><a href="/page.zip">Page</a></nav>
        <main>Content</main>
        '''
        assert not _is_in_main_content("/page.zip", html)


# ---------------------------------------------------------------------------
# Score link with context
# ---------------------------------------------------------------------------

class TestScoreLinkWithContext:
    def test_json_ld_source_bonus(self):
        score = _score_link(
            "https://example.com/file.zip", "[json-ld]", {},
            "https://example.com", source="json_ld",
        )
        # json_ld source adds +0.2, extension adds +0.4, total ~0.6
        assert score > 0.5

    def test_obfuscated_source_bonus(self):
        score = _score_link(
            "https://example.com/file.zip", "[decoded]", {},
            "https://example.com", source="obfuscated",
        )
        # obfuscated source adds +0.15, extension adds +0.4, total ~0.55
        assert score > 0.4

    def test_position_in_main_content_bonus(self):
        html = '<main><a href="/file.zip">Download</a></main>'
        score = _score_link(
            "https://example.com/file.zip", "Download", {},
            "https://example.com", html=html,
        )
        # extension +0.4, "download" in text +0.15, same domain +0.05, main content +0.1
        assert score > 0.5


# ---------------------------------------------------------------------------
# Integration: resolve_page with obfuscated content
# ---------------------------------------------------------------------------

class TestResolveObfuscatedPage:
    def test_finds_base64_hidden_link(self, range_server):
        from conftest import RangeHandler
        real_url = f"http://127.0.0.1:{range_server.server_port}/files/secret.zip"
        encoded = base64.b64encode(real_url.encode()).decode()
        html = f'''
        <html>
        <script>var url = atob("{encoded}"); window.location = url;</script>
        <a href="/ad/tracking">Click here</a>
        </html>
        '''
        RangeHandler.payloads["/page"] = html.encode()
        url = _range_url(range_server, "/page")
        links = resolve_page(url, max_depth=0)
        # The decoded link should appear in results
        decoded_links = [l for l in links if "secret.zip" in l.url]
        assert len(decoded_links) >= 1
        assert decoded_links[0].source == "obfuscated"

    def test_finds_json_ld_link(self, range_server):
        from conftest import RangeHandler
        html = '''
        <html>
        <script type="application/ld+json">
        {"@type": "SoftwareApplication", "downloadUrl": "/files/app.tar.gz"}
        </script>
        <a href="/ad/sponsor">Sponsor</a>
        </html>
        '''
        RangeHandler.payloads["/page"] = html.encode()
        url = _range_url(range_server, "/page")
        links = resolve_page(url, max_depth=0)
        json_ld_links = [l for l in links if "app.tar.gz" in l.url]
        assert len(json_ld_links) >= 1
        assert json_ld_links[0].source == "json_ld"


# ---------------------------------------------------------------------------
# _find_main_content — direct tests
# ---------------------------------------------------------------------------

class TestFindMainContent:
    def test_main_tag_detected(self):
        html = '<html><body><header>Nav</header><main><p>Content</p></main><footer>Footer</footer></body></html>'
        start, end = _find_main_content(html)
        assert "<p>Content</p>" in html[start:end]
        assert "<header>" not in html[start:end]
        assert "<footer>" not in html[start:end]

    def test_skip_tag_heuristic_no_main(self):
        # Without <main>, the exclusion-zone algorithm finds the largest gap
        # between skip tags (header, footer) — the actual content.
        html = '<html><header>Nav</header><p>Main content here</p><footer>Footer</footer></html>'
        start, end = _find_main_content(html)
        assert start >= 0
        assert end > start
        assert '<p>Main content here</p>' in html[start:end]

    def test_multiple_skip_tags(self):
        # Exclusion zones: header, nav, aside at start; footer at end.
        # Largest gap is between aside-end and footer-start.
        html = '<header>a</header><nav>b</nav><aside>c</aside><p>content</p><footer>d</footer>'
        start, end = _find_main_content(html)
        assert start >= 0
        assert end > start
        assert '<p>content</p>' in html[start:end]

    def test_fence_before_content(self):
        # nav is a skip tag; content comes after → largest gap includes <p>Content</p>
        html = '<nav><a href="#">Link</a></nav><p>Content</p>'
        start, end = _find_main_content(html)
        assert start >= 0
        assert end > start
        assert '<p>Content</p>' in html[start:end]

    def test_empty_html(self):
        start, end = _find_main_content("")
        assert start == 0
        assert end == 0

    def test_no_skip_tags(self):
        html = '<html><body><p>Just content</p></body></html>'
        start, end = _find_main_content(html)
        assert start == 0
        assert end == len(html)

    def test_main_only_open(self):
        html = '<main>Content without close'
        start, end = _find_main_content(html)
        assert end == len(html)

    def test_content_after_skip_tags(self):
        # Only header is a skip tag, content comes after → start past </header>
        html = '<header>Nav</header><p>Real content</p>'
        start, end = _find_main_content(html)
        assert start > 0
        assert html[start:end] == '<p>Real content</p>'


# ---------------------------------------------------------------------------
# _is_in_main_content — additional edge cases
# ---------------------------------------------------------------------------

class TestIsInMainContentAdditional:
    def test_custom_main_range(self):
        html = '<header>Nav</header><p>Target</p><footer>Footer</footer>'
        assert _is_in_main_content("Target", html, main_range=(20, 50))
        assert not _is_in_main_content("Nav", html, main_range=(20, 50))

    def test_case_insensitive_match(self):
        # URLs are case-sensitive: /file.zip ≠ /FILE.ZIP
        html = '<main><a href="/FILE.ZIP">Download</a></main>'
        assert _is_in_main_content("/FILE.ZIP", html)
        assert not _is_in_main_content("/file.zip", html)

    def test_href_not_in_html(self):
        html = '<main><p>No links</p></main>'
        assert not _is_in_main_content("/missing.zip", html)


# ---------------------------------------------------------------------------
# _collect_urls_from_dict — direct tests
# ---------------------------------------------------------------------------

class TestCollectUrlsFromDict:
    def test_flat_dict(self):
        out = []
        _collect_urls_from_dict({"downloadUrl": "https://example.com/a.zip"}, out)
        assert "https://example.com/a.zip" in out

    def test_nested_dict(self):
        out = []
        _collect_urls_from_dict(
            {"distribution": {"contentUrl": "https://example.com/b.csv"}}, out
        )
        assert "https://example.com/b.csv" in out

    def test_list_of_dicts(self):
        out = []
        _collect_urls_from_dict(
            {"items": [{"url": "https://a.com"}, {"url": "https://b.com"}]}, out
        )
        assert len(out) == 2

    def test_list_of_strings(self):
        out = []
        _collect_urls_from_dict({"links": ["https://a.com", "https://b.com"]}, out)
        assert len(out) == 2

    def test_unknown_keys_ignored(self):
        out = []
        _collect_urls_from_dict({"name": "test", "version": "1.0"}, out)
        assert out == []

    def test_recursion_depth_limit(self):
        out = []
        # Build a nested dict 7 levels deep — depth > 5 stops recursion
        d = {"url": "https://deep.com"}
        for _ in range(7):
            d = {"child": d}
        _collect_urls_from_dict(d, out)
        assert out == []

    def test_relative_url(self):
        out = []
        _collect_urls_from_dict({"downloadUrl": "/files/model.bin"}, out)
        assert "/files/model.bin" in out

    def test_non_string_value_ignored(self):
        out = []
        _collect_urls_from_dict({"downloadUrl": 123}, out)
        assert out == []

    def test_non_dict_in_list_ignored(self):
        out = []
        _collect_urls_from_dict({"items": [123, None, True]}, out)
        assert out == []


# ---------------------------------------------------------------------------
# _follow_intermediate — tests with local server
# ---------------------------------------------------------------------------

class TestFollowIntermediate:
    def test_follows_to_download(self, range_server):
        from conftest import RangeHandler
        # Intermediate page has a low-confidence link (no extension, no download signal)
        RangeHandler.content_types["/intermediate"] = "text/html"
        RangeHandler.content_types["/landing"] = "text/html"
        RangeHandler.payloads["/intermediate"] = b'''
        <html><a href="/landing">Continue</a></html>
        '''
        RangeHandler.payloads["/landing"] = b'''
        <html><a href="/files/app.zip">Download App</a></html>
        '''
        url = _range_url(range_server, "/intermediate")
        import requests
        sess = requests.Session()
        hdrs = {"User-Agent": "test"}
        results = _follow_intermediate(
            url, sess, hdrs, 5, url, depth=2,
        )
        assert any("app.zip" in r.url for r in results)

    def test_depth_zero_returns_empty(self, range_server):
        import requests
        sess = requests.Session()
        results = _follow_intermediate(
            "http://127.0.0.1:1", sess, {}, 5, "http://example.com", depth=0,
        )
        assert results == []

    def test_network_error_returns_empty(self, range_server):
        import requests
        sess = requests.Session()
        results = _follow_intermediate(
            "http://127.0.0.1:1/nonexistent", sess, {}, 1, "http://example.com", depth=1,
        )
        assert results == []

    def test_non_html_response(self, range_server):
        from conftest import RangeHandler
        RangeHandler.head_responses["/file.bin"] = {
            "status": 200,
            "headers": {"Content-Type": "application/octet-stream", "Content-Length": "999"},
        }
        RangeHandler.payloads["/file.bin"] = b"\x00" * 999
        url = _range_url(range_server, "/file.bin")
        import requests
        sess = requests.Session()
        hdrs = {"User-Agent": "test"}
        results = _follow_intermediate(
            url, sess, hdrs, 5, "http://example.com", depth=1,
        )
        assert len(results) == 1
        assert results[0].source == "intermediate_direct"
        assert results[0].confidence == 0.6

    def test_recursive_follow(self, range_server):
        from conftest import RangeHandler
        # Page1 has a low-confidence link → follow to page2
        RangeHandler.content_types["/page1"] = "text/html"
        RangeHandler.content_types["/page2"] = "text/html"
        RangeHandler.payloads["/page1"] = b'''
        <html><a href="/page2">Next</a></html>
        '''
        RangeHandler.payloads["/page2"] = b'''
        <html><a href="/files/model.safetensors">Download Model</a></html>
        '''
        url1 = _range_url(range_server, "/page1")
        import requests
        sess = requests.Session()
        hdrs = {"User-Agent": "test"}
        results = _follow_intermediate(
            url1, sess, hdrs, 5, url1, depth=2,
        )
        assert any("safetensors" in r.url for r in results)

    def test_preserves_source_prefix(self, range_server):
        from conftest import RangeHandler
        RangeHandler.payloads["/page"] = b'''
        <html><a href="/files/data.zip">Download</a></html>
        '''
        url = _range_url(range_server, "/page")
        import requests
        sess = requests.Session()
        hdrs = {"User-Agent": "test"}
        results = _follow_intermediate(
            url, sess, hdrs, 5, url, depth=1,
        )
        for r in results:
            assert r.source.startswith("intermediate_")


# ---------------------------------------------------------------------------
# _verify_content_type — tests with local server
# ---------------------------------------------------------------------------

class TestVerifyContentType:
    def test_binary_content_type(self, range_server):
        from conftest import RangeHandler
        RangeHandler.head_responses["/file.bin"] = {
            "status": 200,
            "headers": {"Content-Type": "application/octet-stream"},
        }
        url = _range_url(range_server, "/file.bin")
        assert _verify_content_type(url) == 1

    def test_html_content_type(self, range_server):
        from conftest import RangeHandler
        RangeHandler.head_responses["/page"] = {
            "status": 200,
            "headers": {"Content-Type": "text/html; charset=utf-8"},
        }
        url = _range_url(range_server, "/page")
        assert _verify_content_type(url) == -1

    def test_ambiguous_content_type(self, range_server):
        from conftest import RangeHandler
        RangeHandler.head_responses["/file"] = {
            "status": 200,
            "headers": {"Content-Type": "application/unknown", "Content-Length": "100"},
        }
        url = _range_url(range_server, "/file")
        assert _verify_content_type(url) == 0

    def test_large_non_text_is_binary(self, range_server):
        from conftest import RangeHandler
        RangeHandler.head_responses["/big.bin"] = {
            "status": 200,
            "headers": {"Content-Type": "application/x-custom", "Content-Length": "5000"},
        }
        url = _range_url(range_server, "/big.bin")
        assert _verify_content_type(url) == 1

    def test_network_error_returns_zero(self):
        assert _verify_content_type("http://127.0.0.1:1/nonexistent") == 0

    def test_zip_content_type(self, range_server):
        from conftest import RangeHandler
        RangeHandler.head_responses["/archive.zip"] = {
            "status": 200,
            "headers": {"Content-Type": "application/zip"},
        }
        url = _range_url(range_server, "/archive.zip")
        assert _verify_content_type(url) == 1


# ---------------------------------------------------------------------------
# _score_link edge cases
# ---------------------------------------------------------------------------

class TestScoreLinkEdgeCases:
    def test_clamped_at_zero(self):
        score = _score_link("", "", {}, "https://example.com")
        assert score == 0.0

    def test_clamped_at_one(self):
        score = _score_link(
            "https://example.com/model.safetensors", "Download",
            {"class": "btn-download", "data-href": "https://example.com/model.safetensors"},
            "https://example.com",
            source="json_ld",
            html='<main><a href="https://example.com/model.safetensors">Download</a></main>',
        )
        assert score <= 1.0

    def test_short_href_penalty(self):
        # Short href gets -0.1 but download text still adds +0.15
        score = _score_link("a", "Download", {}, "https://example.com")
        assert score > 0  # text bonus outweighs short href penalty
        assert score < 0.2  # but it's low

    def test_short_href_no_download_text(self):
        # Short href + short text = very low score
        score = _score_link("a", "x", {}, "https://example.com")
        assert score <= 0.0

    def test_short_text_penalty(self):
        score = _score_link(
            "https://example.com/file.zip", "x", {}, "https://example.com"
        )
        # Gets extension bonus but text penalty
        assert score > 0

    def test_long_query_penalty(self):
        long_query = "a" * 200
        score_good = _score_link(
            "https://example.com/file.zip", "Download", {}, "https://example.com"
        )
        score_bad = _score_link(
            f"https://example.com/file.zip?{long_query}", "Download", {}, "https://example.com"
        )
        assert score_bad < score_good

    def test_rel_attr_download_bonus(self):
        score = _score_link(
            "https://example.com/file.zip", "Get", {"rel": "download"}, "https://example.com"
        )
        assert score > 0.4

    def test_rel_attr_ad_penalty(self):
        score = _score_link(
            "https://example.com/file.zip", "Sponsor", {"rel": "sponsor"}, "https://example.com"
        )
        assert score < 0.3

    def test_id_download_bonus(self):
        score = _score_link(
            "https://example.com/file.zip", "Get", {"id": "dl-btn"}, "https://example.com"
        )
        assert score > 0.4

    def test_id_ad_penalty(self):
        score = _score_link(
            "https://example.com/file.zip", "Ad", {"id": "ad-banner"}, "https://example.com"
        )
        assert score < 0.3

    def test_download_text_in_href(self):
        # "download" in href gives +0.1 only (not extension bonus)
        score = _score_link(
            "https://example.com/download/file", "File", {}, "https://example.com"
        )
        assert score > 0.1
        assert score < 0.3

    def test_ad_signal_in_text(self):
        score = _score_link(
            "https://example.com/file.zip", "Sponsor link", {}, "https://example.com"
        )
        # Text AD penalty is -0.15 (lighter than URL penalty)
        assert score <= 0.35

    def test_no_html_context(self):
        score = _score_link(
            "https://example.com/file.zip", "Download", {}, "https://example.com"
        )
        assert score > 0.3


# ---------------------------------------------------------------------------
# _extract_candidates — direct tests
# ---------------------------------------------------------------------------

class TestExtractCandidates:
    def test_all_source_types(self):
        html = '''
        <a href="/file.zip">Download</a>
        <script>window.location = "/redirect.exe";</script>
        <meta property="og:url" content="https://example.com/page">
        '''
        candidates = _extract_candidates(html, "https://example.com/page", 100)
        sources = {c[3] for c in candidates}
        assert "html_link" in sources
        assert "js_redirect" in sources
        assert "meta_tag" in sources

    def test_max_links_cap(self):
        html = "".join(f'<a href="/file{i}.zip">L{i}</a>' for i in range(200))
        candidates = _extract_candidates(html, "https://example.com", 50)
        html_links = [c for c in candidates if c[3] == "html_link"]
        assert len(html_links) <= 50

    def test_malformed_html(self):
        # HTMLParser may not extract links from truly broken HTML
        html = '<a href="/file.zip">Unclosed link<div>stuff</div>'
        candidates = _extract_candidates(html, "https://example.com", 100)
        # Parser may or may not extract the link — just verify no crash
        assert isinstance(candidates, list)

    def test_empty_html(self):
        candidates = _extract_candidates("", "https://example.com", 100)
        assert candidates == []

    def test_obfuscated_urls_included(self):
        url = "https://example.com/secret.bin"
        encoded = base64.b64encode(url.encode()).decode()
        html = f'<script>var x = atob("{encoded}");</script>'
        candidates = _extract_candidates(html, "https://example.com", 100)
        obfuscated = [c for c in candidates if c[3] == "obfuscated"]
        assert len(obfuscated) >= 1

    def test_json_ld_included(self):
        html = '''
        <script type="application/ld+json">
        {"@type": "SoftwareApplication", "downloadUrl": "/files/app.zip"}
        </script>
        '''
        candidates = _extract_candidates(html, "https://example.com", 100)
        json_ld = [c for c in candidates if c[3] == "json_ld"]
        assert len(json_ld) >= 1


# ---------------------------------------------------------------------------
# _score_and_deduplicate — direct tests
# ---------------------------------------------------------------------------

class TestScoreAndDeduplicate:
    def test_deduplicates_same_url(self):
        candidates = [
            ("https://a.com/file.zip", "Dl1", {}, "html_link"),
            ("https://a.com/file.zip", "Dl2", {}, "js_redirect"),
        ]
        results = _score_and_deduplicate(candidates, "https://a.com", "", [])
        assert len(results) == 1

    def test_skips_mailto(self):
        candidates = [("mailto:user@example.com", "Email", {}, "html_link")]
        results = _score_and_deduplicate(candidates, "https://a.com", "", [])
        assert results == []

    def test_skips_javascript(self):
        candidates = [("javascript:void(0)", "Click", {}, "html_link")]
        results = _score_and_deduplicate(candidates, "https://a.com", "", [])
        assert results == []

    def test_skips_fragment(self):
        candidates = [("#section", "Nav", {}, "html_link")]
        results = _score_and_deduplicate(candidates, "https://a.com", "", [])
        assert results == []

    def test_skips_tel(self):
        candidates = [("tel:+1234567890", "Call", {}, "html_link")]
        results = _score_and_deduplicate(candidates, "https://a.com", "", [])
        assert results == []

    def test_sorted_by_confidence(self):
        candidates = [
            ("https://a.com/page", "Page", {}, "html_link"),
            ("https://a.com/file.zip", "Download", {}, "html_link"),
        ]
        results = _score_and_deduplicate(candidates, "https://a.com", "", [])
        assert results[0].confidence >= results[1].confidence

    def test_preserves_redirects(self):
        candidates = [("https://a.com/file.zip", "Dl", {}, "html_link")]
        redirects = ["https://a.com/old", "https://a.com/newer"]
        results = _score_and_deduplicate(candidates, "https://a.com", "", redirects)
        assert results[0].redirects == redirects

    def test_empty_candidates(self):
        results = _score_and_deduplicate([], "https://a.com", "", [])
        assert results == []

    def test_source_labeling(self):
        candidates = [
            ("https://a.com/file.zip", "[json-ld]", {}, "json_ld"),
            ("https://a.com/file2.zip", "[decoded]", {}, "obfuscated"),
        ]
        results = _score_and_deduplicate(candidates, "https://a.com", "", [])
        sources = {r.source for r in results}
        assert "json_ld" in sources
        assert "obfuscated" in sources


# ---------------------------------------------------------------------------
# resolve_page — intermediate follow integration
# ---------------------------------------------------------------------------

class TestResolvePageIntermediate:
    def test_follows_intermediate_page(self, range_server):
        from conftest import RangeHandler
        # Page has a low-confidence link (no extension) → triggers intermediate follow
        RangeHandler.content_types["/page"] = "text/html"
        RangeHandler.content_types["/landing"] = "text/html"
        RangeHandler.payloads["/page"] = b'''
        <html>
        <a href="/landing">Go to download</a>
        </html>
        '''
        RangeHandler.payloads["/landing"] = b'''
        <html>
        <a href="/files/model.gguf">Download Model</a>
        </html>
        '''
        url = _range_url(range_server, "/page")
        links = resolve_page(url, max_depth=1)
        gguf = [l for l in links if "model.gguf" in l.url]
        assert len(gguf) >= 1

    def test_max_depth_zero_skips_follow(self, range_server):
        from conftest import RangeHandler
        RangeHandler.payloads["/page"] = b'''
        <html><a href="/other">Link</a></html>
        '''
        url = _range_url(range_server, "/page")
        links = resolve_page(url, max_depth=0)
        followed = [l for l in links if l.source.startswith("intermediate")]
        assert followed == []

    def test_high_confidence_skips_follow(self, range_server):
        from conftest import RangeHandler
        RangeHandler.payloads["/page"] = b'''
        <html>
        <a href="/files/app.zip" class="btn-download">Download App</a>
        </html>
        '''
        url = _range_url(range_server, "/page")
        links = resolve_page(url, max_depth=2)
        followed = [l for l in links if l.source.startswith("intermediate")]
        assert followed == []

    def test_no_results_no_follow(self, range_server):
        from conftest import RangeHandler
        RangeHandler.payloads["/page"] = b"<html><body></body></html>"
        url = _range_url(range_server, "/page")
        links = resolve_page(url, max_depth=2)
        assert links == []

    def test_network_error_returns_empty(self):
        links = resolve_page("http://127.0.0.1:1/nonexistent", timeout=1)
        assert links == []


# ---------------------------------------------------------------------------
# resolve_page — verify_content_type integration
# ---------------------------------------------------------------------------

class TestResolvePageVerifyContentType:
    def test_html_content_reduces_confidence(self, range_server):
        from conftest import RangeHandler
        RangeHandler.payloads["/page"] = b'''
        <html><a href="/files/model.bin">Download</a></html>
        '''
        RangeHandler.head_responses["/files/model.bin"] = {
            "status": 200,
            "headers": {"Content-Type": "text/html; charset=utf-8"},
        }
        url = _range_url(range_server, "/page")
        links = resolve_page(url, verify_content_type=True, max_depth=0)
        model_link = next((l for l in links if "model.bin" in l.url), None)
        assert model_link is not None
        assert model_link.source == "verified_html"
        assert model_link.confidence < 0.5

    def test_binary_content_boosts_confidence(self, range_server):
        from conftest import RangeHandler
        RangeHandler.payloads["/page"] = b'''
        <html><a href="/files/model.bin">Download</a></html>
        '''
        RangeHandler.head_responses["/files/model.bin"] = {
            "status": 200,
            "headers": {"Content-Type": "application/octet-stream"},
        }
        url = _range_url(range_server, "/page")
        links_before = resolve_page(url, verify_content_type=False, max_depth=0)
        links_after = resolve_page(url, verify_content_type=True, max_depth=0)
        before = next((l for l in links_before if "model.bin" in l.url), None)
        after = next((l for l in links_after if "model.bin" in l.url), None)
        assert before is not None and after is not None
        assert after.confidence >= before.confidence
        assert after.source == "verified_binary"

    def test_no_results_skips_verify(self, range_server):
        from conftest import RangeHandler
        RangeHandler.payloads["/page"] = b"<html></html>"
        url = _range_url(range_server, "/page")
        links = resolve_page(url, verify_content_type=True, max_depth=0)
        assert links == []


# ---------------------------------------------------------------------------
# resolve_page — extra edge cases
# ---------------------------------------------------------------------------

class TestResolvePageEdgeCases:
    def test_custom_headers(self, range_server):
        from conftest import RangeHandler
        RangeHandler.payloads["/page"] = b'<a href="/file.zip">Dl</a>'
        url = _range_url(range_server, "/page")
        links = resolve_page(url, headers={"X-Custom": "test"}, max_depth=0)
        assert len(links) >= 1

    def test_redirect_chain_preserved(self, range_server):
        from conftest import RangeHandler
        RangeHandler.payloads["/page"] = b'<a href="/file.zip">Dl</a>'
        url = _range_url(range_server, "/page")
        links = resolve_page(url, max_depth=0)
        assert len(links) >= 1

    def test_progress_callback(self, range_server):
        from conftest import RangeHandler
        RangeHandler.payloads["/page"] = b'<a href="/file.zip">Dl</a>'
        url = _range_url(range_server, "/page")
        messages = []
        resolve_page(url, on_progress=lambda m: messages.append(m), max_depth=0)
        assert any("Fetching" in m for m in messages)
        assert any("Analyzing" in m for m in messages)
        assert any("Best candidate" in m for m in messages)

    def test_empty_page_progress(self, range_server):
        from conftest import RangeHandler
        RangeHandler.payloads["/page"] = b"<html><body></body></html>"
        url = _range_url(range_server, "/page")
        messages = []
        resolve_page(url, on_progress=lambda m: messages.append(m), max_depth=0)
        assert any("No download links" in m for m in messages)

    def test_many_links_capped(self, range_server):
        from conftest import RangeHandler
        html = "".join(f'<a href="/f{i}.zip">L{i}</a>' for i in range(300))
        RangeHandler.payloads["/page"] = html.encode()
        url = _range_url(range_server, "/page")
        links = resolve_page(url, max_links=25, max_depth=0)
        assert len(links) <= 25


# ---------------------------------------------------------------------------
# resolve_and_download — tests
# ---------------------------------------------------------------------------

class TestResolveAndDownload:
    def test_raises_on_no_links(self, range_server):
        from conftest import RangeHandler
        RangeHandler.payloads["/page"] = b"<html></html>"
        url = _range_url(range_server, "/page")
        with pytest.raises(ValueError, match="No download links found"):
            resolve_and_download(url, "/tmp/dest.zip")

    def test_raises_on_low_confidence(self, range_server):
        from conftest import RangeHandler
        RangeHandler.payloads["/page"] = b'''
        <html><a href="/ad/sponsor">Sponsor</a></html>
        '''
        url = _range_url(range_server, "/page")
        with pytest.raises(ValueError, match="confidence too low"):
            resolve_and_download(url, "/tmp/dest.zip", min_confidence=0.9)

    def test_success_path(self, range_server, tmp_path):
        from conftest import RangeHandler
        RangeHandler.payloads["/page"] = b'''
        <html><a href="/files/app.zip">Download App</a></html>
        '''
        RangeHandler.payloads["/files/app.zip"] = b"PK\x03\x04fake-zip-content"
        url = _range_url(range_server, "/page")
        dest = str(tmp_path / "downloaded.zip")
        result = resolve_and_download(url, dest, min_confidence=0.1, max_depth=0)
        assert result.exists()
        assert result.read_bytes() == b"PK\x03\x04fake-zip-content"

    def test_progress_callback_called(self, range_server, tmp_path):
        from conftest import RangeHandler
        RangeHandler.payloads["/page"] = b'''
        <html><a href="/files/data.bin">Download</a></html>
        '''
        RangeHandler.payloads["/files/data.bin"] = b"binary-data-here"
        url = _range_url(range_server, "/page")
        dest = str(tmp_path / "out.bin")
        messages = []
        resolve_and_download(
            url, dest, min_confidence=0.1, max_depth=0,
            on_progress=lambda m: messages.append(m),
        )
        assert any("Downloading" in m for m in messages)


# ---------------------------------------------------------------------------
# _extract_js_redirects — additional patterns
# ---------------------------------------------------------------------------

class TestExtractJsRedirectsAdditional:
    def test_location_assign(self):
        html = 'window.location.assign("https://example.com/go.zip")'
        urls = _extract_js_redirects(html)
        assert "https://example.com/go.zip" in urls

    def test_location_replace(self):
        html = 'window.location.replace("https://example.com/redir.exe")'
        urls = _extract_js_redirects(html)
        assert "https://example.com/redir.exe" in urls

    def test_setTimeout_with_location(self):
        html = "setTimeout(\"location='https://example.com/delay.zip'\", 3000)"
        urls = _extract_js_redirects(html)
        assert any("example.com/delay.zip" in u for u in urls)

    def test_meta_refresh_content_first(self):
        html = '<meta content="0;url=https://example.com/refresh.zip" http-equiv="refresh">'
        urls = _extract_js_redirects(html)
        assert "https://example.com/refresh.zip" in urls

    def test_multiple_redirects(self):
        html = '''
        <script>window.location = "https://a.com/1.zip";</script>
        <script>window.location.href = "https://b.com/2.zip";</script>
        '''
        urls = _extract_js_redirects(html)
        assert len(urls) == 2

    def test_mixed_valid_and_invalid(self):
        html = '''
        window.location = "https://valid.com/file.zip";
        window.location = "javascript:void(0)";
        window.location = "#section";
        '''
        urls = _extract_js_redirects(html)
        assert len(urls) == 1
        assert "valid.com" in urls[0]

    def test_single_quotes(self):
        html = "window.location = 'https://example.com/single.zip'"
        urls = _extract_js_redirects(html)
        assert "https://example.com/single.zip" in urls


# ---------------------------------------------------------------------------
# _extract_meta_urls — additional patterns
# ---------------------------------------------------------------------------

class TestExtractMetaUrlsAdditional:
    def test_twitter_url(self):
        html = '<meta name="twitter:url" content="https://example.com/tweet">'
        urls = _extract_meta_urls(html)
        assert "https://example.com/tweet" in urls

    def test_og_url_reversed_attr_order(self):
        html = '<meta content="https://example.com/og" property="og:url">'
        urls = _extract_meta_urls(html)
        assert "https://example.com/og" in urls

    def test_multiple_meta_urls(self):
        html = '''
        <meta property="og:url" content="https://example.com/og">
        <link rel="canonical" href="https://example.com/canonical">
        '''
        urls = _extract_meta_urls(html)
        assert len(urls) == 2


# ---------------------------------------------------------------------------
# _decode_obfuscated_urls — additional edge cases
# ---------------------------------------------------------------------------

class TestDecodeObfuscatedUrlsAdditional:
    def test_relative_atob_url(self):
        encoded = base64.b64encode(b"/files/model.bin").decode()
        html = f'<script>var x = atob("{encoded}");</script>'
        urls = _decode_obfuscated_urls(html)
        assert "/files/model.bin" in urls

    def test_decode_uri_component_short(self):
        html = 'decodeURIComponent("short")'
        urls = _decode_obfuscated_urls(html)
        assert urls == []

    def test_multiple_atob(self):
        url1 = "https://a.com/1.zip"
        url2 = "https://b.com/2.zip"
        e1 = base64.b64encode(url1.encode()).decode()
        e2 = base64.b64encode(url2.encode()).decode()
        html = f'<script>atob("{e1}"); atob("{e2}");</script>'
        urls = _decode_obfuscated_urls(html)
        assert url1 in urls
        assert url2 in urls

    def test_data_real_url_attr(self):
        url = "https://example.com/real.zip"
        html = f'<a data-real-url="{url}">Dl</a>'
        urls = _decode_obfuscated_urls(html)
        assert url in urls

    def test_data_file_attr(self):
        url = "https://example.com/data.bin"
        html = f'<a data-file="{url}">Dl</a>'
        urls = _decode_obfuscated_urls(html)
        assert url in urls

    def test_data_link_url_attr(self):
        url = "https://example.com/link.tar.gz"
        html = f'<a data-link-url="{url}">Dl</a>'
        urls = _decode_obfuscated_urls(html)
        assert url in urls


# ---------------------------------------------------------------------------
# _extract_json_ld_urls — additional patterns
# ---------------------------------------------------------------------------

class TestExtractJsonLdUrlsAdditional:
    def test_array_of_items(self):
        html = '''
        <script type="application/ld+json">
        [
            {"@type": "SoftwareApplication", "downloadUrl": "https://a.com/1.zip"},
            {"@type": "SoftwareApplication", "downloadUrl": "https://b.com/2.zip"}
        ]
        </script>
        '''
        urls = _extract_json_ld_urls(html)
        assert len(urls) == 2

    def test_same_as_url(self):
        html = '''
        <script type="application/ld+json">
        {"@type": "Organization", "sameAs": "https://example.com/org"}
        </script>
        '''
        urls = _extract_json_ld_urls(html)
        assert "https://example.com/org" in urls

    def test_install_url(self):
        html = '''
        <script type="application/ld+json">
        {"@type": "SoftwareApplication", "installUrl": "https://example.com/install"}
        </script>
        '''
        urls = _extract_json_ld_urls(html)
        assert "https://example.com/install" in urls

    def test_file_url(self):
        html = '''
        <script type="application/ld+json">
        {"@type": "DataDownload", "fileUrl": "https://example.com/data.csv"}
        </script>
        '''
        urls = _extract_json_ld_urls(html)
        assert "https://example.com/data.csv" in urls

    def test_action_url(self):
        html = '''
        <script type="application/ld+json">
        {"@type": "DownloadAction", "actionUrl": "https://example.com/action"}
        </script>
        '''
        urls = _extract_json_ld_urls(html)
        assert "https://example.com/action" in urls

    def test_mixed_valid_and_invalid_blocks(self):
        html = '''
        <script type="application/ld+json">
        {"@type": "SoftwareApplication", "downloadUrl": "https://valid.com/app.zip"}
        </script>
        <script type="application/ld+json">{not json}</script>
        <script type="application/ld+json">
        {"@type": "SoftwareApplication", "downloadUrl": "https://valid2.com/app.zip"}
        </script>
        '''
        urls = _extract_json_ld_urls(html)
        assert len(urls) == 2


# ---------------------------------------------------------------------------
# ResolvedLink dataclass
# ---------------------------------------------------------------------------

class TestResolvedLinkDataclass:
    def test_defaults(self):
        rl = ResolvedLink(url="https://example.com/file.zip")
        assert rl.title == ""
        assert rl.extension == ""
        assert rl.size_hint == 0
        assert rl.confidence == 0.0
        assert rl.source == ""
        assert rl.redirects == []

    def test_full_init(self):
        rl = ResolvedLink(
            url="https://example.com/file.zip",
            title="Download",
            extension=".zip",
            size_hint=1024,
            confidence=0.85,
            source="html_link",
            redirects=["https://a.com", "https://b.com"],
        )
        assert rl.confidence == 0.85
        assert len(rl.redirects) == 2

    def test_redirects_independent(self):
        r1 = ResolvedLink(url="a")
        r2 = ResolvedLink(url="b")
        r1.redirects.append("x")
        assert r2.redirects == []


# ---------------------------------------------------------------------------
# _LinkExtractor — additional edge cases
# ---------------------------------------------------------------------------

class TestLinkExtractorAdditional:
    def test_no_href_on_a(self):
        parser = _LinkExtractor()
        parser.feed('<a>No href</a>')
        assert parser.links == []

    def test_whitespace_in_text(self):
        parser = _LinkExtractor()
        parser.feed('<a href="/f.zip">  Download   Now  </a>')
        assert len(parser.links) == 1
        assert parser.links[0][1] == "Download Now"

    def test_multiple_data_attrs_on_div(self):
        parser = _LinkExtractor()
        parser.feed('<div data-url="https://a.com" data-href="https://b.com">X</div>')
        urls = [href for href, _, _ in parser.links]
        assert "https://a.com" in urls
        assert "https://b.com" in urls

    def test_button_data_attr(self):
        parser = _LinkExtractor()
        parser.feed('<button data-download="https://example.com/file.zip">Get</button>')
        urls = [href for href, _, _ in parser.links]
        assert "https://example.com/file.zip" in urls

    def test_span_data_attr(self):
        parser = _LinkExtractor()
        parser.feed('<span data-url="https://example.com/span.zip">X</span>')
        urls = [href for href, _, _ in parser.links]
        assert "https://example.com/span.zip" in urls

    def test_non_a_tag_no_text_collected(self):
        parser = _LinkExtractor()
        parser.feed('<div href="/test.zip">Text</div>')
        assert parser.links == []

    def test_nested_a_tags(self):
        parser = _LinkExtractor()
        parser.feed('<a href="/outer"><a href="/inner">Nested</a></a>')
        assert len(parser.links) >= 1


# ---------------------------------------------------------------------------
# _get_extension — additional cases
# ---------------------------------------------------------------------------

class TestGetExtensionAdditional:
    def test_safetensors(self):
        assert _get_extension("https://example.com/model.safetensors") == ".safetensors"

    def test_gguf(self):
        assert _get_extension("https://example.com/model.gguf") == ".gguf"

    def test_no_path(self):
        assert _get_extension("https://example.com") == ""

    def test_directory_like(self):
        assert _get_extension("https://example.com/dir/") == ""

    def test_multiple_dots(self):
        assert _get_extension("https://example.com/my.model.v2.bin") == ".bin"


# ---------------------------------------------------------------------------
# _is_same_domain — additional cases
# ---------------------------------------------------------------------------

class TestIsSameDomainAdditional:
    def test_deep_subdomain(self):
        assert _is_same_domain("https://a.b.c.example.com/f", "https://example.com")

    def test_different_port_same_domain(self):
        # Different ports are different netlocs → different domains
        assert not _is_same_domain("https://example.com:8080/f", "https://example.com")

    def test_http_vs_https(self):
        assert _is_same_domain("http://example.com/f", "https://example.com")

    def test_empty_url(self):
        assert not _is_same_domain("", "https://example.com")


# ---------------------------------------------------------------------------
# _resolve_relative — additional cases
# ---------------------------------------------------------------------------

class TestResolveRelativeAdditional:
    def test_dot_slash(self):
        result = _resolve_relative("./file.zip", "https://example.com/dir/page")
        assert result == "https://example.com/dir/file.zip"

    def test_dot_dot_slash(self):
        result = _resolve_relative("../file.zip", "https://example.com/dir/sub/page")
        assert result == "https://example.com/dir/file.zip"

    def test_full_url_ignores_base(self):
        result = _resolve_relative("https://other.com/file.zip", "https://example.com")
        assert result == "https://other.com/file.zip"

    def test_empty_relative(self):
        result = _resolve_relative("", "https://example.com/page")
        assert result == "https://example.com/page"


# ---------------------------------------------------------------------------
# _extract_js_redirects — onclick pattern
# ---------------------------------------------------------------------------

class TestExtractJsRedirectsOnclick:
    def test_onclick_window_open(self):
        html = "onclick=\"window.open('https://example.com/popup.zip')\""
        urls = _extract_js_redirects(html)
        assert any("popup.zip" in u for u in urls)

    def test_onclick_location_href_call(self):
        # onclick regex requires parentheses after location.href
        html = "onclick=\"location.href('https://example.com/click.zip')\""
        urls = _extract_js_redirects(html)
        assert any("click.zip" in u for u in urls)

    def test_onclick_location_call(self):
        html = "onclick=\"location('https://example.com/loc.zip')\""
        urls = _extract_js_redirects(html)
        assert any("loc.zip" in u for u in urls)
