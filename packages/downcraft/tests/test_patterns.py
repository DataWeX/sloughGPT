"""Tests for downcraft.patterns — pattern matching for download page structures."""

import base64
import pytest

from downcraft.resolve.patterns import (
    DataAttributeMatcher,
    EmbeddedPlayerMatcher,
    Extraction,
    JsonBlobMatcher,
    JsonLdMatcher,
    JsRedirectMatcher,
    JsVariableMatcher,
    OEmbedMatcher,
    ObfuscationMatcher,
    extract_all,
)


# ---------------------------------------------------------------------------
# Extraction dataclass
# ---------------------------------------------------------------------------

class TestExtraction:
    def test_fields(self):
        e = Extraction(url="https://a.com/f.zip", source="test", confidence=0.9)
        assert e.url == "https://a.com/f.zip"
        assert e.source == "test"
        assert e.confidence == 0.9

    def test_default_confidence(self):
        e = Extraction(url="https://a.com/f.zip", source="test")
        assert e.confidence == 1.0

    def test_frozen(self):
        e = Extraction(url="https://a.com/f.zip", source="test")
        with pytest.raises(AttributeError):
            e.url = "https://other.com"


# ---------------------------------------------------------------------------
# JsVariableMatcher
# ---------------------------------------------------------------------------

class TestJsVariableMatcher:
    def setup_method(self):
        self.m = JsVariableMatcher()

    def test_name(self):
        assert self.m.name == "js_variable"

    def test_var_download_url(self):
        html = 'var downloadUrl = "https://cdn.example.com/file.zip";'
        results = self.m.extract(html)
        assert len(results) == 1
        assert results[0].url == "https://cdn.example.com/file.zip"
        assert results[0].source == "js_variable"

    def test_let_real_url(self):
        html = "let realUrl = 'https://real-host.com/model.safetensors';"
        results = self.m.extract(html)
        assert len(results) == 1
        assert results[0].url == "https://real-host.com/model.safetensors"

    def test_window_variable(self):
        html = 'window.downloadUrl = "https://window.com/file.zip";'
        results = self.m.extract(html)
        assert len(results) == 1
        assert results[0].url == "https://window.com/file.zip"

    def test_multiple_variables(self):
        html = '''
        var url1 = "https://a.com/1.zip";
        let url2 = "https://b.com/2.bin";
        window.url3 = "https://c.com/3.tar";
        '''
        results = self.m.extract(html)
        assert len(results) == 3
        urls = [r.url for r in results]
        assert "https://a.com/1.zip" in urls
        assert "https://b.com/2.bin" in urls
        assert "https://c.com/3.tar" in urls

    def test_ignores_javascript_prefix(self):
        html = 'var url = "javascript:alert(1)";'
        assert self.m.extract(html) == []

    def test_ignores_hash(self):
        html = 'var url = "#section";'
        assert self.m.extract(html) == []

    def test_ignores_void(self):
        html = 'var url = "void(0)";'
        assert self.m.extract(html) == []

    def test_ignores_unrelated_variables(self):
        html = 'var userName = "John"; let count = 42;'
        assert self.m.extract(html) == []

    def test_relative_url(self):
        html = 'var downloadUrl = "/files/model.bin";'
        results = self.m.extract(html)
        assert len(results) == 1
        assert results[0].url == "/files/model.bin"

    def test_case_insensitive(self):
        html = 'VAR DownloadUrl = "https://a.com/f.zip";'
        results = self.m.extract(html)
        assert len(results) == 1

    def test_const_href(self):
        html = 'const finalHref = "https://final.com/data.bin";'
        results = self.m.extract(html)
        assert len(results) == 1
        assert results[0].url == "https://final.com/data.bin"


# ---------------------------------------------------------------------------
# JsRedirectMatcher
# ---------------------------------------------------------------------------

class TestJsRedirectMatcher:
    def setup_method(self):
        self.m = JsRedirectMatcher()

    def test_name(self):
        assert self.m.name == "js_redirect"

    def test_window_location(self):
        html = 'window.location = "https://redirect.com/file.zip";'
        results = self.m.extract(html)
        assert any(r.url == "https://redirect.com/file.zip" for r in results)

    def test_window_location_href(self):
        html = 'window.location.href = "https://redirect.com/file.zip";'
        results = self.m.extract(html)
        assert any(r.url == "https://redirect.com/file.zip" for r in results)

    def test_window_location_assign(self):
        html = 'window.location.assign("https://assign.com/file.zip");'
        results = self.m.extract(html)
        assert any(r.url == "https://assign.com/file.zip" for r in results)

    def test_window_location_replace(self):
        html = 'window.location.replace("https://replace.com/file.zip");'
        results = self.m.extract(html)
        assert any(r.url == "https://replace.com/file.zip" for r in results)

    def test_document_location(self):
        html = 'document.location = "https://doc.com/file.zip";'
        results = self.m.extract(html)
        assert any(r.url == "https://doc.com/file.zip" for r in results)

    def test_meta_refresh(self):
        html = '<meta http-equiv="refresh" content="5;url=https://meta.com/file.zip">'
        results = self.m.extract(html)
        assert any(r.url == "https://meta.com/file.zip" for r in results)

    def test_meta_refresh_content_first(self):
        html = '<meta content="5;url=https://meta2.com/file.zip" http-equiv="refresh">'
        results = self.m.extract(html)
        assert any(r.url == "https://meta2.com/file.zip" for r in results)

    def test_ignores_javascript_prefix(self):
        html = 'window.location = "javascript:void(0)";'
        results = self.m.extract(html)
        assert not any(r.url.startswith("javascript:") for r in results)

    def test_ignores_hash(self):
        html = 'window.location = "#top";'
        results = self.m.extract(html)
        assert not any(r.url == "#top" for r in results)


# ---------------------------------------------------------------------------
# DataAttributeMatcher
# ---------------------------------------------------------------------------

class TestDataAttributeMatcher:
    def setup_method(self):
        self.m = DataAttributeMatcher()

    def test_name(self):
        assert self.m.name == "data_attribute"

    def test_data_download_url(self):
        html = '<a data-download-url="https://dl.com/file.zip">Download</a>'
        results = self.m.extract(html)
        assert any(r.url == "https://dl.com/file.zip" for r in results)

    def test_data_real_url(self):
        html = '<div data-real-url="https://real.com/file.zip"></div>'
        results = self.m.extract(html)
        assert any(r.url == "https://real.com/file.zip" for r in results)

    def test_data_countdown_url(self):
        html = '<div data-countdown-url="https://timer.com/file.zip"></div>'
        results = self.m.extract(html)
        assert any(r.url == "https://timer.com/file.zip" for r in results)

    def test_data_popunder(self):
        html = '<div data-popunder="https://pop.com/file.zip"></div>'
        results = self.m.extract(html)
        assert any(r.url == "https://pop.com/file.zip" for r in results)

    def test_data_href(self):
        html = '<a data-href="https://href.com/file.zip">Link</a>'
        results = self.m.extract(html)
        assert any(r.url == "https://href.com/file.zip" for r in results)

    def test_base64_encoded(self):
        url = "https://encoded.com/file.zip"
        b64 = base64.b64encode(url.encode()).decode()
        html = f'<div data-download-url="{b64}"></div>'
        results = self.m.extract(html)
        assert any(r.url == url for r in results)

    def test_relative_url(self):
        html = '<a data-download="/files/model.bin">Download</a>'
        results = self.m.extract(html)
        assert any(r.url == "/files/model.bin" for r in results)

    def test_multiple_attributes(self):
        html = '''
        <div data-countdown-url="https://a.com/1.zip"></div>
        <div data-real-url="https://b.com/2.zip"></div>
        '''
        results = self.m.extract(html)
        urls = [r.url for r in results]
        assert "https://a.com/1.zip" in urls
        assert "https://b.com/2.zip" in urls

    def test_ignores_empty_value(self):
        html = '<div data-download-url=""></div>'
        assert self.m.extract(html) == []

    def test_ignores_plain_text(self):
        html = '<div data-download-url="not a url"></div>'
        assert self.m.extract(html) == []


# ---------------------------------------------------------------------------
# ObfuscationMatcher
# ---------------------------------------------------------------------------

class TestObfuscationMatcher:
    def setup_method(self):
        self.m = ObfuscationMatcher()

    def test_name(self):
        assert self.m.name == "obfuscated"

    def test_atob(self):
        url = "https://obfuscated.com/file.zip"
        b64 = base64.b64encode(url.encode()).decode()
        html = f'var x = atob("{b64}");'
        results = self.m.extract(html)
        assert any(r.url == url for r in results)

    def test_decode_uri_component(self):
        import urllib.parse
        encoded = urllib.parse.quote("https://decoded.com/file.zip")
        html = f'var x = decodeURIComponent("{encoded}");'
        results = self.m.extract(html)
        assert any(r.url == "https://decoded.com/file.zip" for r in results)

    def test_from_char_code(self):
        url = "https://char.com/file.zip"
        codes = ",".join(str(ord(c)) for c in url)
        html = f'var x = String.fromCharCode({codes});'
        results = self.m.extract(html)
        assert any(r.url == url for r in results)

    def test_hex_data_attr(self):
        url = "https://hex.com/file.zip"
        hex_str = "\\x".join(format(ord(c), "02x") for c in url)
        hex_str = "\\x" + hex_str
        html = f'<div data-url="{hex_str}"></div>'
        results = self.m.extract(html)
        assert any(r.url == url for r in results)

    def test_atob_too_short_ignored(self):
        html = 'var x = atob("short");'
        assert self.m.extract(html) == []

    def test_from_char_code_empty(self):
        html = 'var x = String.fromCharCode();'
        assert self.m.extract(html) == []


# ---------------------------------------------------------------------------
# JsonBlobMatcher
# ---------------------------------------------------------------------------

class TestJsonBlobMatcher:
    def setup_method(self):
        self.m = JsonBlobMatcher()

    def test_name(self):
        assert self.m.name == "json_blob"

    def test_initial_state(self):
        html = '''
        <script>
        window.__INITIAL_STATE__ = {"download": {"url": "https://state.com/file.bin"}};
        </script>
        '''
        results = self.m.extract(html)
        assert any(r.url == "https://state.com/file.bin" for r in results)

    def test_nuxt(self):
        html = '''
        <script>
        window.__NUXT__ = {"config": {"downloadUrl": "https://nuxt.com/data.zip"}};
        </script>
        '''
        results = self.m.extract(html)
        assert any(r.url == "https://nuxt.com/data.zip" for r in results)

    def test_next_data(self):
        html = '''
        <script>
        window.__NEXT_DATA__ = {"props": {"contentUrl": "https://next.com/file.tar"}};
        </script>
        '''
        results = self.m.extract(html)
        assert any(r.url == "https://next.com/file.tar" for r in results)

    def test_config_variable(self):
        html = '''
        <script>
        var config = {"downloadUrl": "https://cfg.com/model.onnx"};
        </script>
        '''
        results = self.m.extract(html)
        assert any(r.url == "https://cfg.com/model.onnx" for r in results)

    def test_nested_urls(self):
        html = '''
        <script>
        window.__INITIAL_STATE__ = {
            "files": [
                {"downloadUrl": "https://a.com/1.zip"},
                {"url": "https://b.com/2.bin"}
            ]
        };
        </script>
        '''
        results = self.m.extract(html)
        urls = [r.url for r in results]
        assert "https://a.com/1.zip" in urls
        assert "https://b.com/2.bin" in urls

    def test_invalid_json_ignored(self):
        html = '''
        <script>
        window.__INITIAL_STATE__ = {invalid json};
        </script>
        '''
        assert self.m.extract(html) == []

    def test_no_blob_returns_empty(self):
        html = '<p>No script tags here</p>'
        assert self.m.extract(html) == []

    def test_relative_urls(self):
        html = '''
        <script>
        window.__INITIAL_STATE__ = {"fileUrl": "/files/model.bin"};
        </script>
        '''
        results = self.m.extract(html)
        assert any(r.url == "/files/model.bin" for r in results)


# ---------------------------------------------------------------------------
# JsonLdMatcher
# ---------------------------------------------------------------------------

class TestJsonLdMatcher:
    def setup_method(self):
        self.m = JsonLdMatcher()

    def test_name(self):
        assert self.m.name == "json_ld"

    def test_download_url(self):
        html = '''
        <script type="application/ld+json">
        {"@type": "SoftwareApplication", "downloadUrl": "https://ld.com/app.zip"}
        </script>
        '''
        results = self.m.extract(html)
        assert any(r.url == "https://ld.com/app.zip" for r in results)

    def test_content_url(self):
        html = '''
        <script type="application/ld+json">
        {"@type": "MediaObject", "contentUrl": "https://ld.com/video.mp4"}
        </script>
        '''
        results = self.m.extract(html)
        assert any(r.url == "https://ld.com/video.mp4" for r in results)

    def test_invalid_json_ignored(self):
        html = '''
        <script type="application/ld+json">
        {invalid json}
        </script>
        '''
        assert self.m.extract(html) == []

    def test_no_json_ld(self):
        html = '<p>No structured data</p>'
        assert self.m.extract(html) == []


# ---------------------------------------------------------------------------
# extract_all
# ---------------------------------------------------------------------------

class TestExtractAll:
    def test_deduplicates_urls(self):
        html = '''
        <script>var url = "https://example.com/file.zip";</script>
        <div data-download-url="https://example.com/file.zip"></div>
        '''
        results = extract_all(html)
        urls = [r.url for r in results]
        assert urls.count("https://example.com/file.zip") == 1

    def test_empty_html(self):
        assert extract_all("") == []

    def test_no_matches(self):
        html = '<p>Hello world</p>'
        assert extract_all(html) == []

    def test_multiple_matchers(self):
        html = '''
        <script>var url1 = "https://a.com/1.zip";</script>
        <div data-real-url="https://b.com/2.zip"></div>
        '''
        results = extract_all(html)
        urls = [r.url for r in results]
        assert "https://a.com/1.zip" in urls
        assert "https://b.com/2.zip" in urls

    def test_custom_matchers(self):
        html = '''
        <script>var url1 = "https://a.com/1.zip";</script>
        <div data-real-url="https://b.com/2.zip"></div>
        '''
        results = extract_all(html, matchers=[JsVariableMatcher()])
        urls = [r.url for r in results]
        assert "https://a.com/1.zip" in urls
        assert "https://b.com/2.zip" not in urls

    def test_all_layers_combined(self):
        html = '''
        <script>var realUrl = "https://real.com/model.bin";</script>
        <div data-real-url="https://actual.com/data.tar"></div>
        <script>
        window.__INITIAL_STATE__ = {"downloadUrl": "https://state.com/file.zip"};
        </script>
        '''
        results = extract_all(html)
        urls = [r.url for r in results]
        assert "https://real.com/model.bin" in urls
        assert "https://actual.com/data.tar" in urls
        assert "https://state.com/file.zip" in urls


# ---------------------------------------------------------------------------
# OEmbedMatcher
# ---------------------------------------------------------------------------

class TestOEmbedMatcher:
    def setup_method(self):
        self.m = OEmbedMatcher()

    def test_name(self):
        assert self.m.name == "oembed"

    def test_link_rel_alternate(self):
        html = '<link rel="alternate" type="application/json+oembed" href="https://example.com/oembed?url=https://example.com/page">'
        results = self.m.extract(html)
        assert len(results) == 1
        assert results[0].url == "https://example.com/oembed?url=https://example.com/page"
        assert results[0].confidence == 0.7

    def test_type_before_rel(self):
        html = '<link type="application/json+oembed" rel="alternate" href="https://example.com/oembed?url=https://example.com/page">'
        results = self.m.extract(html)
        assert len(results) == 1
        assert results[0].url == "https://example.com/oembed?url=https://example.com/page"

    def test_no_oembed_returns_empty(self):
        html = '<link rel="stylesheet" href="style.css">'
        assert self.m.extract(html) == []

    def test_multiple_oembed_ignored(self):
        html = '''
        <link rel="alternate" type="application/json+oembed" href="https://a.com/oembed?url=1">
        <link rel="alternate" type="application/json+oembed" href="https://b.com/oembed?url=2">
        '''
        results = self.m.extract(html)
        assert len(results) == 2


# ---------------------------------------------------------------------------
# EmbeddedPlayerMatcher
# ---------------------------------------------------------------------------

class TestEmbeddedPlayerMatcher:
    def setup_method(self):
        self.m = EmbeddedPlayerMatcher()

    def test_name(self):
        assert self.m.name == "embedded_player"

    def test_youtube_embed(self):
        html = '<iframe width="560" height="315" src="https://www.youtube.com/embed/dQw4w9WgXcQ"></iframe>'
        results = self.m.extract(html)
        assert len(results) == 1
        assert results[0].url == "https://www.youtube.com/embed/dQw4w9WgXcQ"

    def test_vimeo_embed(self):
        html = '<iframe src="https://player.vimeo.com/video/123456789"></iframe>'
        results = self.m.extract(html)
        assert len(results) == 1
        assert results[0].url == "https://player.vimeo.com/video/123456789"

    def test_data_src(self):
        html = '<iframe data-src="https://example.com/embed/player"></iframe>'
        results = self.m.extract(html)
        assert len(results) == 1
        assert results[0].url == "https://example.com/embed/player"

    def test_relative_url(self):
        html = '<iframe src="/embed/video/123"></iframe>'
        results = self.m.extract(html)
        assert len(results) == 1
        assert results[0].url == "/embed/video/123"

    def test_ignores_javascript(self):
        html = '<iframe src="javascript:void(0)"></iframe>'
        assert self.m.extract(html) == []

    def test_ignores_hash(self):
        html = '<iframe src="#modal"></iframe>'
        assert self.m.extract(html) == []

    def test_no_iframe_returns_empty(self):
        html = '<p>No iframes here</p>'
        assert self.m.extract(html) == []

    def test_multiple_iframes(self):
        html = '''
        <iframe src="https://a.com/embed/1"></iframe>
        <iframe src="https://b.com/embed/2"></iframe>
        '''
        results = self.m.extract(html)
        assert len(results) == 2
        urls = [r.url for r in results]
        assert "https://a.com/embed/1" in urls
        assert "https://b.com/embed/2" in urls
