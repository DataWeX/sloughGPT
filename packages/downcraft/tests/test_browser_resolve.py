"""Tests for downcraft.resolve.browser — headless Chromium resolver."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))

from downcraft.resolve.browser import (
    BrowserResolver,
    _extract_links_from_html,
    _score_captured_urls,
)
from downcraft.resolve.scraper import ResolvedLink

# ---------------------------------------------------------------------------
# _score_captured_urls
# ---------------------------------------------------------------------------


class TestScoreCapturedUrls:
    def test_rar_file_high_score(self):
        captured = [
            {
                "url": "https://example.com/part01.rar",
                "content_type": "application/x-rar",
                "content_length": 50_000_000,
                "status": 200,
            }
        ]
        links = _score_captured_urls(captured, "https://example.com", "")
        assert len(links) == 1
        assert links[0].confidence >= 0.7
        assert links[0].extension == ".rar"
        assert links[0].source == "browser_network"

    def test_html_page_low_score(self):
        captured = [
            {
                "url": "https://example.com/page.html",
                "content_type": "text/html",
                "content_length": 1024,
                "status": 200,
            }
        ]
        links = _score_captured_urls(captured, "https://example.com", "")
        assert len(links) == 0  # HTML filtered out

    def test_large_file_bonus(self):
        captured = [
            {
                "url": "https://example.com/big.zip",
                "content_type": "application/zip",
                "content_length": 200_000_000,
                "status": 200,
            }
        ]
        links = _score_captured_urls(captured, "https://example.com", "")
        assert len(links) == 1
        assert links[0].confidence >= 0.7  # base + ext + large size

    def test_tracking_url_penalized(self):
        captured = [
            {
                "url": "https://ads.tracking.example.com/pixel.gif",
                "content_type": "image/gif",
                "content_length": 1024,
                "status": 200,
            }
        ]
        links = _score_captured_urls(captured, "https://example.com", "")
        assert len(links) == 1
        assert links[0].confidence < 0.4

    def test_empty_captured(self):
        links = _score_captured_urls([], "https://example.com", "")
        assert links == []


# ---------------------------------------------------------------------------
# _extract_links_from_html
# ---------------------------------------------------------------------------


class TestExtractLinksFromHtml:
    def test_finds_rar_links(self):
        html = '<a href="https://example.com/part01.rar">Download Part 1</a>'
        links = _extract_links_from_html(html, "https://example.com")
        assert len(links) >= 1
        rar_links = [l for l in links if l.extension == ".rar"]
        assert len(rar_links) == 1
        assert rar_links[0].confidence >= 0.5

    def test_skips_html_pages(self):
        html = '<a href="https://example.com/page.html">Some page</a>'
        links = _extract_links_from_html(html, "https://example.com")
        assert len(links) == 0

    def test_skips_javascript_href(self):
        html = '<a href="javascript:void(0)">Click</a>'
        links = _extract_links_from_html(html, "https://example.com")
        assert len(links) == 0

    def test_resolves_relative_urls(self):
        html = '<a href="/files/part01.rar">Download</a>'
        links = _extract_links_from_html(html, "https://example.com/page")
        assert len(links) == 1
        assert links[0].url == "https://example.com/files/part01.rar"

    def test_download_context_bonus(self):
        html = '<a href="https://example.com/file.zip">Download Click here</a>'
        links = _extract_links_from_html(html, "https://example.com")
        assert len(links) == 1
        assert links[0].confidence >= 0.6  # ext + download context


# ---------------------------------------------------------------------------
# BrowserResolver
# ---------------------------------------------------------------------------


class TestBrowserResolver:
    def test_returns_empty_when_playwright_missing(self):
        """BrowserResolver returns empty list when playwright not installed."""
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "playwright.sync_api":
                raise ImportError("No module named 'playwright'")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=mock_import):
            resolver = BrowserResolver()
            links = resolver.resolve("https://example.com")
            assert links == []

    def test_init_defaults(self):
        resolver = BrowserResolver()
        assert resolver._headless is True
        assert resolver._timeout_ms == 30_000
        assert resolver._wait_after_load_ms == 3000

    def test_init_custom(self):
        resolver = BrowserResolver(
            headless=False,
            timeout_ms=60_000,
            wait_after_load_ms=5000,
            user_agent="CustomBot/1.0",
        )
        assert resolver._headless is False
        assert resolver._timeout_ms == 60_000
        assert resolver._wait_after_load_ms == 5000
        assert resolver._user_agent == "CustomBot/1.0"


# ---------------------------------------------------------------------------
# resolve_page_browser (integration with scraper)
# ---------------------------------------------------------------------------


class TestResolvePageBrowser:
    def test_returns_empty_when_playwright_missing(self):
        """resolve_page_browser returns empty when playwright missing."""
        import builtins

        from downcraft.resolve.scraper import resolve_page_browser

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "playwright.sync_api":
                raise ImportError("No module named 'playwright'")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=mock_import):
            links = resolve_page_browser("https://example.com")
            assert links == []

    def test_min_confidence_filter(self):
        """resolve_page_browser filters by min_confidence."""
        from downcraft.resolve.scraper import resolve_page_browser

        mock_links = [
            ResolvedLink(url="https://a.com/f.rar", confidence=0.8, source="test"),
            ResolvedLink(url="https://a.com/g.zip", confidence=0.2, source="test"),
        ]

        with patch("downcraft.resolve.browser.BrowserResolver.resolve", return_value=mock_links):
            links = resolve_page_browser("https://example.com", min_confidence=0.5)
            assert len(links) == 1
            assert links[0].url == "https://a.com/f.rar"


# ---------------------------------------------------------------------------
# resolve_with_browser_fallback
# ---------------------------------------------------------------------------


class TestResolveWithBrowserFallback:
    def test_returns_http_when_good(self):
        """Returns HTTP results when confidence is high enough."""
        from downcraft.resolve.scraper import resolve_with_browser_fallback

        good_link = ResolvedLink(
            url="https://example.com/file.zip",
            confidence=0.8,
            extension=".zip",
            source="http",
        )

        with patch("downcraft.resolve.scraper.resolve_page", return_value=[good_link]):
            links = resolve_with_browser_fallback("https://example.com")
            assert len(links) == 1
            assert links[0].url == "https://example.com/file.zip"

    def test_falls_back_to_browser_when_low_confidence(self):
        """Falls back to browser when HTTP confidence is too low."""
        from downcraft.resolve.scraper import resolve_with_browser_fallback

        bad_link = ResolvedLink(
            url="https://example.com/page",
            confidence=0.1,
            extension="",
            source="http",
        )
        browser_link = ResolvedLink(
            url="https://cdn.example.com/file.rar",
            confidence=0.9,
            extension=".rar",
            source="browser",
        )

        with (
            patch("downcraft.resolve.scraper.resolve_page", return_value=[bad_link]),
            patch("downcraft.resolve.scraper.resolve_page_browser", return_value=[browser_link]),
        ):
            links = resolve_with_browser_fallback("https://example.com")
            assert len(links) == 1
            assert links[0].url == "https://cdn.example.com/file.rar"

    def test_merges_http_and_browser_results(self):
        """Merges browser and HTTP results without duplicates."""
        from downcraft.resolve.scraper import resolve_with_browser_fallback

        http_link = ResolvedLink(
            url="https://example.com/page",
            confidence=0.4,
            extension="",
            source="http",
        )
        browser_link = ResolvedLink(
            url="https://cdn.example.com/file.rar",
            confidence=0.9,
            extension=".rar",
            source="browser",
        )

        with (
            patch("downcraft.resolve.scraper.resolve_page", return_value=[http_link]),
            patch("downcraft.resolve.scraper.resolve_page_browser", return_value=[browser_link]),
        ):
            links = resolve_with_browser_fallback("https://example.com", min_confidence=0.3)
            assert len(links) == 2
            # Browser link should be first (higher confidence)
            assert links[0].source == "browser"
            assert links[1].source == "http"
