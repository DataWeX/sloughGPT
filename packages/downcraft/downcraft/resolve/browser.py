"""
Browser-based URL resolver using Playwright.

Falls back to headless Chromium when the pure-HTTP scraper can't find
real download links (JS-heavy pages, anti-bot redirects, CAPTCHAs).

Usage::

    from downcraft.resolve.browser import BrowserResolver

    resolver = BrowserResolver()
    links = resolver.resolve("https://sharemods.com/...")
    # Returns list[ResolvedLink] — same shape as resolve_page()

Requires: ``pip install downcraft[browser]`` (playwright>=1.40)
"""

from __future__ import annotations

import logging
import re
import urllib.parse
from collections.abc import Callable
from typing import Any

from .scraper import DOWNLOAD_EXTENSIONS, ResolvedLink, _get_extension

logger = logging.getLogger(__name__)

# File extensions we consider worth downloading
_DOWNLOAD_EXT_SET = DOWNLOAD_EXTENSIONS

# JS patterns that signal a download URL might appear soon
_DOWNLOAD_JS_PATS: list[re.Pattern] = [
    re.compile(r"\.rar|\.zip|\.7z|\.tar|\.gz|\.exe|\.dmg|\.iso", re.I),
    re.compile(r"download|direct|cdn", re.I),
]

# Extensions that indicate an intermediate page, not a file
_HTML_EXTENSIONS = {".html", ".htm", ".php", ".asp", ".aspx", ".jsp"}


class BrowserResolver:
    """Headless Chromium resolver for JS-heavy download pages.

    Launches a headless browser, navigates to the page, waits for JS to
    execute, and intercepts network requests to find the actual download URL.

    Args:
        headless: Run in headless mode (default True).
        timeout_ms: Navigation timeout in milliseconds (default 30000).
        wait_after_load_ms: Extra wait after page load for JS (default 3000).
        user_agent: Custom user agent string.
    """

    def __init__(
        self,
        headless: bool = True,
        timeout_ms: int = 30_000,
        wait_after_load_ms: int = 3000,
        user_agent: str | None = None,
    ):
        self._headless = headless
        self._timeout_ms = timeout_ms
        self._wait_after_load_ms = wait_after_load_ms
        self._user_agent = user_agent

    def resolve(
        self,
        url: str,
        *,
        on_progress: Callable[[str], None] | None = None,
    ) -> list[ResolvedLink]:
        """Resolve a page using headless Chromium.

        Args:
            url: Page URL to resolve.
            on_progress: Optional status callback.

        Returns:
            List of ResolvedLink sorted by confidence (best first).
            Empty list if browser unavailable or no links found.
        """
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            logger.warning("playwright not installed — run: pip install downcraft[browser]")
            return []

        if on_progress:
            on_progress(f"Launching browser for {url}")

        captured_urls: list[dict[str, Any]] = []

        try:
            with sync_playwright() as pw:
                browser = pw.chromium.launch(headless=self._headless)
                context = browser.new_context(
                    user_agent=self._user_agent
                    or (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/125.0.0.0 Safari/537.36"
                    ),
                )
                page = context.new_page()

                # Intercept network requests to capture download URLs
                page.on("response", lambda resp: _on_response(resp, captured_urls))

                if on_progress:
                    on_progress(f"Navigating to {url}")

                page.goto(url, wait_until="load", timeout=self._timeout_ms)

                # Extra wait for JS countdowns, dynamic content
                if self._wait_after_load_ms > 0:
                    page.wait_for_timeout(self._wait_after_load_ms)

                # Try clicking common download button patterns
                _try_click_download(page)

                # Wait a bit more after click for redirect
                page.wait_for_timeout(2000)

                # Extract links from final page state
                html = page.content()
                final_url = page.url

                if on_progress:
                    on_progress(f"Page loaded, analyzing {len(captured_urls)} network requests")

                links = _score_captured_urls(captured_urls, final_url, html)
                links.extend(_extract_links_from_html(html, final_url))

                # Deduplicate and sort
                seen: set[str] = set()
                unique: list[ResolvedLink] = []
                for link in links:
                    if link.url not in seen:
                        seen.add(link.url)
                        unique.append(link)
                unique.sort(key=lambda r: r.confidence, reverse=True)

                browser.close()

                if on_progress:
                    top = unique[0] if unique else None
                    if top:
                        on_progress(f"Best: {top.url} (confidence={top.confidence})")
                    else:
                        on_progress("No download links found via browser")

                return unique

        except Exception as exc:
            logger.warning("Browser resolve failed for %s: %s", url, exc)
            return []


def _on_response(response: Any, captured: list[dict[str, Any]]) -> None:
    """Capture network responses that look like file downloads."""
    try:
        url = response.url
        status = response.status
        content_type = response.headers.get("content-type", "")
        content_length = response.headers.get("content-length", "0")

        # Skip non-successful responses
        if status < 200 or status >= 400:
            return

        # Skip HTML pages
        if "text/html" in content_type:
            return

        # Check if URL looks like a file download
        parsed = urllib.parse.urlparse(url)
        path_lower = parsed.path.lower()

        is_file = any(path_lower.endswith(ext) for ext in _DOWNLOAD_EXT_SET)
        is_binary = any(
            ct in content_type
            for ct in (
                "application/octet-stream",
                "application/x-rar",
                "application/zip",
                "application/x-7z",
                "application/x-gzip",
                "application/x-tar",
            )
        )
        has_size = content_length.isdigit() and int(content_length) > 1024

        if is_file or is_binary or has_size:
            captured.append(
                {
                    "url": url,
                    "content_type": content_type,
                    "content_length": int(content_length) if content_length.isdigit() else 0,
                    "status": status,
                }
            )
    except Exception:
        pass


def _try_click_download(page: Any) -> None:
    """Try clicking common download button patterns."""
    selectors = [
        'a[href*="download"]',
        'button:has-text("Download")',
        'a:has-text("Download")',
        'a:has-text("Click here")',
        'a:has-text("Start Download")',
        '[class*="download"]',
        '[id*="download"]',
        ".btn-download",
        "#download-button",
    ]
    for sel in selectors:
        try:
            el = page.locator(sel).first
            if el.is_visible(timeout=1000):
                el.click(timeout=3000)
                logger.debug("Clicked: %s", sel)
                return
        except Exception:
            continue


def _score_captured_urls(
    captured: list[dict[str, Any]],
    page_url: str,
    html: str,
) -> list[ResolvedLink]:
    """Score captured network URLs as download candidates."""
    results: list[ResolvedLink] = []

    for entry in captured:
        url = entry["url"]
        size = entry.get("content_length", 0)

        ext = _get_extension(url)
        if ext in _HTML_EXTENSIONS:
            continue

        score = 0.5  # base score for any file-like response

        if ext in _DOWNLOAD_EXT_SET:
            score += 0.3
        if size > 1_000_000:  # > 1MB
            score += 0.1
        if size > 100_000_000:  # > 100MB
            score += 0.1

        # Penalize common CDN/tracking patterns
        parsed = urllib.parse.urlparse(url)
        if any(w in parsed.netloc.lower() for w in ("ads", "tracking", "analytics", "pixel")):
            score -= 0.3

        results.append(
            ResolvedLink(
                url=url,
                title="[browser-captured]",
                extension=ext,
                size_hint=size,
                confidence=round(max(0.0, min(1.0, score)), 3),
                source="browser_network",
            )
        )

    return results


def _extract_links_from_html(html: str, page_url: str) -> list[ResolvedLink]:
    """Extract download links from the page HTML after JS execution."""
    results: list[ResolvedLink] = []

    # Find all href links
    for m in re.finditer(r'href\s*=\s*["\']([^"\']+)["\']', html, re.I):
        href = m.group(1).strip()
        if not href or href.startswith(("javascript:", "#", "mailto:")):
            continue

        # Resolve relative URLs
        if not href.startswith(("http://", "https://")):
            href = urllib.parse.urljoin(page_url, href)

        ext = _get_extension(href)
        if ext in _HTML_EXTENSIONS:
            continue

        score = 0.3
        if ext in _DOWNLOAD_EXT_SET:
            score += 0.3

        # Check link text context
        idx = m.start()
        context = html[max(0, idx - 200) : idx].lower()
        if any(w in context for w in ("download", "click here", "direct link")):
            score += 0.15

        results.append(
            ResolvedLink(
                url=href,
                title="[browser-html]",
                extension=ext,
                confidence=round(max(0.0, min(1.0, score)), 3),
                source="browser_html",
            )
        )

    return results
