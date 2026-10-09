"""
downcraft.resolve — Extract real download URLs from ad-heavy pages.

Provides pattern matching, HTML scraping, and third-party host support
to find actual download links hidden behind ads, popups, JS redirects,
and obfuscation.

Submodules:
    - ``patterns``: regex-based URL extraction from HTML/JS
    - ``scraper``: HTTP page fetcher with scoring and deduplication
    - ``browser``: headless Chromium resolver for JS-heavy pages
"""

from .patterns import Extraction, extract_all
from .scraper import (
    ResolvedLink,
    resolve_and_download,
    resolve_page,
    resolve_page_browser,
    resolve_with_browser_fallback,
)

__all__ = [
    "ResolvedLink",
    "resolve_page",
    "resolve_and_download",
    "resolve_page_browser",
    "resolve_with_browser_fallback",
    "Extraction",
    "extract_all",
]
