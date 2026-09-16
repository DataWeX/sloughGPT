"""
downcraft.resolve — Extract real download URLs from ad-heavy pages.

Provides pattern matching, HTML scraping, and third-party host support
to find actual download links hidden behind ads, popups, JS redirects,
and obfuscation.
"""

from .patterns import Extraction, extract_all
from .scraper import ResolvedLink, resolve_and_download, resolve_page

__all__ = [
    "ResolvedLink",
    "resolve_page",
    "resolve_and_download",
    "Extraction",
    "extract_all",
]
