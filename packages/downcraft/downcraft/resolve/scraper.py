"""
Link resolver — extract real download URLs from ad-heavy pages.

Given a web page URL, fetches the HTML and applies heuristics to find
the actual download link hidden behind ads, popups, JS redirects,
meta refreshes, base64 obfuscation, and other obfuscation.

No headless browser needed — pure HTTP + regex + HTML parsing.
"""

import base64
import json
import logging
import os
import re
import urllib.parse
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable, Dict, FrozenSet, List, Optional, Set, Tuple

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DOWNLOAD_EXTENSIONS: Set[str] = {
    ".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2",
    ".tar.xz", ".txz", ".7z", ".rar", ".deb", ".rpm",
    ".exe", ".msi", ".dmg", ".pkg", ".app", ".apk",
    ".iso", ".img", ".pdf", ".epub",
    ".whl", ".gem", ".bin",
    ".safetensors", ".gguf", ".ggml", ".npz", ".onnx",
    ".pt", ".pth", ".ckpt",
}

_HTML_EXTENSIONS: FrozenSet[str] = frozenset({
    ".html", ".htm", ".php", ".asp", ".aspx", ".jsp",
})

DOWNLOAD_SIGNALS: Set[str] = {
    "download", "dl", "fetch", "get", "save", "grab", "acquire",
    "install", "setup", "release", "latest", "stable",
}

AD_SIGNALS: Set[str] = {
    "ad", "ads", "advert", "sponsor", "promo", "tracking",
    "click", "redirect", "ref", "affiliate", "camp",
    "popup", "interstitial", "survey", "captcha", "verify",
    "human", "bot", "security", "cloudflare", "challenge",
    "short", "bit.ly", "tinyurl", "t.co", "goo.gl",
    "analytics", "pixel", "beacon", "track",
}

_AD_RES: Dict[str, re.Pattern] = {
    w: re.compile(r"\b" + re.escape(w) + r"\b", re.IGNORECASE)
    for w in AD_SIGNALS
}

_DOWNLOAD_RES: Dict[str, re.Pattern] = {
    w: re.compile(r"\b" + re.escape(w) + r"\b", re.IGNORECASE)
    for w in DOWNLOAD_SIGNALS
}

_JSON_LD_URL_KEYS: Set[str] = {
    "downloadUrl", "contentUrl", "url", "sameAs",
    "installUrl", "fileUrl", "actionUrl",
}

_DOWNLOAD_CLASS_TOKENS: Tuple[str, ...] = ("download", "dl", "btn-download")
_AD_CLASS_TOKENS: Tuple[str, ...] = ("ad", "sponsor", "promo", "popup")

_DATA_URL_ATTRS: Tuple[str, ...] = ("data-href", "data-url", "data-download", "data-link")

_DATA_ATTR_TAGS: Set[str] = {"a", "button", "div", "span"}

_OBFUSCATED_DATA_ATTRS: Tuple[str, ...] = (
    "data-download-url", "data-real-url", "data-file", "data-link-url",
    "data-countdown-url", "data-timer-url", "data-final-url",
    "data-popunder", "data-pop", "data-href-real",
    "data-action-url", "data-redirect", "data-target",
)

_BINARY_CONTENT_TYPES: Set[str] = {
    "application/octet-stream", "application/zip", "application/x-tar",
    "application/gzip", "application/pdf", "application/x-safetensors",
    "application/macbinary", "application/x-bittorrent",
}

_SKIP_TAGS: FrozenSet[str] = frozenset({
    "header", "nav", "aside", "footer", "script", "style",
})

_SKIP_TAG_START_RES: Dict[str, re.Pattern] = {
    t: re.compile(rf"<{t}[\s>]", re.IGNORECASE) for t in _SKIP_TAGS
}
_SKIP_TAG_END_RES: Dict[str, re.Pattern] = {
    t: re.compile(rf"</{t}>", re.IGNORECASE) for t in _SKIP_TAGS
}
_MAIN_START_RE = re.compile(r"<main[\s>]", re.IGNORECASE)
_MAIN_END_RE = re.compile(r"</main>", re.IGNORECASE)

_HEX_DATA_ATTR_RE = re.compile(
    r"(?:data-[\w-]+)\s*=\s*\"((?:\\x[0-9a-fA-F]{2}){8,})\""
)


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class ResolvedLink:
    """A resolved download link with metadata."""
    url: str
    title: str = ""
    extension: str = ""
    size_hint: int = 0
    confidence: float = 0.0
    source: str = ""
    redirects: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# URL utilities
# ---------------------------------------------------------------------------

def _get_extension(url: str) -> str:
    """Extract file extension from URL, handling compound extensions."""
    parsed = urllib.parse.urlparse(url)
    path = parsed.path.lower()
    for compound in (".tar.gz", ".tar.bz2", ".tar.xz"):
        if path.endswith(compound):
            return compound
    _, ext = os.path.splitext(path)
    return ext


def _is_same_domain(url: str, page_url: str) -> bool:
    """Check if two URLs are on the same domain."""
    try:
        u = urllib.parse.urlparse(url)
        p = urllib.parse.urlparse(page_url)
        return u.netloc == p.netloc or u.netloc.endswith("." + p.netloc)
    except Exception:
        return False


def _resolve_relative(url: str, base: str) -> str:
    """Resolve a relative URL against a base URL."""
    return urllib.parse.urljoin(base, url)


# ---------------------------------------------------------------------------
# HTML link extractor
# ---------------------------------------------------------------------------

class _LinkExtractor(HTMLParser):
    """Extract <a href> links and data-* URLs from HTML."""

    def __init__(self) -> None:
        super().__init__()
        self.links: List[Tuple[str, str, Dict[str, str]]] = []
        self._current_href: Optional[str] = None
        self._current_text: List[str] = []
        self._current_attrs: Dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        attr_dict = dict(attrs)
        if tag == "a":
            self._current_href = attr_dict.get("href", "")
            self._current_text = []
            self._current_attrs = attr_dict
        if tag in _DATA_ATTR_TAGS:
            for key, val in attr_dict.items():
                if key.startswith("data-") and val:
                    self.links.append((val, f"[{key}]", attr_dict))

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._current_href:
            text = " ".join("".join(self._current_text).split())
            self.links.append((self._current_href, text, self._current_attrs))
        self._current_href = None
        self._current_text = []
        self._current_attrs = {}

    def handle_data(self, data: str) -> None:
        if self._current_href is not None:
            self._current_text.append(data)


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def _extract_js_redirects(html: str) -> List[str]:
    """Find URLs hidden in JavaScript redirects and meta refresh tags."""
    urls: List[str] = []
    patterns = (
        r'window\.location(?:\.href)?\s*=\s*["\']([^"\']+)["\']',
        r'window\.location\.assign\s*\(\s*["\']([^"\']+)["\']',
        r'window\.location\.replace\s*\(\s*["\']([^"\']+)["\']',
        r'document\.location(?:\.href)?\s*=\s*["\']([^"\']+)["\']',
        r'setTimeout\s*\(\s*["\'](?:location(?:\.href)?\s*=\s*)?["\']?([^"\']+)["\']',
        r'<meta[^>]+http-equiv\s*=\s*["\']refresh["\'][^>]+content\s*=\s*["\'][^"\']*url=([^"\']+)["\']',
        r'<meta[^>]+content\s*=\s*["\'][^"\']*url=([^"\']+)["\'][^>]+http-equiv\s*=\s*["\']refresh["\']',
        r'window\.open\s*\(\s*["\']([^"\']+)["\']',
        r'onclick\s*=\s*["\'][^"\']*(?:location(?:\.href)?|window\.open)\s*\(\s*["\']([^"\']+)["\']',
    )
    for pat in patterns:
        for match in re.finditer(pat, html, re.IGNORECASE):
            url = match.group(1).strip()
            if url and not url.startswith(("javascript:", "#", "void")):
                urls.append(url)
    return urls


def _extract_meta_urls(html: str) -> List[str]:
    """Find URLs in meta tags (og:url, twitter:url, canonical)."""
    urls: List[str] = []
    patterns = (
        r'<meta[^>]+property\s*=\s*["\']og:url["\'][^>]+content\s*=\s*["\']([^"\']+)["\']',
        r'<meta[^>]+content\s*=\s*["\']([^"\']+)["\'][^>]+property\s*=\s*["\']og:url["\']',
        r'<meta[^>]+name\s*=\s*["\']twitter:url["\'][^>]+content\s*=\s*["\']([^"\']+)["\']',
        r'<link[^>]+rel\s*=\s*["\']canonical["\'][^>]+href\s*=\s*["\']([^"\']+)["\']',
    )
    for pat in patterns:
        for match in re.finditer(pat, html, re.IGNORECASE):
            urls.append(match.group(1).strip())
    return urls


def _decode_obfuscated_urls(html: str) -> List[str]:
    """Extract URLs hidden via base64, hex, or other obfuscation.

    Handles:
      - ``atob("base64string")``
      - ``decodeURIComponent("encoded")``
      - ``data-download-url="..."`` (raw or base64)
      - ``String.fromCharCode(72, 116, ...)``
      - ``data-*="\\x68\\x74\\x74\\x70..."`` (hex-escaped)
    """
    urls: List[str] = []

    for m in re.finditer(
        r'atob\s*\(\s*["\']([A-Za-z0-9+/=_-]{20,})["\']', html
    ):
        try:
            decoded = base64.b64decode(m.group(1)).decode("utf-8", errors="ignore")
            if "://" in decoded or decoded.startswith("/"):
                urls.append(decoded.strip())
        except Exception:
            pass

    for m in re.finditer(
        r'decodeURIComponent\s*\(\s*["\']([^"\']{10,})["\']', html
    ):
        try:
            decoded = urllib.parse.unquote(m.group(1))
            if "://" in decoded or decoded.startswith("/"):
                urls.append(decoded.strip())
        except Exception:
            pass

    for attr in _OBFUSCATED_DATA_ATTRS:
        for m in re.finditer(
            rf'{attr}\s*=\s*["\']([^"\']+)["\']', html, re.IGNORECASE
        ):
            val = m.group(1).strip()
            try:
                decoded = base64.b64decode(val).decode("utf-8", errors="ignore")
                if "://" in decoded or decoded.startswith("/"):
                    urls.append(decoded.strip())
                    continue
            except Exception:
                pass
            if "://" in val or val.startswith("/"):
                urls.append(val)

    for m in re.finditer(
        r'String\.fromCharCode\s*\(\s*([\d,\s]+)\s*\)', html
    ):
        try:
            codes = [int(c.strip()) for c in m.group(1).split(",") if c.strip()]
            decoded = "".join(chr(c) for c in codes)
            if "://" in decoded or decoded.startswith("/"):
                urls.append(decoded.strip())
        except Exception:
            pass

    for m in _HEX_DATA_ATTR_RE.finditer(html):
        try:
            decoded = m.group(1).encode("ascii").decode("unicode_escape")
            if "://" in decoded:
                urls.append(decoded.strip())
        except Exception:
            pass

    return urls


def _extract_js_variable_urls(html: str) -> List[str]:
    """Extract URLs from JS variable assignments.

    Catches countdown/download page patterns:
      - ``var downloadUrl = "https://..."``
      - ``let realUrl = 'https://...'``
      - ``window.finalUrl = "https://..."``
    """
    urls: List[str] = []
    for pat in (
        r'(?:var|let|const)\s+\w*(?:url|link|download|href|file|src|target)\w*\s*=\s*["\']([^"\']+)["\']',
        r'window\.\w*(?:url|link|download|href|file|src|target)\w*\s*=\s*["\']([^"\']+)["\']',
    ):
        for m in re.finditer(pat, html, re.IGNORECASE):
            url = m.group(1).strip()
            if url and not url.startswith(("javascript:", "#", "void")):
                urls.append(url)
    return urls


def _extract_json_blob_urls(html: str) -> List[str]:
    """Extract URLs from embedded JSON blobs (framework state, config)."""
    urls: List[str] = []
    for pat in (
        r'window\.__(?:INITIAL_STATE|NUXT|NEXT_DATA|APP_DATA)__\s*=\s*(\{.+?\});',
        r'(?:var|let|const)\s+config\s*=\s*(\{.+?\});',
    ):
        for m in re.finditer(pat, html, re.IGNORECASE | re.DOTALL):
            try:
                data = json.loads(m.group(1))
                _collect_urls_from_dict(data, urls)
            except (json.JSONDecodeError, ValueError):
                pass
    return urls


def _extract_json_ld_urls(html: str) -> List[str]:
    """Extract download URLs from JSON-LD structured data."""
    urls: List[str] = []
    for m in re.finditer(
        r'<script[^>]+type\s*=\s*["\']application/ld\+json["\'][^>]*>(.*?)</script>',
        html, re.IGNORECASE | re.DOTALL,
    ):
        try:
            data = json.loads(m.group(1))
            items = data if isinstance(data, list) else [data]
            for item in items:
                if isinstance(item, dict):
                    _collect_urls_from_dict(item, urls)
        except (json.JSONDecodeError, ValueError):
            pass
    return urls


def _collect_urls_from_dict(d: dict, out: List[str], depth: int = 0) -> None:
    """Recursively collect URL values from a JSON-LD dict."""
    if depth > 5:
        return
    for key, val in d.items():
        if isinstance(val, str) and key in _JSON_LD_URL_KEYS:
            if "://" in val or val.startswith("/"):
                out.append(val.strip())
        elif isinstance(val, dict):
            _collect_urls_from_dict(val, out, depth + 1)
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, dict):
                    _collect_urls_from_dict(item, out, depth + 1)
                elif isinstance(item, str) and "://" in item:
                    out.append(item.strip())


# ---------------------------------------------------------------------------
# Position-aware content detection
# ---------------------------------------------------------------------------

def _find_main_content(html: str) -> Tuple[int, int]:
    """Find the byte offset range of the main content area.

    Prefers an explicit ``<main>`` tag; otherwise computes exclusion zones
    from header/nav/aside/footer/script/style tags and returns the largest
    content gap between them.
    """
    ms = _MAIN_START_RE.search(html)
    me = _MAIN_END_RE.search(html)
    if ms and me:
        return ms.end(), me.start()

    html_len = len(html)

    skip_regions: List[Tuple[int, int]] = []
    for tag in _SKIP_TAGS:
        for m in _SKIP_TAG_START_RES[tag].finditer(html):
            end_m = _SKIP_TAG_END_RES[tag].search(html[m.end():])
            if end_m:
                skip_regions.append((m.start(), m.end() + end_m.end()))

    if not skip_regions:
        return 0, html_len

    skip_regions.sort()
    merged: List[Tuple[int, int]] = [skip_regions[0]]
    for start, end in skip_regions[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    gaps: List[Tuple[int, int]] = []
    prev_end = 0
    for start, end in merged:
        if start > prev_end:
            gaps.append((prev_end, start))
        prev_end = max(prev_end, end)
    if prev_end < html_len:
        gaps.append((prev_end, html_len))

    if not gaps:
        return 0, 0

    best = max(gaps, key=lambda g: g[1] - g[0])
    return best


def _is_in_main_content(
    href: str, html: str, main_range: Optional[Tuple[int, int]] = None,
) -> bool:
    """Check if *href* appears in the main content area.

    Searches for both the full URL and the path portion, since the
    candidate href is already resolved to absolute but the original HTML
    typically contains relative paths (e.g. ``/file.zip``).
    """
    start, end = main_range if main_range is not None else _find_main_content(html)
    if len(href) > 1:
        idx = html.find(href, start)
        if idx != -1 and idx <= end:
            return True
    parsed = urllib.parse.urlparse(href)
    if parsed.path and len(parsed.path) > 1:
        idx = html.find(parsed.path, start)
        if idx != -1 and idx <= end:
            return True
    return False


# ---------------------------------------------------------------------------
# HTML page filter
# ---------------------------------------------------------------------------

def _is_html_link(link: ResolvedLink) -> bool:
    """Return True if a link points to an HTML page, not a downloadable file."""
    return not link.extension or link.extension in _HTML_EXTENSIONS


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_link(
    href: str,
    text: str,
    attrs: Dict[str, str],
    page_url: str,
    *,
    html: str = "",
    source: str = "",
    main_range: Optional[Tuple[int, int]] = None,
) -> float:
    """Score a link 0.0–1.0 for download likelihood."""
    score = 0.0
    href_lower = href.lower()
    text_lower = text.lower()

    ext = _get_extension(href)
    if ext in _HTML_EXTENSIONS:
        score -= 0.5
    elif ext in DOWNLOAD_EXTENSIONS:
        score += 0.4

    for pat in _DOWNLOAD_RES.values():
        if pat.search(text_lower):
            score += 0.15
            break
    for pat in _DOWNLOAD_RES.values():
        if pat.search(href_lower):
            score += 0.1
            break

    parsed_url = urllib.parse.urlparse(href_lower)
    url_core = f"{parsed_url.netloc}{parsed_url.path}"
    for pat in _AD_RES.values():
        if pat.search(url_core):
            score -= 0.4
    for pat in _AD_RES.values():
        if pat.search(text_lower):
            score -= 0.15

    for attr_key in ("class", "id", "rel"):
        val = attrs.get(attr_key, "").lower()
        if any(w in val for w in _DOWNLOAD_CLASS_TOKENS):
            score += 0.15
        if any(w in val for w in _AD_CLASS_TOKENS):
            score -= 0.3

    if any(attrs.get(key) for key in _DATA_URL_ATTRS):
        score += 0.1

    if source == "json_ld":
        score += 0.2
    elif source == "obfuscated":
        score += 0.15

    if _is_same_domain(href, page_url):
        score += 0.05

    if html and _is_in_main_content(href, html, main_range):
        score += 0.1

    if len(href) < 5:
        score -= 0.1
    if len(text) < 2:
        score -= 0.05

    if len(parsed_url.query) > 100:
        score -= 0.1

    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Content-Type verification
# ---------------------------------------------------------------------------

def _verify_content_type(url: str) -> int:
    """HEAD request to verify Content-Type.

    Returns 1 (binary), -1 (HTML), or 0 (ambiguous/failed).
    """
    try:
        resp = requests.head(
            url, timeout=5, allow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        ct = resp.headers.get("Content-Type", "").lower()
        cl = resp.headers.get("Content-Length", "0")
        if "text/html" in ct:
            return -1
        if any(bt in ct for bt in _BINARY_CONTENT_TYPES):
            return 1
        if cl.isdigit() and int(cl) > 1024 and "text/" not in ct:
            return 1
        return 0
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Candidate extraction
# ---------------------------------------------------------------------------

def _extract_candidates(
    html: str, final_url: str, max_links: int,
) -> List[Tuple[str, str, Dict[str, str], str]]:
    """Extract all link candidates from HTML.

    Returns list of (resolved_url, text, attrs, source_label) tuples.
    """
    from .patterns import extract_all

    candidates: List[Tuple[str, str, Dict[str, str], str]] = []

    parser = _LinkExtractor()
    try:
        parser.feed(html)
    except Exception:
        pass
    for href, text, attrs in parser.links[:max_links]:
        candidates.append((_resolve_relative(href, final_url), text, attrs, "html_link"))

    for ex in extract_all(html):
        candidates.append((_resolve_relative(ex.url, final_url), f"[{ex.source}]", {}, ex.source))

    return candidates


def _score_and_deduplicate(
    candidates: List[Tuple[str, str, Dict[str, str], str]],
    page_url: str, html: str, page_redirects: List[str],
) -> List[ResolvedLink]:
    """Score, deduplicate, and return sorted ResolvedLink list."""
    main_range = _find_main_content(html) if html else None
    seen: Set[str] = set()
    results: List[ResolvedLink] = []

    for href, text, attrs, source in candidates:
        if href.startswith(("mailto:", "javascript:", "#", "tel:")):
            continue
        if href in seen:
            continue
        seen.add(href)

        score = _score_link(
            href, text, attrs, page_url,
            html=html, source=source, main_range=main_range,
        )
        results.append(ResolvedLink(
            url=href, title=text, extension=_get_extension(href),
            confidence=round(score, 3), source=source,
            redirects=page_redirects,
        ))

    results.sort(key=lambda r: r.confidence, reverse=True)
    return results


# ---------------------------------------------------------------------------
# Intermediate page follow
# ---------------------------------------------------------------------------

def _follow_intermediate(
    url: str,
    session: requests.Session,
    headers: Dict[str, str],
    timeout: int,
    original_page_url: str,
    depth: int,
    on_progress: Optional[Callable[[str], None]] = None,
) -> List[ResolvedLink]:
    """Follow an intermediate redirect page to find the real download."""
    if depth <= 0:
        return []

    if on_progress:
        on_progress(f"Following intermediate page: {url}")

    try:
        resp = session.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        resp.raise_for_status()
    except requests.RequestException:
        return []

    html = resp.text
    final_url = resp.url

    ct = resp.headers.get("Content-Type", "").lower()
    if "text/html" not in ct and "text/plain" not in ct:
        return [ResolvedLink(
            url=final_url, title="[direct file]",
            extension=_get_extension(final_url),
            confidence=0.6, source="intermediate_direct",
        )]

    candidates = _extract_candidates(html, final_url, max_links=50)
    candidates = [(h, t, a, f"intermediate_{s}") for h, t, a, s in candidates]
    results = _score_and_deduplicate(candidates, original_page_url, html, [])

    if results and results[0].confidence < 0.4 and not results[0].extension:
        deeper = _follow_intermediate(
            results[0].url, session, headers, timeout,
            original_page_url, depth - 1, on_progress,
        )
        if deeper:
            existing_urls = {r.url for r in results}
            for dl in deeper:
                if dl.url not in existing_urls:
                    results.append(dl)
            results.sort(key=lambda r: r.confidence, reverse=True)

    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resolve_page(
    page_url: str,
    *,
    session: Optional[requests.Session] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 15,
    max_links: int = 100,
    max_depth: int = 2,
    verify_content_type: bool = False,
    on_progress: Optional[Callable[[str], None]] = None,
) -> List[ResolvedLink]:
    """Fetch a page and extract ranked download links.

    Args:
        page_url: URL of the page to scrape.
        session: Optional requests session for connection reuse.
        headers: Extra HTTP headers.
        timeout: Request timeout in seconds.
        max_links: Max links to consider per page.
        max_depth: Max pages to follow for intermediate redirects.
        verify_content_type: If True, HEAD-request the top candidate.
        on_progress: Status callback.

    Returns:
        List of ResolvedLink sorted by confidence (best first).
    """
    sess = session or requests.Session()
    hdrs = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }
    if headers:
        hdrs.update(headers)

    if on_progress:
        on_progress(f"Fetching {page_url}")

    try:
        resp = sess.get(page_url, headers=hdrs, timeout=timeout, allow_redirects=True)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.warning("Failed to fetch %s: %s", page_url, e)
        return []

    final_url = resp.url
    html = resp.text
    page_redirects = [r.url for r in resp.history] if hasattr(resp, "history") else []

    if on_progress:
        on_progress(f"Analyzing page ({len(html)} bytes)")

    candidates = _extract_candidates(html, final_url, max_links)
    results = _score_and_deduplicate(candidates, page_url, html, page_redirects)

    if max_depth > 0 and results:
        top = results[0]
        if top.confidence < 0.5 and not top.extension:
            followed = _follow_intermediate(
                top.url, sess, hdrs, timeout, page_url, max_depth, on_progress,
            )
            if followed:
                existing_urls = {r.url for r in results}
                for fl in followed:
                    if fl.url not in existing_urls:
                        results.append(fl)
                        existing_urls.add(fl.url)
                results.sort(key=lambda r: r.confidence, reverse=True)

    results = [r for r in results if not _is_html_link(r)]

    if on_progress:
        top = results[0] if results else None
        if top:
            on_progress(f"Best candidate: {top.url} (confidence={top.confidence})")
        else:
            on_progress("No download links found")

    if verify_content_type and results:
        best = results[0]
        if on_progress:
            on_progress(f"Verifying Content-Type of {best.url}")
        ct_result = _verify_content_type(best.url)
        if ct_result == -1:
            best.confidence = max(0.0, best.confidence - 0.3)
            best.source = "verified_html"
            if on_progress:
                on_progress("Top candidate returns HTML — confidence reduced")
        elif ct_result == 1:
            best.confidence = min(1.0, best.confidence + 0.1)
            best.source = "verified_binary"
            if on_progress:
                on_progress("Top candidate confirmed as binary file")
        results.sort(key=lambda r: r.confidence, reverse=True)

    return results


def resolve_and_download(
    page_url: str,
    dest: str,
    *,
    session: Optional[requests.Session] = None,
    headers: Optional[Dict[str, str]] = None,
    min_confidence: float = 0.3,
    max_depth: int = 2,
    on_progress: Optional[Callable[[str], None]] = None,
) -> Path:
    """Resolve a page, then download the best candidate.

    Args:
        page_url: URL of the page to scrape.
        dest: Local file path to save to.
        session: Optional requests session.
        headers: Extra HTTP headers.
        min_confidence: Minimum confidence to accept a link.
        max_depth: Max intermediate pages to follow.
        on_progress: Status callback.

    Returns:
        Path to downloaded file.

    Raises:
        ValueError: If no link meets the confidence threshold.
        downloader.DownloadError: If download fails.
    """
    from downcraft.download import http as downloader

    links = resolve_page(
        page_url, session=session, headers=headers,
        max_depth=max_depth, on_progress=on_progress,
    )
    if not links:
        raise ValueError(f"No download links found on {page_url}")

    best = links[0]
    if best.confidence < min_confidence:
        raise ValueError(
            f"Best link confidence too low: {best.confidence:.2f} < {min_confidence} "
            f"(url={best.url}, title={best.title!r})"
        )

    if on_progress:
        on_progress(f"Downloading {best.url} (confidence={best.confidence})")

    return downloader.download_file(
        best.url,
        Path(dest),
        on_chunk=lambda done, total: on_progress(
            f"Downloaded {done}/{total} bytes"
        ) if on_progress else None,
    )
