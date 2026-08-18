"""
Link resolver — extract real download URLs from ad-heavy pages.

Given a web page URL, fetches the HTML and applies heuristics to find
the actual download link hidden behind ads, popups, JS redirects,
meta refreshes, and other obfuscation.

No headless browser needed — pure HTTP + regex + HTML parsing.
"""

import logging
import re
import urllib.parse
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Callable, Dict, List, Optional, Set, Tuple

import requests

logger = logging.getLogger(__name__)

# File extensions that indicate a downloadable file
DOWNLOAD_EXTENSIONS = {
    # Archives
    ".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2",
    ".tar.xz", ".txz", ".7z", ".rar", ".deb", ".rpm",
    # Installers
    ".exe", ".msi", ".dmg", ".pkg", ".app", ".apk",
    # Images
    ".iso", ".img", ".bin",
    # Documents
    ".pdf", ".epub",
    # Code / data
    ".whl", ".tar.gz", ".gem",
    # Models / ML
    ".safetensors", ".gguf", ".ggml", ".npz", ".onnx",
    ".bin", ".pt", ".pth", ".ckpt",
}

# Words that indicate a download link (case-insensitive)
DOWNLOAD_SIGNALS = {
    "download", "dl", "fetch", "get", "save", "grab", "acquire",
    "install", "setup", "release", "latest", "stable",
}

# Words that indicate an ad / tracking / fake link (case-insensitive).
# Short words (<=3 chars) use word-boundary matching to avoid false positives
# like "ad" inside "download".
AD_SIGNALS = {
    "ad", "ads", "advert", "sponsor", "promo", "tracking",
    "click", "redirect", "ref", "affiliate", "camp",
    "popup", "interstitial", "survey", "captcha", "verify",
    "human", "bot", "security", "cloudflare", "challenge",
    "short", "bit.ly", "tinyurl", "t.co", "goo.gl",
    "analytics", "pixel", "beacon", "track",
}

# Ad signals that need word-boundary matching (too short for plain substring)
AD_SIGNALS_BOUNDED = {"ad", "ads", "ref", "bot"}


@dataclass
class ResolvedLink:
    """A resolved download link with metadata."""
    url: str
    title: str = ""
    extension: str = ""
    size_hint: int = 0  # 0 = unknown
    confidence: float = 0.0  # 0.0–1.0
    source: str = ""  # where it was found
    redirects: List[str] = field(default_factory=list)


class _LinkExtractor(HTMLParser):
    """Extract <a href> links and their text from HTML."""

    def __init__(self):
        super().__init__()
        self.links: List[Tuple[str, str, Dict[str, str]]] = []  # (href, text, attrs)
        self._current_href: Optional[str] = None
        self._current_text: List[str] = []
        self._current_attrs: Dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]):
        if tag == "a":
            attr_dict = dict(attrs)
            href = attr_dict.get("href", "")
            self._current_href = href
            self._current_text = []
            self._current_attrs = attr_dict
        # Also check data-* attributes on any tag
        if tag in ("a", "button", "div", "span"):
            attr_dict = dict(attrs)
            for key, val in attr_dict.items():
                if key.startswith("data-") and val:
                    self.links.append((val, f"[{key}]", attr_dict))

    def handle_endtag(self, tag: str):
        if tag == "a" and self._current_href:
            text = " ".join("".join(self._current_text).split())
            self.links.append((self._current_href, text, self._current_attrs))
        self._current_href = None
        self._current_text = []
        self._current_attrs = {}

    def handle_data(self, data: str):
        if self._current_href is not None:
            self._current_text.append(data)


def _extract_js_redirects(html: str) -> List[str]:
    """Find URLs hidden in JavaScript redirects."""
    urls = []
    patterns = [
        # window.location = "..."  /  window.location.href = "..."
        r'window\.location(?:\.href)?\s*=\s*["\']([^"\']+)["\']',
        # window.location.assign("...")
        r'window\.location\.assign\s*\(\s*["\']([^"\']+)["\']',
        # window.location.replace("...")
        r'window\.location\.replace\s*\(\s*["\']([^"\']+)["\']',
        # document.location = "..."
        r'document\.location(?:\.href)?\s*=\s*["\']([^"\']+)["\']',
        # setTimeout("location='...'")
        r'setTimeout\s*\(\s*["\'](?:location(?:\.href)?\s*=\s*)?["\']?([^"\']+)["\']',
        # meta refresh
        r'<meta[^>]+http-equiv\s*=\s*["\']refresh["\'][^>]+content\s*=\s*["\'][^"\']*url=([^"\']+)["\']',
        r'<meta[^>]+content\s*=\s*["\'][^"\']*url=([^"\']+)["\'][^>]+http-equiv\s*=\s*["\']refresh["\']',
        # window.open("...")
        r'window\.open\s*\(\s*["\']([^"\']+)["\']',
        # onclick with location
        r'onclick\s*=\s*["\'][^"\']*(?:location(?:\.href)?|window\.open)\s*\(\s*["\']([^"\']+)["\']',
    ]
    for pat in patterns:
        for match in re.finditer(pat, html, re.IGNORECASE):
            url = match.group(1).strip()
            if url and not url.startswith(("javascript:", "#", "void")):
                urls.append(url)
    return urls


def _extract_meta_urls(html: str) -> List[str]:
    """Find URLs in meta tags (og:url, twitter:url, etc.)."""
    urls = []
    patterns = [
        r'<meta[^>]+property\s*=\s*["\']og:url["\'][^>]+content\s*=\s*["\']([^"\']+)["\']',
        r'<meta[^>]+content\s*=\s*["\']([^"\']+)["\'][^>]+property\s*=\s*["\']og:url["\']',
        r'<meta[^>]+name\s*=\s*["\']twitter:url["\'][^>]+content\s*=\s*["\']([^"\']+)["\']',
        r'<link[^>]+rel\s*=\s*["\']canonical["\'][^>]+href\s*=\s*["\']([^"\']+)["\']',
    ]
    for pat in patterns:
        for match in re.finditer(pat, html, re.IGNORECASE):
            urls.append(match.group(1).strip())
    return urls


def _score_link(href: str, text: str, attrs: Dict[str, str], page_url: str) -> float:
    """Score a link 0.0–1.0 for download likelihood."""
    score = 0.0
    href_lower = href.lower()
    text_lower = text.lower()
    combined = f"{href_lower} {text_lower}"

    # Extension match — strongest signal
    ext = _get_extension(href)
    if ext in DOWNLOAD_EXTENSIONS:
        score += 0.4

    # Download signals in text or href
    for word in DOWNLOAD_SIGNALS:
        if word in text_lower:
            score += 0.15
            break
    for word in DOWNLOAD_SIGNALS:
        if word in href_lower:
            score += 0.1
            break

    # Ad signals — penalize heavily
    for word in AD_SIGNALS:
        if word in AD_SIGNALS_BOUNDED:
            # Short words need word-boundary matching to avoid false positives
            # e.g., "ad" inside "download" should NOT trigger
            if re.search(r'\b' + re.escape(word) + r'\b', combined):
                score -= 0.4
        else:
            if word in combined:
                score -= 0.4

    # Class/id signals
    for attr_key in ("class", "id", "rel"):
        val = attrs.get(attr_key, "").lower()
        if any(w in val for w in ("download", "dl", "btn-download")):
            score += 0.15
        if any(w in val for w in ("ad", "sponsor", "promo", "popup")):
            score -= 0.3

    # data-href / data-url — often used for real download targets
    for key in ("data-href", "data-url", "data-download", "data-link"):
        if attrs.get(key):
            score += 0.1

    # External link with matching domain — moderate signal
    if _is_same_domain(href, page_url):
        score += 0.05

    # Very short href or text — suspicious
    if len(href) < 5:
        score -= 0.1
    if len(text) < 2:
        score -= 0.05

    # Query string heavy — suspicious (tracking params)
    parsed = urllib.parse.urlparse(href)
    if len(parsed.query) > 100:
        score -= 0.1

    return max(0.0, min(1.0, score))


def _get_extension(url: str) -> str:
    """Extract file extension from URL, handling compound extensions."""
    parsed = urllib.parse.urlparse(url)
    path = parsed.path.lower()
    # Handle compound extensions like .tar.gz
    for compound in (".tar.gz", ".tar.bz2", ".tar.xz"):
        if path.endswith(compound):
            return compound
    _, ext = __import__("os").path.splitext(path)
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


def resolve_page(
    page_url: str,
    *,
    session: Optional[requests.Session] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 15,
    max_links: int = 100,
    on_progress: Optional[Callable[[str], None]] = None,
) -> List[ResolvedLink]:
    """Fetch a page and extract ranked download links.

    Args:
        page_url: URL of the page to scrape.
        session: Optional requests session for connection reuse.
        headers: Extra HTTP headers.
        timeout: Request timeout in seconds.
        max_links: Max links to consider.
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

    # Track the full redirect chain
    redirects = [r.url for r in resp.history] if hasattr(resp, "history") else []
    final_url = resp.url
    html = resp.text

    if on_progress:
        on_progress(f"Analyzing page ({len(html)} bytes)")

    # --- Extract candidates from HTML ---
    candidates: List[Tuple[str, str, Dict[str, str], str]] = []  # href, text, attrs, source

    # 1. <a> tags
    parser = _LinkExtractor()
    try:
        parser.feed(html)
    except Exception:
        pass
    for href, text, attrs in parser.links[:max_links]:
        resolved = _resolve_relative(href, final_url)
        candidates.append((resolved, text, attrs, "html_link"))

    # 2. JavaScript redirects
    for url in _extract_js_redirects(html):
        resolved = _resolve_relative(url, final_url)
        candidates.append((resolved, "[js-redirect]", {}, "js_redirect"))

    # 3. Meta URLs
    for url in _extract_meta_urls(html):
        resolved = _resolve_relative(url, final_url)
        candidates.append((resolved, "[meta]", {}, "meta_tag"))

    # --- Score and deduplicate ---
    seen: Set[str] = set()
    results: List[ResolvedLink] = []

    for href, text, attrs, source in candidates:
        # Skip fragments, mailto, javascript
        if href.startswith(("mailto:", "javascript:", "#", "tel:")):
            continue
        # Skip duplicates
        if href in seen:
            continue
        seen.add(href)

        score = _score_link(href, text, attrs, page_url)
        ext = _get_extension(href)

        rl = ResolvedLink(
            url=href,
            title=text,
            extension=ext,
            confidence=round(score, 3),
            source=source,
            redirects=redirects if source == "js_redirect" else [],
        )
        results.append(rl)

    # Sort by confidence descending
    results.sort(key=lambda r: r.confidence, reverse=True)

    if on_progress:
        top = results[0] if results else None
        if top:
            on_progress(f"Best candidate: {top.url} (confidence={top.confidence})")
        else:
            on_progress("No download links found")

    return results


def resolve_and_download(
    page_url: str,
    dest: str,
    *,
    session: Optional[requests.Session] = None,
    headers: Optional[Dict[str, str]] = None,
    min_confidence: float = 0.3,
    on_progress: Optional[Callable[[str], None]] = None,
) -> "downloader.Path":
    """Resolve a page, then download the best candidate.

    Args:
        page_url: URL of the page to scrape.
        dest: Local file path to save to.
        session: Optional requests session.
        headers: Extra HTTP headers.
        min_confidence: Minimum confidence to accept a link.
        on_progress: Status callback.

    Returns:
        Path to downloaded file.

    Raises:
        ValueError: If no link meets the confidence threshold.
        downloader.DownloadError: If download fails.
    """
    from . import downloader

    links = resolve_page(
        page_url, session=session, headers=headers, on_progress=on_progress
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

    dest_path = downloader.Path(dest)
    return downloader.download_file(
        best.url,
        dest_path,
        on_chunk=lambda done, total: on_progress(
            f"Downloaded {done}/{total} bytes"
        ) if on_progress else None,
    )
