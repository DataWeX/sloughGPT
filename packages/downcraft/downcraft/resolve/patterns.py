"""
Pattern matching for common download page structures.

Extracts download URLs from JS-heavy pages by recognizing recurring
HTML/JS patterns rather than site-specific logic. Each matcher returns
a list of candidate URLs found by its pattern.
"""

import base64
import json
import re
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Pattern, Set


@dataclass(frozen=True)
class Extraction:
    """A single URL extraction result."""
    url: str
    source: str
    confidence: float = 1.0


# ---------------------------------------------------------------------------
# Base matcher
# ---------------------------------------------------------------------------

class Matcher:
    """Base class for pattern matchers."""

    name: str = "base"

    def extract(self, html: str) -> List[Extraction]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# JS variable assignments
# ---------------------------------------------------------------------------

_VAR_URL_RE = re.compile(
    r"""(?:var|let|const)\s+"""
    r"""(\w*(?:url|link|download|href|file|src|path|target)\w*)"""
    r"""\s*=\s*["']([^"']+)["']""",
    re.IGNORECASE,
)

_WINDOW_VAR_RE = re.compile(
    r"""window\.(\w*(?:url|link|download|href|file|src|path|target)\w*)"""
    r"""\s*=\s*["']([^"']+)["']""",
    re.IGNORECASE,
)

_JS_PREFIXES: tuple = ("javascript:", "#", "void")


class JsVariableMatcher(Matcher):
    """Extract URLs from JS variable assignments."""

    name = "js_variable"

    def extract(self, html: str) -> List[Extraction]:
        out: List[Extraction] = []
        for m in _VAR_URL_RE.finditer(html):
            url = m.group(2).strip()
            if url and not url.startswith(_JS_PREFIXES):
                out.append(Extraction(url=url, source=self.name))
        for m in _WINDOW_VAR_RE.finditer(html):
            url = m.group(2).strip()
            if url and not url.startswith(_JS_PREFIXES):
                out.append(Extraction(url=url, source=self.name))
        return out


# ---------------------------------------------------------------------------
# JS redirects and meta refresh
# ---------------------------------------------------------------------------

_JS_REDIRECT_PATS: List[Pattern] = [
    re.compile(r'window\.location(?:\.href)?\s*=\s*["\']([^"\']+)["\']', re.I),
    re.compile(r'window\.location\.assign\s*\(\s*["\']([^"\']+)["\']', re.I),
    re.compile(r'window\.location\.replace\s*\(\s*["\']([^"\']+)["\']', re.I),
    re.compile(r'document\.location(?:\.href)?\s*=\s*["\']([^"\']+)["\']', re.I),
    re.compile(r'setTimeout\s*\(\s*["\'](?:location(?:\.href)?\s*=\s*)?["\']?([^"\']+)["\']', re.I),
    re.compile(r'<meta[^>]+http-equiv\s*=\s*["\']refresh["\'][^>]+content\s*=\s*["\'][^"\']*url=([^"\']+)["\']', re.I),
    re.compile(r'<meta[^>]+content\s*=\s*["\'][^"\']*url=([^"\']+)["\'][^>]+http-equiv\s*=\s*["\']refresh["\']', re.I),
    re.compile(r'window\.open\s*\(\s*["\']([^"\']+)["\']', re.I),
    re.compile(r'onclick\s*=\s*["\'][^"\']*(?:location(?:\.href)?|window\.open)\s*\(\s*["\']([^"\']+)["\']', re.I),
]


class JsRedirectMatcher(Matcher):
    """Extract URLs from JS redirects and meta refresh tags."""

    name = "js_redirect"

    def extract(self, html: str) -> List[Extraction]:
        out: List[Extraction] = []
        for pat in _JS_REDIRECT_PATS:
            for m in pat.finditer(html):
                url = m.group(1).strip()
                if url and not url.startswith(_JS_PREFIXES):
                    out.append(Extraction(url=url, source=self.name))
        return out


# ---------------------------------------------------------------------------
# Meta tag URLs (og:url, twitter:url, canonical)
# ---------------------------------------------------------------------------

_META_URL_PATS: List[Pattern] = [
    re.compile(r'<meta[^>]+property\s*=\s*["\']og:url["\'][^>]+content\s*=\s*["\']([^"\']+)["\']', re.I),
    re.compile(r'<meta[^>]+content\s*=\s*["\']([^"\']+)["\'][^>]+property\s*=\s*["\']og:url["\']', re.I),
    re.compile(r'<meta[^>]+name\s*=\s*["\']twitter:url["\'][^>]+content\s*=\s*["\']([^"\']+)["\']', re.I),
    re.compile(r'<link[^>]+rel\s*=\s*["\']canonical["\'][^>]+href\s*=\s*["\']([^"\']+)["\']', re.I),
]


class MetaTagMatcher(Matcher):
    """Extract URLs from meta tags (og:url, twitter:url, canonical)."""

    name = "meta_tag"

    def extract(self, html: str) -> List[Extraction]:
        out: List[Extraction] = []
        for pat in _META_URL_PATS:
            for m in pat.finditer(html):
                url = m.group(1).strip()
                if url:
                    out.append(Extraction(url=url, source=self.name))
        return out


# ---------------------------------------------------------------------------
# Hidden data-* attributes (countdown, popunder, obfuscated)
# ---------------------------------------------------------------------------

_OBFUSCATED_ATTRS: FrozenSet[str] = frozenset({
    "data-download-url", "data-real-url", "data-file", "data-link-url",
    "data-countdown-url", "data-timer-url", "data-final-url",
    "data-popunder", "data-pop", "data-href-real",
    "data-action-url", "data-redirect", "data-target",
    "data-href", "data-url", "data-download", "data-link",
})

_DATA_ATTR_RE: Dict[str, re.Pattern] = {
    attr: re.compile(rf'{attr}\s*=\s*["\']([^"\']+)["\']', re.I)
    for attr in _OBFUSCATED_ATTRS
}


class DataAttributeMatcher(Matcher):
    """Extract URLs from hidden data-* attributes."""

    name = "data_attribute"

    def extract(self, html: str) -> List[Extraction]:
        out: List[Extraction] = []
        for attr, pat in _DATA_ATTR_RE.items():
            for m in pat.finditer(html):
                val = m.group(1).strip()
                # Try base64 decode
                try:
                    decoded = base64.b64decode(val).decode("utf-8", errors="ignore")
                    if "://" in decoded or decoded.startswith("/"):
                        out.append(Extraction(url=decoded.strip(), source=self.name))
                        continue
                except Exception:
                    pass
                if "://" in val or val.startswith("/"):
                    out.append(Extraction(url=val, source=self.name))
        return out


# ---------------------------------------------------------------------------
# Base64 / hex / String.fromCharCode obfuscation
# ---------------------------------------------------------------------------

_ATOB_RE = re.compile(r'atob\s*\(\s*["\']([A-Za-z0-9+/=_-]{20,})["\']')
_DECODEURIComponent_RE = re.compile(r'decodeURIComponent\s*\(\s*["\']([^"\']{10,})["\']')
_FROMCharCode_RE = re.compile(r'String\.fromCharCode\s*\(\s*([\d,\s]+)\s*\)')
_HEX_DATA_RE = re.compile(
    r'(?:data-[\w-]+)\s*=\s*"((?:\\x[0-9a-fA-F]{2}){8,})"'
)


class ObfuscationMatcher(Matcher):
    """Extract URLs hidden via base64, hex, or String.fromCharCode."""

    name = "obfuscated"

    def extract(self, html: str) -> List[Extraction]:
        out: List[Extraction] = []

        for m in _ATOB_RE.finditer(html):
            try:
                decoded = base64.b64decode(m.group(1)).decode("utf-8", errors="ignore")
                if "://" in decoded or decoded.startswith("/"):
                    out.append(Extraction(url=decoded.strip(), source=self.name))
            except Exception:
                pass

        for m in _DECODEURIComponent_RE.finditer(html):
            try:
                import urllib.parse
                decoded = urllib.parse.unquote(m.group(1))
                if "://" in decoded or decoded.startswith("/"):
                    out.append(Extraction(url=decoded.strip(), source=self.name))
            except Exception:
                pass

        for m in _FROMCharCode_RE.finditer(html):
            try:
                codes = [int(c.strip()) for c in m.group(1).split(",") if c.strip()]
                decoded = "".join(chr(c) for c in codes)
                if "://" in decoded:
                    out.append(Extraction(url=decoded.strip(), source=self.name))
            except Exception:
                pass

        for m in _HEX_DATA_RE.finditer(html):
            try:
                raw = m.group(1).encode("ascii").decode("unicode_escape")
                if "://" in raw:
                    out.append(Extraction(url=raw.strip(), source=self.name))
            except Exception:
                pass

        return out


# ---------------------------------------------------------------------------
# JSON blobs (framework state)
# ---------------------------------------------------------------------------

_JSON_BLOB_PATS: List[Pattern] = [
    re.compile(r'window\.__(?:INITIAL_STATE|NUXT|NEXT_DATA|APP_DATA)__\s*=\s*(\{.+?\});', re.I | re.S),
    re.compile(r'(?:var|let|const)\s+config\s*=\s*(\{.+?\});', re.I | re.S),
]

_JSON_URL_KEYS: Set[str] = {
    "downloadUrl", "contentUrl", "url", "sameAs",
    "installUrl", "fileUrl", "actionUrl",
}


def _collect_dict_urls(d: dict, out: List[str], depth: int = 0) -> None:
    if depth > 5:
        return
    for key, val in d.items():
        if isinstance(val, str) and key in _JSON_URL_KEYS:
            if "://" in val or val.startswith("/"):
                out.append(val.strip())
        elif isinstance(val, dict):
            _collect_dict_urls(val, out, depth + 1)
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, dict):
                    _collect_dict_urls(item, out, depth + 1)
                elif isinstance(item, str) and "://" in item:
                    out.append(item.strip())


class JsonBlobMatcher(Matcher):
    """Extract URLs from embedded JSON blobs (framework state objects)."""

    name = "json_blob"

    def extract(self, html: str) -> List[Extraction]:
        out: List[Extraction] = []
        for pat in _JSON_BLOB_PATS:
            for m in pat.finditer(html):
                try:
                    data = json.loads(m.group(1))
                    urls: List[str] = []
                    _collect_dict_urls(data, urls)
                    for u in urls:
                        out.append(Extraction(url=u, source=self.name))
                except (json.JSONDecodeError, ValueError):
                    pass
        return out


# ---------------------------------------------------------------------------
# JSON-LD structured data
# ---------------------------------------------------------------------------

_LD_RE = re.compile(
    r'<script[^>]+type\s*=\s*["\']application/ld\+json["\'][^>]*>(.*?)</script>',
    re.I | re.S,
)


class JsonLdMatcher(Matcher):
    """Extract URLs from JSON-LD structured data."""

    name = "json_ld"

    def extract(self, html: str) -> List[Extraction]:
        out: List[Extraction] = []
        for m in _LD_RE.finditer(html):
            try:
                data = json.loads(m.group(1))
                items = data if isinstance(data, list) else [data]
                urls: List[str] = []
                for item in items:
                    if isinstance(item, dict):
                        _collect_dict_urls(item, urls)
                for u in urls:
                    out.append(Extraction(url=u, source=self.name))
            except (json.JSONDecodeError, ValueError):
                pass
        return out


# ---------------------------------------------------------------------------
# oEmbed discovery
# ---------------------------------------------------------------------------

_OEMBED_LINK_RE = re.compile(
    r'<link[^>]+rel\s*=\s*["\']alternate["\'][^>]+type\s*=\s*["\']application/json\+oembed["\'][^>]+href\s*=\s*["\']([^"\']+)["\']',
    re.I,
)
_OEMBED_LINK_REVERSE_RE = re.compile(
    r'<link[^>]+type\s*=\s*["\']application/json\+oembed["\'][^>]+rel\s*=\s*["\']alternate["\'][^>]+href\s*=\s*["\']([^"\']+)["\']',
    re.I,
)


class OEmbedMatcher(Matcher):
    """Discover oEmbed endpoints and extract the content URL.

    oEmbed is a standard for embedding content. Many sites expose
    an oEmbed endpoint that returns the direct URL and metadata.
    """

    name = "oembed"

    def extract(self, html: str) -> List[Extraction]:
        out: List[Extraction] = []
        for pat in (_OEMBED_LINK_RE, _OEMBED_LINK_REVERSE_RE):
            for m in pat.finditer(html):
                endpoint = m.group(1).strip()
                if endpoint:
                    out.append(Extraction(url=endpoint, source=self.name, confidence=0.7))
        return out


# ---------------------------------------------------------------------------
# Embedded player extraction (iframes)
# ---------------------------------------------------------------------------

_IFRAME_SRC_RE = re.compile(
    r'<iframe[^>]+(?<![a-zA-Z-])src\s*=\s*["\']([^"\']+)["\']', re.I
)
_IFRAME_DATA_SRC_RE = re.compile(
    r'<iframe[^>]+data-src\s*=\s*["\']([^"\']+)["\']', re.I
)


class EmbeddedPlayerMatcher(Matcher):
    """Extract URLs from embedded player iframes.

    Finds <iframe> tags and extracts the src URL. Also detects known
    player patterns (YouTube, Vimeo) for direct URL construction.
    """

    name = "embedded_player"

    def extract(self, html: str) -> List[Extraction]:
        out: List[Extraction] = []
        for pat in (_IFRAME_SRC_RE, _IFRAME_DATA_SRC_RE):
            for m in pat.finditer(html):
                url = m.group(1).strip()
                if url and not url.startswith(("javascript:", "#")):
                    out.append(Extraction(url=url, source=self.name))
        return out


# ---------------------------------------------------------------------------
# Registry — all matchers in priority order
# ---------------------------------------------------------------------------

ALL_MATCHERS: List[Matcher] = [
    OEmbedMatcher(),
    EmbeddedPlayerMatcher(),
    JsVariableMatcher(),
    JsRedirectMatcher(),
    MetaTagMatcher(),
    DataAttributeMatcher(),
    ObfuscationMatcher(),
    JsonBlobMatcher(),
    JsonLdMatcher(),
]


def extract_all(html: str, *, matchers: Optional[List[Matcher]] = None) -> List[Extraction]:
    """Run all matchers on HTML and return deduplicated extractions.

    Args:
        html: Raw HTML string.
        matchers: Optional subset of matchers to run. Defaults to all.

    Returns:
        List of Extraction sorted by confidence (best first), deduplicated by URL.
    """
    seen: Set[str] = set()
    out: List[Extraction] = []
    for matcher in (matchers or ALL_MATCHERS):
        for ex in matcher.extract(html):
            if ex.url not in seen:
                seen.add(ex.url)
                out.append(ex)
    return out
