"""sitemap — parse sitemap.xml. Zero deps (stdlib xml + urllib).

Usage:
    from bawl import parse_sitemap
    urls = parse_sitemap("https://example.com/sitemap.xml")
    for url in urls[:10]:
        print(url)
"""

import xml.etree.ElementTree as ET
from typing import Optional
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .fetch import AGENT


def parse(url: str, *, timeout: int = 15) -> list[str]:
    """Fetch and parse a sitemap.xml. Handles sitemap indexes recursively.

    Args:
        url: URL to sitemap.xml (or sitemap index XML).
        timeout: Request timeout in seconds.

    Returns:
        Flat list of all URLs from all nested sitemaps.

    Side effects:
        - Fetches the sitemap (and any sub-sitemaps) over the network.
    """
    urls: list[str] = []
    _parse_one(url, urls, timeout=timeout)
    return urls


def _parse_one(url: str, acc: list[str], *, timeout: int) -> None:
    """Recursively parse a single sitemap or sitemap index, appending URLs to acc.

    Args:
        url: URL to fetch and parse.
        acc: Accumulator list — URLs are appended here.
        timeout: Request timeout.

    Side effects:
        - Fetches XML over the network.
        - Mutates acc in place.
    """
    try:
        req = Request(url, headers={"User-Agent": AGENT})
        resp = urlopen(req, timeout=timeout)
        xml = resp.read()
    except Exception:
        return

    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return

    tag = root.tag.split("}")[-1] if "}" in root.tag else root.tag
    ns = root.tag[:root.tag.index("}") + 1] if "}" in root.tag else ""

    if tag == "sitemapindex":
        for child in root:
            loc = child.find(f"{ns}loc")
            if loc is not None and loc.text:
                _parse_one(loc.text.strip(), acc, timeout=timeout)
    elif tag == "urlset":
        for child in root:
            loc = child.find(f"{ns}loc")
            if loc is not None and loc.text:
                acc.append(loc.text.strip())
