"""
crawl — recursive page crawler. Concurrent, same-domain, depth-controlled, robots.txt aware.

Usage:
    from bawl import crawl
    pages = crawl("https://docs.python.org/3/", depth=2, max_pages=10)
    for p in pages:
        print(p.url, p.title)
"""

import fnmatch
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

from .fetch import AGENT
from .parse import parse, Page

_robots: dict[str, RobotFileParser] = {}


class ProgressTracker:
    """Tracks crawl progress for live terminal display.

    Usage:
        tracker = ProgressTracker(depth=3)
        crawl(..., on_page=lambda p: tracker.inc())
        print(tracker.status())  # "3 pages, 0 errors, 2.1s"
    """

    def __init__(self, total_depth: int = 0):
        self.pages = 0
        self.errors = 0
        self.depth = 0
        self.total_depth = total_depth
        self._start = time.time()

    def inc(self, n: int = 1) -> None:
        """Increment page count."""
        self.pages += n

    def error(self, n: int = 1) -> None:
        """Increment error count."""
        self.errors += n

    def elapsed(self) -> float:
        """Seconds since creation."""
        return time.time() - self._start

    def status(self) -> str:
        """Short one-line status string."""
        elapsed = self.elapsed()
        parts = [f"{self.pages} page{'s' if self.pages != 1 else ''}"]
        if self.errors:
            parts.append(f"{self.errors} error{'s' if self.errors != 1 else ''}")
        parts.append(f"{elapsed:.1f}s")
        if self.total_depth:
            parts.append(f"depth {self.depth}/{self.total_depth}")
        return ", ".join(parts) + " "


def _normalize(url: str) -> str:
    """Normalize a URL: lower scheme+host, strip fragment, strip trailing slash."""
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    path = parsed.path.lower().rstrip("/") or "/"
    if path == "/" and not parsed.path:
        path = ""
    query = parsed.query
    return scheme + "://" + netloc + path + (f"?{query}" if query else "")


def _same_origin(url_a: str, url_b: str) -> bool:
    """Check if two normalized URLs share the same scheme+netloc."""
    return urlparse(url_a).netloc.lower().lstrip("www.") == urlparse(url_b).netloc.lower().lstrip("www.")


def _check_robots(url: str, user_agent: str = AGENT) -> bool:
    """Check if a URL is allowed by robots.txt. Caches parsers per domain.

    Args:
        url: Full URL to check.
        user_agent: User-agent string for robots.txt matching.

    Returns:
        True if allowed (or robots.txt unreachable), False if disallowed.

    Side effects:
        - Caches RobotFileParser per domain in _robots dict.
        - May fetch robots.txt on first access per domain.
    """
    parsed = urlparse(url)
    domain = f"{parsed.scheme}://{parsed.netloc}"
    if domain not in _robots:
        rp = RobotFileParser()
        rp.set_url(f"{domain}/robots.txt")
        try:
            rp.read()
        except Exception:
            rp = RobotFileParser()
        _robots[domain] = rp
    return _robots[domain].can_fetch(user_agent, url)


def _url_matches_exclude(url: str, patterns: list[str]) -> bool:
    """Check if a URL matches any of the given glob patterns.

    Args:
        url: URL to check.
        patterns: List of fnmatch glob patterns.

    Returns:
        True if the URL matches any pattern.
    """
    for pat in patterns:
        if fnmatch.fnmatch(url, pat):
            return True
    return False


def _url_matches_include(url: str, patterns: list[str]) -> bool:
    """Check if a URL matches any of the given include patterns.

    Args:
        url: URL to check.
        patterns: List of fnmatch glob patterns.

    Returns:
        True if the URL matches at least one pattern.
    """
    if not patterns:
        return True
    for pat in patterns:
        if fnmatch.fnmatch(url, pat):
            return True
    return False


def _check_filters(url: str, include: list[str], exclude: list[str]) -> bool:
    """Check if a URL passes include/exclude filters.

    Args:
        url: URL to check.
        include: List of glob patterns; URL must match at least one (empty = all).
        exclude: List of glob patterns; URL must not match any (empty = none).

    Returns:
        True if the URL should be crawled.
    """
    if not _url_matches_include(url, include):
        return False
    if exclude and _url_matches_exclude(url, exclude):
        return False
    return True


def _links_from_page(page: Page, base_url: str, domain: str,
                     same_domain: bool,
                     include: Optional[list[str]] = None,
                     exclude: Optional[list[str]] = None) -> list[str]:
    """Extract and normalize same-origin links from a page.

    Args:
        page: Parsed page.
        base_url: URL used to resolve relative links.
        domain: Domain to filter against (netloc string).
        same_domain: If True, only return links matching domain.
        include: Optional list of glob patterns; only URLs matching at least one.
        exclude: Optional list of glob patterns; matching URLs are skipped.

    Returns:
        List of normalized absolute URLs.
    """
    out: list[str] = []
    for link in page.links:
        href = link["href"].strip()
        parsed = urlparse(href)
        if parsed.scheme not in ("http", "https", ""):
            continue
        if same_domain and parsed.netloc:
            if parsed.netloc.lower().lstrip("www.") != domain.lower().lstrip("www."):
                continue
        absolute = urljoin(base_url, href)
        normalized = _normalize(absolute)
        if not _check_filters(normalized, include or [], exclude or []):
            continue
        out.append(normalized)
    return out


def _text_hash(page: Page) -> int:
    """Return a hash of the page's visible text for content dedup."""
    return hash(page.text)


def crawl(
    seed: str,
    *,
    depth: int = 1,
    max_pages: int = 50,
    rate: float = 0.5,
    timeout: int = 15,
    same_domain: bool = True,
    respect_robots: bool = True,
    on_page: Optional[Callable[[Page], None]] = None,
    workers: int = 5,
    dedup: bool = False,
    include: Optional[list[str]] = None,
    exclude: Optional[list[str]] = None,
) -> list[Page]:
    """Crawl URLs recursively from a seed URL. Uses a thread pool for concurrent fetching.

    Args:
        seed: Starting URL.
        depth: Max link-following depth (0 = seed only).
        max_pages: Max pages to crawl total.
        rate: Minimum seconds between requests to the same domain.
        timeout: Request timeout in seconds.
        same_domain: Only follow links to the same domain as seed.
        respect_robots: Obey robots.txt if True.
        on_page: Optional callback invoked after each successful page parse
                 (useful for streaming saves).
        workers: Number of concurrent fetcher threads.
        dedup: If True, skip pages whose text content matches a page already seen.
        include: Optional list of glob patterns; only matching URLs are crawled.
        exclude: Optional list of glob patterns; matching URLs are skipped.

    Returns:
        List of crawled Page objects (in fetch order, dedup skipped).

    Side effects:
        - Fetches live URLs over the network using a thread pool.
        - Calls _check_robots() which may fetch robots.txt.
        - Calls on_page() callback for each successful page.
    """
    visited: set[str] = set()
    seen_hashes: set[int] = set()
    results: list[Page] = []
    seed_norm = _normalize(seed)
    domain = urlparse(seed).netloc.lower().lstrip("www.")
    dedup_lock: Optional[Any] = threading.Lock() if dedup else None
    include = include or []
    exclude = exclude or []

    def _add_page(page: Page) -> bool:
        if dedup:
            with dedup_lock:
                h = _text_hash(page)
                if h in seen_hashes:
                    return False
                seen_hashes.add(h)
        results.append(page)
        if on_page:
            on_page(page)
        return True

    # Level 0: seed
    if not _check_filters(seed_norm, include, exclude):
        return results
    if respect_robots and not _check_robots(seed):
        return results
    page = parse(seed, timeout=timeout, rate=rate)
    if page:
        _add_page(page)

    if depth == 0 or not page:
        return results

    # BFS levels: each level fetches in parallel
    current_level: list[tuple[str, int]] = []
    for link_url in _links_from_page(page, seed, domain, same_domain, include, exclude):
        if link_url not in visited:
            current_level.append((link_url, 1))
            visited.add(link_url)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for level in range(1, depth + 1):
            if not current_level or len(results) >= max_pages:
                break

            next_level: list[tuple[str, int]] = []
            futures: dict = {}

            for url, _ in current_level:
                if len(results) >= max_pages:
                    break
                if not _check_filters(url, include, exclude):
                    continue
                if respect_robots and not _check_robots(url):
                    continue

                future = pool.submit(parse, url, timeout=timeout, rate=rate)
                futures[future] = url

            for future in as_completed(futures):
                if len(results) >= max_pages:
                    break
                url = futures[future]
                page = future.result()
                if page is None:
                    continue
                if not _add_page(page):
                    continue

                if level < depth:
                    for link_url in _links_from_page(page, url, domain, same_domain, include, exclude):
                        if link_url not in visited and len(results) + len(next_level) < max_pages:
                            next_level.append((link_url, level + 1))
                            visited.add(link_url)

            current_level = next_level

    return results


def crawl_urls(
    urls: list[str],
    *,
    rate: float = 0.5,
    timeout: int = 15,
    on_page: Optional[Callable[[Page], None]] = None,
    workers: int = 5,
    dedup: bool = False,
    include: Optional[list[str]] = None,
    exclude: Optional[list[str]] = None,
) -> list[Page]:
    """Fetch a list of URLs concurrently (non-recursive).

    Args:
        urls: List of URLs to fetch.
        rate: Rate limit between requests to the same domain.
        timeout: Request timeout.
        on_page: Optional callback after each page.
        workers: Number of concurrent fetcher threads.
        dedup: If True, skip pages whose text content matches a page already seen.
        include: Optional list of glob patterns; only matching URLs are fetched.
        exclude: Optional list of glob patterns; matching URLs are skipped.

    Returns:
        List of fetched Page objects.
    """
    results: list[Page] = []
    seen_hashes: set[int] = set()
    dedup_lock = threading.Lock() if dedup else None
    include = include or []
    exclude = exclude or []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for u in urls:
            if not _check_filters(u, include, exclude):
                continue
            futures[pool.submit(parse, u, timeout=timeout, rate=rate)] = u
        for future in as_completed(futures):
            page = future.result()
            if not page:
                continue
            if dedup:
                with dedup_lock:
                    h = _text_hash(page)
                    if h in seen_hashes:
                        continue
                    seen_hashes.add(h)
            results.append(page)
            if on_page:
                on_page(page)
    return results
