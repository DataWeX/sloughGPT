"""fetch — get HTML from a URL. Zero deps (stdlib urllib).

Usage:
    from bawl import fetch
    html = fetch("https://example.com")
    if html: print(len(html))
"""

import threading
import time
from typing import Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

try:
    from importlib.metadata import version as _v
    _VER = _v("bawl")
except Exception:
    _VER = "0.0"
AGENT = f"bawl/{_VER} (github.com/bawl)"
_hits: dict[str, float] = {}
_lock = threading.Lock()


def fetch(
    url: str, *, timeout: int = 15, rate: float = 0.5
) -> Optional[str]:
    """Fetch HTML from a URL. Thread-safe per-domain rate limiting.

    Args:
        url: Full URL to fetch.
        timeout: Request timeout in seconds.
        rate: Minimum seconds between requests to the same domain.

    Returns:
        HTML string, or None if the request failed or content is not text/html.

    Side effects:
        - Throttles requests per-domain using _throttle().
    """
    domain = url.split("/")[2] if "//" in url else url
    _throttle(domain, rate)

    req = Request(url, headers={"User-Agent": AGENT, "Accept": "text/html"})
    try:
        resp = urlopen(req, timeout=timeout)
        ct = resp.headers.get("Content-Type", "")
        if "text" not in ct and "html" not in ct:
            return None
        return resp.read().decode("utf-8", errors="replace")
    except URLError:
        return None
    except Exception:
        return None


def _throttle(domain: str, rate: float) -> None:
    """Sleep if we're requesting the same domain too fast. Thread-safe.

    Args:
        domain: Domain string (e.g. "example.com").
        rate: Minimum gap in seconds.

    Side effects:
        - Calls time.sleep() if needed.
        - Updates _hits[domain] with current time.
    """
    with _lock:
        now = time.time()
        last = _hits.get(domain, 0)
        gap = now - last
        if gap < rate:
            sleep = rate - gap
        else:
            sleep = 0
        _hits[domain] = now
    if sleep:
        time.sleep(sleep)
