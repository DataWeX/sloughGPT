"""bawl — fetch, parse, crawl, sitemap, store. Zero-dependency. Composable."""

__version__ = "0.3.0"

from .fetch import fetch
from .parse import parse, parse_html
from .store import save, load, dumps, loads, dumps_json_array, save_json_array
from .crawl import crawl, crawl_urls
from .sitemap import parse as parse_sitemap

__all__ = [
    "fetch", "parse", "parse_html",
    "save", "load", "dumps", "loads",
    "dumps_json_array", "save_json_array",
    "crawl", "crawl_urls", "parse_sitemap",
]
