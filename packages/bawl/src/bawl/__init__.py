"""bawl — fetch, parse, crawl, sitemap, store. Zero-dependency. Composable."""

__version__ = "0.3.0"

from .crawl import crawl, crawl_urls
from .fetch import fetch
from .parse import parse, parse_html
from .sitemap import parse as parse_sitemap
from .store import dumps, dumps_json_array, load, loads, save, save_json_array

__all__ = [
    "fetch",
    "parse",
    "parse_html",
    "save",
    "load",
    "dumps",
    "loads",
    "dumps_json_array",
    "save_json_array",
    "crawl",
    "crawl_urls",
    "parse_sitemap",
]
