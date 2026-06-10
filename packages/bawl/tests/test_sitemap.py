"""Tests for bawl.sitemap"""

from bawl.sitemap import parse


def test_sitemap_parse():
    urls = parse("https://example.com/sitemap.xml")
    # example.com has no sitemap, so it should be empty
    assert isinstance(urls, list)


def test_sitemap_from_nonexistent():
    urls = parse("https://nosuchdomain99999.xyz/sitemap.xml")
    assert urls == []


def test_module_export():
    from bawl import parse_sitemap
    assert callable(parse_sitemap)
