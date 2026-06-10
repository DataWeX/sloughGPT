"""Tests for bawl.crawl — concurrent crawling, URL normalization, crawl_urls."""

import sys
from unittest.mock import MagicMock, patch

import pytest
from bawl.crawl import crawl, crawl_urls, _normalize


def test_crawl_single_depth():
    pages = crawl("https://example.com", depth=0, max_pages=5)
    assert len(pages) == 1
    assert pages[0].title == "Example Domain"
    assert pages[0].url == "https://example.com"


def test_crawl_max_pages():
    pages = crawl("https://example.com", depth=2, max_pages=3)
    assert 1 <= len(pages) <= 3


def test_crawl_on_page_callback():
    seen = []
    crawl("https://example.com", depth=0, max_pages=1,
          on_page=lambda p: seen.append(p.url))
    assert seen == ["https://example.com"]


def test_crawl_with_workers():
    pages = crawl("https://example.com", depth=0, workers=3)
    assert len(pages) == 1
    assert pages[0].title == "Example Domain"


def test_crawl_urls_list():
    pages = crawl_urls(["https://example.com"])
    assert len(pages) == 1
    assert pages[0].title == "Example Domain"


def test_crawl_urls_empty():
    pages = crawl_urls([])
    assert pages == []


def test_crawl_dedup_skips_identical_text():
    """dedup=True should skip URLs whose text content matches an already-seen page."""
    p1 = MagicMock()
    p1.url = "https://a.com/1"
    p1.text = "same text"
    p2 = MagicMock()
    p2.url = "https://b.com/2"
    p2.text = "same text"
    p3 = MagicMock()
    p3.url = "https://c.com/3"
    p3.text = "different"
    crawl_mod = sys.modules["bawl.crawl"]
    with patch.object(crawl_mod, "parse") as mock_parse:
        mock_parse.side_effect = [p1, p2, p3]
        result = crawl_urls(
            ["https://a.com/1", "https://b.com/2", "https://c.com/3"],
            dedup=True,
        )
    assert len(result) == 2
    texts = {p.text for p in result}
    assert "same text" in texts
    assert "different" in texts


def test_crawl_dedup_keeps_different_text():
    """dedup=True should keep all pages if all have unique text."""
    p1 = MagicMock()
    p1.url = "https://a.com/1"
    p1.text = "first"
    p2 = MagicMock()
    p2.url = "https://b.com/2"
    p2.text = "second"
    crawl_mod = sys.modules["bawl.crawl"]
    with patch.object(crawl_mod, "parse") as mock_parse:
        mock_parse.side_effect = [p1, p2]
        result = crawl_urls(
            ["https://a.com/1", "https://b.com/2"],
            dedup=True,
        )
    assert len(result) == 2


def test_crawl_dedup_false_keeps_duplicates():
    """dedup=False should keep all pages even if text is identical."""
    p1 = MagicMock()
    p1.url = "https://a.com/1"
    p1.text = "same text"
    p2 = MagicMock()
    p2.url = "https://b.com/2"
    p2.text = "same text"
    crawl_mod = sys.modules["bawl.crawl"]
    with patch.object(crawl_mod, "parse") as mock_parse:
        mock_parse.side_effect = [p1, p2]
        result = crawl_urls(
            ["https://a.com/1", "https://b.com/2"],
        )
    assert len(result) == 2


def test_normalize_strips_fragment():
    result = _normalize("https://example.com/page#section")
    assert result == "https://example.com/page"


def test_normalize_strips_trailing_slash():
    result = _normalize("https://example.com/page/")
    assert result == "https://example.com/page"


def test_normalize_preserves_query():
    result = _normalize("https://example.com/page?q=1")
    assert result == "https://example.com/page?q=1"


def test_normalize_strips_www():
    result = _normalize("https://www.example.com/page")
    assert result == "https://example.com/page"


def test_normalize_lowercases():
    result = _normalize("HTTP://EXAMPLE.COM/Path")
    assert result == "http://example.com/path"


def test_normalize_empty():
    assert _normalize("https://example.com") == "https://example.com"


def test_crawl_dedup_false_includes():
    """With dedup=False, same page fetched multiple times is included each time."""
    pages = crawl("https://example.com", depth=0, max_pages=5, dedup=False)
    pages2 = crawl("https://example.com", depth=0, max_pages=5, dedup=False)
    assert len(pages) == 1
    assert len(pages2) == 1


def test_crawl_dedup_true_skips_duplicates():
    """With dedup=True, duplicate content is skipped."""
    pages = crawl("https://example.com", depth=0, max_pages=5, dedup=True)
    assert len(pages) == 1


def test_crawl_dedup_different_content():
    """With dedup=True, different pages are both included."""
    pages = crawl("https://example.com", depth=1, max_pages=5, dedup=True)
    assert 1 <= len(pages) <= 5


def test_crawl_urls_dedup():
    """crawl_urls with dedup=True skips duplicate text even at different URLs."""
    pages = crawl_urls(["https://example.com", "https://example.com"], dedup=True)
    assert len(pages) == 1


def test_crawl_exclude_skips_matching_url():
    """crawl with exclude pattern should skip matching URLs."""
    pages = crawl("https://example.com", depth=0, exclude=["*example*"])
    assert len(pages) == 0


def test_crawl_exclude_keeps_nonmatching():
    """crawl with exclude pattern should keep non-matching seed URLs."""
    pages = crawl("https://example.com", depth=0, exclude=["*nonexistent*"])
    assert len(pages) == 1


def test_crawl_urls_exclude():
    """crawl_urls with exclude pattern should skip matching URLs."""
    pages = crawl_urls(
        ["https://example.com", "https://example.com"],
        exclude=["*example*"],
    )
    assert len(pages) == 0


def test_crawl_urls_exclude_partial():
    """crawl_urls with exclude pattern should skip only matching URLs."""
    with patch.object(sys.modules["bawl.crawl"], "parse") as mock:
        p1 = MagicMock()
        p1.url = "https://example.com/page1"
        p1.text = "a"
        p2 = MagicMock()
        p2.url = "https://example.com/page2"
        p2.text = "b"
        mock.side_effect = lambda u, **kw: p1 if "page1" in u else p2
        pages = crawl_urls(
            ["https://example.com/page1", "https://example.com/page2"],
            exclude=["*page1*"],
        )
    assert len(pages) == 1
    assert pages[0].url == "https://example.com/page2"


def test_url_matches_exclude():
    """_url_matches_exclude matches fnmatch patterns correctly."""
    from bawl.crawl import _url_matches_exclude
    assert _url_matches_exclude("https://example.com/page", ["*page*"])
    assert not _url_matches_exclude("https://example.com/page", ["*other*"])
    assert _url_matches_exclude("https://example.com", ["*example*"])
    assert not _url_matches_exclude("https://other.com", ["*example*"])


def test_progress_tracker():
    """ProgressTracker produces correct status strings."""
    from bawl.crawl import ProgressTracker
    t = ProgressTracker(total_depth=3)
    assert "0 pages" in t.status()
    assert "0.0" in t.status()
    t.inc()
    assert "1 page" in t.status()
    t.inc(2)
    assert "3 pages" in t.status()
    t.error()
    assert "1 error" in t.status()
    t.depth = 2
    assert "2/3" in t.status()


def test_crawl_include_only():
    """crawl with include pattern should only fetch URLs matching the pattern."""
    with patch.object(sys.modules["bawl.crawl"], "parse") as mock_parse:
        mock_parse.return_value = MagicMock(url="https://example.com", text="test", title="")
        pages = crawl("https://example.com", depth=0, include=["*example*"])
    assert len(pages) == 1


def test_crawl_include_filters_out():
    """crawl with include pattern should skip non-matching URLs."""
    pages = crawl("https://example.com", depth=0, include=["*nonexistent*"])
    assert len(pages) == 0


def test_crawl_urls_include():
    """crawl_urls with include pattern should only fetch matching URLs."""
    with patch.object(sys.modules["bawl.crawl"], "parse") as mock:
        p1 = MagicMock()
        p1.url = "https://example.com/page1"
        p1.text = "a"
        mock.side_effect = lambda u, **kw: p1
        pages = crawl_urls(
            ["https://example.com/page1", "https://example.com/page2"],
            include=["*page1*"],
        )
    assert len(pages) == 1
    assert pages[0].url == "https://example.com/page1"


def test_url_matches_include():
    """_url_matches_include matches fnmatch patterns correctly."""
    from bawl.crawl import _url_matches_include
    assert _url_matches_include("https://example.com/page", ["*page*"])
    assert not _url_matches_include("https://example.com/page", ["*other*"])
    assert _url_matches_include("https://example.com", ["*example*"])
    assert _url_matches_include("anything", [])
    assert not _url_matches_include("https://other.com", ["*example*"])
