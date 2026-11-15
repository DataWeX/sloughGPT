"""Unit tests for the crawl page store (pure domain, no server imports)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from domain.training._internal.crawl_store import (
    CrawlIngestResult,
    CrawlStoreError,
    ingest_crawl,
    normalize_page,
    normalize_pages,
    parse_jsonl_pages,
)


def _page(url="https://example.com/a", text="hello world", **extra):
    base = {"url": url, "text": text}
    base.update(extra)
    return base


class TestNormalizePage:
    def test_valid_page_passes(self):
        out = normalize_page(_page(title="Title", domain="example.com"))
        assert out is not None
        assert out["url"] == "https://example.com/a"
        assert out["text"] == "hello world"
        assert out["title"] == "Title"
        assert out["domain"] == "example.com"
        assert out["word_count"] == 2

    def test_non_dict_dropped(self):
        assert normalize_page("not a page") is None
        assert normalize_page(None) is None
        assert normalize_page(["url", "text"]) is None

    def test_missing_url_dropped(self):
        assert normalize_page({"text": "hi"}) is None

    def test_empty_url_dropped(self):
        assert normalize_page(_page(url="   ")) is None

    def test_missing_text_dropped(self):
        assert normalize_page({"url": "https://x.com"}) is None

    def test_empty_text_dropped(self):
        assert normalize_page(_page(text="  ")) is None

    def test_non_string_url_dropped(self):
        assert normalize_page(_page(url=42)) is None

    def test_extra_keys_ignored(self):
        out = normalize_page(_page(unknown_key="keep me out"))
        assert "unknown_key" not in out


class TestNormalizePages:
    def test_counts_kept_and_dropped(self):
        kept, dropped = normalize_pages(
            [_page(), _page(url="https://b.com"), {"text": "no url"}, "junk"]
        )
        assert len(kept) == 2
        assert dropped == 2


class TestParseJsonlPages:
    def test_parses_multiple_lines(self):
        raw = json.dumps(_page()) + "\n" + json.dumps(_page(url="https://b.com")) + "\n"
        pages = parse_jsonl_pages(raw)
        assert len(pages) == 2
        assert pages[1]["url"] == "https://b.com"

    def test_blank_lines_skipped(self):
        raw = "\n\n" + json.dumps(_page()) + "\n\n"
        assert len(parse_jsonl_pages(raw)) == 1

    def test_bad_line_names_lineno(self):
        raw = json.dumps(_page()) + "\n{not json\n"
        with pytest.raises(CrawlStoreError, match="line 2"):
            parse_jsonl_pages(raw)


class TestIngestCrawl:
    def test_stores_raw_and_corpus(self, tmp_path: Path):
        result = ingest_crawl(
            "site_a",
            [_page(), _page(url="https://b.com", text="second page")],
            datasets_dir=tmp_path,
            source="crawl-run-1",
        )
        assert isinstance(result, CrawlIngestResult)
        assert result.pages_received == 2
        assert result.pages_stored == 2
        assert result.pages_dropped == 0

        raw_path = Path(result.raw_path)
        assert raw_path.is_file()
        assert raw_path.parent == tmp_path / "site_a" / "raw"
        raw_lines = raw_path.read_text(encoding="utf-8").splitlines()
        assert len(raw_lines) == 2
        rec = json.loads(raw_lines[0])
        assert rec["url"] == "https://example.com/a"
        assert rec["source"] == "crawl-run-1"
        assert rec["_stored_at"] > 0

        corpus_lines = Path(result.corpus_path).read_text(encoding="utf-8").splitlines()
        assert len(corpus_lines) == 2
        row = json.loads(corpus_lines[0])
        assert row == {
            "content": "hello world",
            "url": "https://example.com/a",
            "title": "",
            "source": "crawl-run-1",
        }

    def test_second_ingest_appends_corpus_and_new_raw_file(self, tmp_path: Path):
        first = ingest_crawl("site_a", [_page()], datasets_dir=tmp_path)
        second = ingest_crawl("site_a", [_page(url="https://b.com")], datasets_dir=tmp_path)
        assert Path(second.raw_path) != Path(first.raw_path)
        assert Path(second.raw_path).is_file()
        corpus_lines = Path(second.corpus_path).read_text(encoding="utf-8").splitlines()
        assert len(corpus_lines) == 2  # appended, not overwritten

    def test_same_second_ingests_get_unique_raw_names(self, tmp_path: Path):
        paths = {
            ingest_crawl(
                "site_a", [_page(url=f"https://x.com/{i}")], datasets_dir=tmp_path
            ).raw_path
            for i in range(3)
        }
        assert len(paths) == 3

    def test_invalid_name_rejected(self, tmp_path: Path):
        with pytest.raises(CrawlStoreError, match="invalid dataset name"):
            ingest_crawl("bad/name", [_page()], datasets_dir=tmp_path)
        with pytest.raises(CrawlStoreError, match="invalid dataset name"):
            ingest_crawl("", [_page()], datasets_dir=tmp_path)

    def test_empty_batch_rejected(self, tmp_path: Path):
        with pytest.raises(CrawlStoreError, match="empty"):
            ingest_crawl("site_a", [], datasets_dir=tmp_path)

    def test_all_dropped_rejected(self, tmp_path: Path):
        with pytest.raises(CrawlStoreError, match="no usable pages"):
            ingest_crawl("site_a", [{"text": "no url"}, "junk"], datasets_dir=tmp_path)
        assert not (tmp_path / "site_a" / "raw").exists()

    def test_partial_drop_stores_kept(self, tmp_path: Path):
        result = ingest_crawl("site_a", [_page(), {"text": "no url"}], datasets_dir=tmp_path)
        assert result.pages_received == 2
        assert result.pages_stored == 1
        assert result.pages_dropped == 1
