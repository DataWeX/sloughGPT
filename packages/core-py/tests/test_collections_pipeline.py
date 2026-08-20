"""Tests for domains.collections — Record, sources, stores, filters, collector.

Covers: Record dataclass, FileSource, FileStore, MemoryStore, LengthFilter,
DedupFilter, KeywordFilter, RegexFilter, FilterChain, Collector.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from domains.collections.sources import Record, FileSource
from domains.collections.stores import FileStore, MemoryStore
from domains.collections.filters import (
    LengthFilter,
    DedupFilter,
    KeywordFilter,
    RegexFilter,
    FilterChain,
)
from domains.collections.collector import Collector


# ── Record ───────────────────────────────────────────────────────────

class TestRecord:
    def test_creation(self):
        r = Record(content="hello")
        assert r.content == "hello"
        assert "timestamp" in r.metadata

    def test_custom_metadata(self):
        r = Record(content="data", metadata={"key": "val"})
        assert r.metadata["key"] == "val"

    def test_to_dict(self):
        r = Record(content="test", metadata={"a": 1})
        d = r.to_dict()
        assert d["content"] == "test"
        assert d["metadata"]["a"] == 1


# ── FileSource ───────────────────────────────────────────────────────

class TestFileSource:
    def test_read(self, tmp_path):
        p = tmp_path / "data.jsonl"
        p.write_text('{"content": "line1"}\n{"content": "line2"}\n')
        src = FileSource(path=str(p))
        records = list(src.read())
        assert len(records) == 2
        assert records[0].content == "line1"

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        p.write_text("")
        src = FileSource(path=str(p))
        assert list(src.read()) == []


# ── MemoryStore ──────────────────────────────────────────────────────

class TestMemoryStore:
    def test_write_and_read(self):
        store = MemoryStore()
        store.write(Record(content="a"))
        store.write(Record(content="b"))
        items = list(store.read_all())
        assert len(items) == 2
        assert items[0].content == "a"

    def test_count(self):
        store = MemoryStore()
        assert store.count() == 0
        store.write(Record(content="x"))
        assert store.count() == 1


# ── FileStore ────────────────────────────────────────────────────────

class TestFileStore:
    def test_write_and_read(self, tmp_path):
        p = tmp_path / "out.jsonl"
        store = FileStore(path=str(p))
        store.write(Record(content="hello"))
        items = list(store.read_all())
        assert len(items) == 1
        assert items[0].content == "hello"

    def test_count(self, tmp_path):
        p = tmp_path / "out.jsonl"
        store = FileStore(path=str(p))
        assert store.count() == 0
        store.write(Record(content="x"))
        assert store.count() == 1


# ── Filters ──────────────────────────────────────────────────────────

class TestLengthFilter:
    def test_in_range(self):
        f = LengthFilter(min_length=2, max_length=10)
        assert f.accept(Record(content="hello")) is True

    def test_too_short(self):
        f = LengthFilter(min_length=5)
        assert f.accept(Record(content="hi")) is False

    def test_too_long(self):
        f = LengthFilter(max_length=3)
        assert f.accept(Record(content="hello")) is False


class TestDedupFilter:
    def test_first_passes(self):
        f = DedupFilter()
        assert f.accept(Record(content="unique")) is True

    def test_duplicate_rejected(self):
        f = DedupFilter()
        f.accept(Record(content="same"))
        assert f.accept(Record(content="same")) is False


class TestKeywordFilter:
    def test_include_mode(self):
        f = KeywordFilter(keywords=["python", "test"], mode="include")
        assert f.accept(Record(content="I love python")) is True
        assert f.accept(Record(content="I love java")) is False

    def test_exclude_mode(self):
        f = KeywordFilter(keywords=["spam"], mode="exclude")
        assert f.accept(Record(content="good content")) is True
        assert f.accept(Record(content="spam message")) is False


class TestRegexFilter:
    def test_match(self):
        f = RegexFilter(pattern=r"\d{3}-\d{4}")
        assert f.accept(Record(content="call 555-1234")) is True
        assert f.accept(Record(content="no number")) is False


class TestFilterChain:
    def test_all_pass(self):
        chain = FilterChain([LengthFilter(min_length=1), DedupFilter()])
        assert chain.accept(Record(content="hello")) is True

    def test_one_rejects(self):
        chain = FilterChain([LengthFilter(max_length=3), DedupFilter()])
        assert chain.accept(Record(content="hello")) is False

    def test_empty_chain(self):
        chain = FilterChain([])
        assert chain.accept(Record(content="anything")) is True

    def test_stats(self):
        chain = FilterChain([LengthFilter(min_length=1, max_length=3)])
        chain.accept(Record(content="hi"))
        chain.accept(Record(content="hello"))
        assert chain.stats["accepted"] == 1
        assert chain.stats["rejected"] == 1


# ── Collector ────────────────────────────────────────────────────────

class TestCollector:
    def test_collect(self, tmp_path):
        src_path = tmp_path / "src.jsonl"
        src_path.write_text('{"content": "a"}\n{"content": "b"}\n')
        source = FileSource(path=str(src_path))
        store = MemoryStore()
        collector = Collector(source, store)
        count = collector.collect()
        assert count == 2
        assert store.count() == 2

    def test_collect_with_filter(self, tmp_path):
        src_path = tmp_path / "src.jsonl"
        src_path.write_text('{"content": "short"}\n{"content": "this is a longer line"}\n')
        source = FileSource(path=str(src_path))
        store = MemoryStore()
        collector = Collector(source, store, filters=[LengthFilter(min_length=10)])
        count = collector.collect()
        assert count == 1

    def test_stats(self, tmp_path):
        src_path = tmp_path / "src.jsonl"
        src_path.write_text('{"content": "a"}\n')
        source = FileSource(path=str(src_path))
        store = MemoryStore()
        collector = Collector(source, store)
        collector.collect()
        assert collector.stats["collected"] == 1
