"""Tests for domains.collections.filters — all filter types and FilterChain.

Covers: LengthFilter, DedupFilter, KeywordFilter, RegexFilter, LanguageFilter,
SamplerFilter, TransformFilter, TruncateFilter, PrefixFilter, MetadataFilter,
FilterChain.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_core_dir = str(Path(__file__).resolve().parents[2])
if _core_dir not in sys.path:
    sys.path.insert(0, _core_dir)

from domains.collections.sources import Record
from domains.collections.filters import (
    LengthFilter,
    DedupFilter,
    KeywordFilter,
    RegexFilter,
    LanguageFilter,
    SamplerFilter,
    TransformFilter,
    TruncateFilter,
    PrefixFilter,
    MetadataFilter,
    FilterChain,
)


class TestLengthFilter:
    def test_within_range(self):
        f = LengthFilter(min_length=2, max_length=10)
        assert f.accept(Record(content="hello")) is True

    def test_too_short(self):
        f = LengthFilter(min_length=10)
        assert f.accept(Record(content="hi")) is False

    def test_too_long(self):
        f = LengthFilter(max_length=5)
        assert f.accept(Record(content="toolong")) is False


class TestDedupFilter:
    def test_first_accept(self):
        f = DedupFilter()
        assert f.accept(Record(content="hello")) is True

    def test_duplicate_reject(self):
        f = DedupFilter()
        f.accept(Record(content="hello"))
        assert f.accept(Record(content="hello")) is False

    def test_reset(self):
        f = DedupFilter()
        f.accept(Record(content="hello"))
        f.reset()
        assert f.accept(Record(content="hello")) is True


class TestKeywordFilter:
    def test_include(self):
        f = KeywordFilter(keywords=["python", "code"], mode="include")
        assert f.accept(Record(content="I love python")) is True
        assert f.accept(Record(content="I love rust")) is False

    def test_exclude(self):
        f = KeywordFilter(keywords=["spam"], mode="exclude")
        assert f.accept(Record(content="hello")) is True
        assert f.accept(Record(content="spam email")) is False

    def test_empty_keywords(self):
        f = KeywordFilter(keywords=[])
        assert f.accept(Record(content="anything")) is True


class TestRegexFilter:
    def test_include(self):
        f = RegexFilter(pattern=r"\d{3}-\d{4}", mode="include")
        assert f.accept(Record(content="call 555-1234 now")) is True
        assert f.accept(Record(content="no number here")) is False

    def test_exclude(self):
        f = RegexFilter(pattern=r"test", mode="exclude")
        assert f.accept(Record(content="hello")) is True
        assert f.accept(Record(content="test case")) is False

    def test_empty_pattern(self):
        f = RegexFilter(pattern="")
        assert f.accept(Record(content="anything")) is True


class TestLanguageFilter:
    def test_english(self):
        f = LanguageFilter(allowed_chars_ratio=0.8)
        assert f.accept(Record(content="Hello world")) is True

    def test_non_english(self):
        f = LanguageFilter(allowed_chars_ratio=0.9)
        assert f.accept(Record(content="日本語テスト")) is False

    def test_empty(self):
        f = LanguageFilter()
        assert f.accept(Record(content="")) is False


class TestSamplerFilter:
    def test_samples_subset(self):
        f = SamplerFilter(rate=0.5)
        results = [f.accept(Record(content="x")) for _ in range(100)]
        # Should accept roughly half
        assert 20 < sum(results) < 80


class TestTransformFilter:
    def test_always_accepts(self):
        f = TransformFilter()
        assert f.accept(Record(content="hello")) is True

    def test_transform(self):
        f = TransformFilter(transform_fn=lambda r: Record(content=r.content.upper()))
        result = f.transform(Record(content="hello"))
        assert result.content == "HELLO"

    def test_transform_non_record(self):
        f = TransformFilter(transform_fn=lambda r: "string result")
        result = f.transform(Record(content="hello"))
        assert result.content == "string result"


class TestTruncateFilter:
    def test_within_limit(self):
        f = TruncateFilter(max_length=100)
        r = Record(content="hello")
        f.accept(r)
        assert r.content == "hello"

    def test_truncates(self):
        f = TruncateFilter(max_length=5)
        r = Record(content="toolongcontent")
        f.accept(r)
        assert len(r.content) == 5


class TestPrefixFilter:
    def test_adds_prefix(self):
        f = PrefixFilter(prefix="[TAG] ")
        r = Record(content="hello")
        f.accept(r)
        assert r.content == "[TAG] hello"

    def test_empty_prefix(self):
        f = PrefixFilter(prefix="")
        r = Record(content="hello")
        f.accept(r)
        assert r.content == "hello"


class TestMetadataFilter:
    def test_include(self):
        f = MetadataFilter(key="type", values=["news", "blog"], mode="include")
        assert f.accept(Record(content="x", metadata={"type": "news"})) is True
        assert f.accept(Record(content="x", metadata={"type": "spam"})) is False

    def test_exclude(self):
        f = MetadataFilter(key="type", values=["spam"], mode="exclude")
        assert f.accept(Record(content="x", metadata={"type": "news"})) is True
        assert f.accept(Record(content="x", metadata={"type": "spam"})) is False

    def test_empty_key(self):
        f = MetadataFilter(key="", values=["x"])
        assert f.accept(Record(content="x")) is True


class TestFilterChain:
    def test_all_pass(self):
        chain = FilterChain([LengthFilter(min_length=1), KeywordFilter(keywords=[])])
        assert chain.accept(Record(content="hello")) is True
        assert chain.stats["accepted"] == 1

    def test_one_fails(self):
        chain = FilterChain([LengthFilter(min_length=100), KeywordFilter(keywords=[])])
        assert chain.accept(Record(content="short")) is False
        assert chain.stats["rejected"] == 1

    def test_filter_records(self):
        chain = FilterChain([LengthFilter(min_length=10)])
        records = [Record(content="short"), Record(content="long enough")]
        result = chain.filter_records(records)
        assert len(result) == 1

    def test_add_filter(self):
        chain = FilterChain()
        chain.add(LengthFilter(min_length=1))
        assert len(chain.filters) == 1
