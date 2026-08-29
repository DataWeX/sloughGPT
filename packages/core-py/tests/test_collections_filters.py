"""Tests for domains.collections.filters — LengthFilter, DedupFilter, KeywordFilter, RegexFilter, FilterChain; domains.collections.sources — Record."""

from domains.collections.filters import (
    LengthFilter, DedupFilter, KeywordFilter, RegexFilter, FilterChain,
)
from domains.collections.sources import Record


def _record(content: str) -> Record:
    return Record(content=content)


class TestRecord:
    def test_defaults(self):
        r = Record(content="hello")
        assert r.content == "hello"
        assert "timestamp" in r.metadata

    def test_custom_metadata(self):
        r = Record(content="hi", metadata={"author": "test"})
        assert r.metadata["author"] == "test"


class TestLengthFilter:
    def test_within_range(self):
        lf = LengthFilter(min_length=3, max_length=10)
        assert lf.accept(_record("hello")) is True

    def test_too_short(self):
        lf = LengthFilter(min_length=10)
        assert lf.accept(_record("hi")) is False

    def test_too_long(self):
        lf = LengthFilter(max_length=3)
        assert lf.accept(_record("hello world")) is False


class TestDedupFilter:
    def test_unique(self):
        df = DedupFilter()
        assert df.accept(_record("first")) is True

    def test_duplicate(self):
        df = DedupFilter()
        df.accept(_record("same"))
        assert df.accept(_record("same")) is False

    def test_reset(self):
        df = DedupFilter()
        df.accept(_record("text"))
        df.reset()
        assert df.accept(_record("text")) is True


class TestKeywordFilter:
    def test_include(self):
        kf = KeywordFilter(keywords=["python", "code"], mode="include")
        assert kf.accept(_record("I love python")) is True
        assert kf.accept(_record("I love java")) is False

    def test_exclude(self):
        kf = KeywordFilter(keywords=["spam"], mode="exclude")
        assert kf.accept(_record("hello world")) is True
        assert kf.accept(_record("spam email")) is False

    def test_empty_keywords(self):
        kf = KeywordFilter(keywords=[])
        assert kf.accept(_record("anything")) is True


class TestRegexFilter:
    def test_include(self):
        rf = RegexFilter(pattern=r"\b\d{3}\b", mode="include")
        assert rf.accept(_record("code 123")) is True
        assert rf.accept(_record("no numbers")) is False

    def test_exclude(self):
        rf = RegexFilter(pattern=r"test", mode="exclude")
        assert rf.accept(_record("hello")) is True
        assert rf.accept(_record("test case")) is False


class TestFilterChain:
    def test_chain(self):
        fc = FilterChain([LengthFilter(min_length=3), KeywordFilter(keywords=["hi"], mode="include")])
        assert fc.accept(_record("hi there")) is True
        assert fc.accept(_record("lo")) is False
