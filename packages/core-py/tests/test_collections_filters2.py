"""Tests for domains.collections.filters — all filter classes and FilterChain."""

from domains.collections.filters import (
    LengthFilter,
    DedupFilter,
    KeywordFilter,
    RegexFilter,
    LanguageFilter,
    FilterChain,
    SamplerFilter,
    TransformFilter,
    TruncateFilter,
    PrefixFilter,
    MetadataFilter,
)
from domains.collections.sources import Record


def _rec(content: str, metadata: dict | None = None) -> Record:
    if metadata is not None:
        return Record(content=content, metadata=metadata)
    return Record(content=content)


class TestLengthFilter:
    def test_within_bounds(self):
        f = LengthFilter(min_length=3, max_length=10)
        assert f.accept(_rec("hello")) is True

    def test_at_min_boundary(self):
        f = LengthFilter(min_length=5, max_length=20)
        assert f.accept(_rec("12345")) is True

    def test_at_max_boundary(self):
        f = LengthFilter(min_length=1, max_length=5)
        assert f.accept(_rec("12345")) is True

    def test_below_min(self):
        f = LengthFilter(min_length=10)
        assert f.accept(_rec("hi")) is False

    def test_above_max(self):
        f = LengthFilter(max_length=3)
        assert f.accept(_rec("toolong")) is False

    def test_empty_string(self):
        f = LengthFilter(min_length=1, max_length=100)
        assert f.accept(_rec("")) is False

    def test_defaults(self):
        f = LengthFilter()
        assert f.min_length == 10
        assert f.max_length == 100000


class TestDedupFilter:
    def test_first_accept(self):
        d = DedupFilter()
        assert d.accept(_rec("abc")) is True

    def test_second_reject(self):
        d = DedupFilter()
        d.accept(_rec("abc"))
        assert d.accept(_rec("abc")) is False

    def test_different_content_accepted(self):
        d = DedupFilter()
        d.accept(_rec("aaa"))
        assert d.accept(_rec("bbb")) is True

    def test_reset_allows_reduplicates(self):
        d = DedupFilter()
        d.accept(_rec("x"))
        d.reset()
        assert d.accept(_rec("x")) is True

    def test_see_order_matters(self):
        d = DedupFilter()
        d.accept(_rec("first"))
        assert d.accept(_rec("second")) is True
        assert d.accept(_rec("first")) is False

    def test_many_unique(self):
        d = DedupFilter()
        for i in range(50):
            assert d.accept(_rec(f"item_{i}")) is True
        assert d.accept(_rec("item_0")) is False

    def test_seen_set_starts_empty(self):
        d = DedupFilter()
        assert len(d._seen) == 0


class TestKeywordFilter:
    def test_include_mode_match(self):
        f = KeywordFilter(keywords=["python"], mode="include")
        assert f.accept(_rec("I love python")) is True

    def test_include_mode_no_match(self):
        f = KeywordFilter(keywords=["python"], mode="include")
        assert f.accept(_rec("I love java")) is False

    def test_exclude_mode_match(self):
        f = KeywordFilter(keywords=["spam"], mode="exclude")
        assert f.accept(_rec("spam email")) is False

    def test_exclude_mode_no_match(self):
        f = KeywordFilter(keywords=["spam"], mode="exclude")
        assert f.accept(_rec("hello world")) is True

    def test_empty_keywords_always_true(self):
        f = KeywordFilter(keywords=[], mode="include")
        assert f.accept(_rec("anything")) is True

    def test_case_insensitive(self):
        f = KeywordFilter(keywords=["Python"], mode="include")
        assert f.accept(_rec("I love PYTHON")) is True

    def test_multiple_keywords_any_match(self):
        f = KeywordFilter(keywords=["foo", "bar"], mode="include")
        assert f.accept(_rec("has foo")) is True
        assert f.accept(_rec("has bar")) is True
        assert f.accept(_rec("has neither")) is False

    def test_multiple_keywords_exclude(self):
        f = KeywordFilter(keywords=["foo", "bar"], mode="exclude")
        assert f.accept(_rec("has foo")) is False
        assert f.accept(_rec("clean")) is True

    def test_defaults(self):
        f = KeywordFilter()
        assert f.keywords == []
        assert f.mode == "include"


class TestRegexFilter:
    def test_include_mode_match(self):
        f = RegexFilter(pattern=r"\d+", mode="include")
        assert f.accept(_rec("has 42 digits")) is True

    def test_include_mode_no_match(self):
        f = RegexFilter(pattern=r"\d+", mode="include")
        assert f.accept(_rec("no digits here")) is False

    def test_exclude_mode_match(self):
        f = RegexFilter(pattern=r"bad", mode="exclude")
        assert f.accept(_rec("this is bad")) is False

    def test_exclude_mode_no_match(self):
        f = RegexFilter(pattern=r"bad", mode="exclude")
        assert f.accept(_rec("this is good")) is True

    def test_empty_pattern_always_true(self):
        f = RegexFilter(pattern="", mode="include")
        assert f.accept(_rec("anything")) is True

    def test_case_insensitive_default(self):
        f = RegexFilter(pattern=r"hello", mode="include")
        assert f.accept(_rec("HELLO world")) is True

    def test_complex_pattern(self):
        f = RegexFilter(pattern=r"^[A-Z][a-z]+\.\s", mode="include")
        assert f.accept(_rec("Hello. world")) is True
        assert f.accept(_rec("hello. world")) is True

    def test_reject_when_no_match(self):
        f = RegexFilter(pattern=r"^[A-Z][a-z]+\.\s", mode="include")
        assert f.accept(_rec("123 no start")) is False

    def test_defaults(self):
        f = RegexFilter()
        assert f.pattern == ""
        assert f.mode == "include"


class TestLanguageFilter:
    def test_all_ascii(self):
        f = LanguageFilter(allowed_chars_ratio=0.8)
        assert f.accept(_rec("Hello world!")) is True

    def test_mostly_ascii(self):
        f = LanguageFilter(allowed_chars_ratio=0.8)
        assert f.accept(_rec("Hello 你好 world")) is True

    def test_non_ascii_dominant(self):
        f = LanguageFilter(allowed_chars_ratio=0.8)
        assert f.accept(_rec("你好世界测试数据")) is False

    def test_empty_content(self):
        f = LanguageFilter(allowed_chars_ratio=0.8)
        assert f.accept(_rec("")) is False

    def test_ratio_one_requires_all_ascii(self):
        f = LanguageFilter(allowed_chars_ratio=1.0)
        assert f.accept(_rec("pure ascii")) is True
        assert f.accept(_rec("has 你好")) is False

    def test_ratio_zero_always_true(self):
        f = LanguageFilter(allowed_chars_ratio=0.0)
        assert f.accept(_rec("anything")) is True

    def test_defaults(self):
        f = LanguageFilter()
        assert f.allowed_chars_ratio == 0.8


class TestFilterChain:
    def test_empty_chain_accepts_all(self):
        fc = FilterChain()
        assert fc.accept(_rec("anything")) is True

    def test_single_pass(self):
        fc = FilterChain([LengthFilter(min_length=3)])
        assert fc.accept(_rec("hello")) is True

    def test_single_reject(self):
        fc = FilterChain([LengthFilter(min_length=100)])
        assert fc.accept(_rec("hi")) is False

    def test_all_must_pass(self):
        fc = FilterChain([
            LengthFilter(min_length=3),
            KeywordFilter(keywords=["hi"], mode="include"),
        ])
        assert fc.accept(_rec("hi there")) is True
        assert fc.accept(_rec("lo")) is False
        assert fc.accept(_rec("bye there")) is False

    def test_stats_tracking(self):
        fc = FilterChain([LengthFilter(min_length=10)])
        fc.accept(_rec("short"))
        fc.accept(_rec("this is long enough"))
        assert fc.stats["accepted"] == 1
        assert fc.stats["rejected"] == 1

    def test_add_returns_self(self):
        fc = FilterChain()
        result = fc.add(LengthFilter())
        assert result is fc

    def test_add_chaining(self):
        fc = FilterChain()
        fc.add(LengthFilter(min_length=1)).add(KeywordFilter(keywords=["x"]))
        assert len(fc.filters) == 2

    def test_filter_records(self):
        fc = FilterChain([LengthFilter(min_length=5)])
        records = [_rec("hi"), _rec("hello"), _rec("yo"), _rec("greetings")]
        result = fc.filter_records(records)
        assert len(result) == 2
        assert result[0].content == "hello"
        assert result[1].content == "greetings"

    def test_filter_records_empty(self):
        fc = FilterChain()
        assert fc.filter_records([]) == []

    def test_stats_accumulate(self):
        fc = FilterChain([LengthFilter(min_length=100)])
        for _ in range(5):
            fc.accept(_rec("short"))
        assert fc.stats["rejected"] == 5
        assert fc.stats["accepted"] == 0


class TestSamplerFilter:
    def test_rate_zero_rejects_all(self):
        f = SamplerFilter(rate=0.0)
        for _ in range(100):
            assert f.accept(_rec("item")) is False

    def test_rate_one_accepts_all(self):
        f = SamplerFilter(rate=1.0)
        for _ in range(100):
            assert f.accept(_rec("item")) is True

    def test_rate_half_approximate(self):
        f = SamplerFilter(rate=0.5)
        accepted = sum(1 for _ in range(1000) if f.accept(_rec("item")))
        assert 300 < accepted < 700

    def test_deterministic_sequence(self):
        f1 = SamplerFilter(rate=0.5)
        f2 = SamplerFilter(rate=0.5)
        results1 = [f1.accept(_rec("x")) for _ in range(20)]
        results2 = [f2.accept(_rec("x")) for _ in range(20)]
        assert results1 == results2

    def test_defaults(self):
        f = SamplerFilter()
        assert f.rate == 0.1


class TestTransformFilter:
    def test_always_accepts(self):
        f = TransformFilter()
        assert f.accept(_rec("anything")) is True

    def test_default_identity_transform(self):
        f = TransformFilter()
        r = _rec("hello")
        result = f.transform(r)
        assert result.content == "hello"

    def test_custom_transform(self):
        f = TransformFilter(transform_fn=lambda r: Record(content=r.content.upper(), metadata=r.metadata))
        result = f.transform(_rec("hello"))
        assert result.content == "HELLO"

    def test_non_record_return_wraps_string(self):
        f = TransformFilter(transform_fn=lambda r: r.content[::-1])
        result = f.transform(_rec("abc"))
        assert result.content == "cba"
        assert "timestamp" in result.metadata

    def test_transform_preserves_metadata(self):
        f = TransformFilter(transform_fn=lambda r: Record(content="new", metadata=r.metadata))
        original = _rec("old", {"custom": True})
        result = f.transform(original)
        assert result.metadata.get("custom") is True


class TestTruncateFilter:
    def test_short_content_unchanged(self):
        f = TruncateFilter(max_length=100)
        r = _rec("hello")
        f.accept(r)
        assert r.content == "hello"

    def test_long_content_truncated(self):
        f = TruncateFilter(max_length=5)
        r = _rec("hello world")
        f.accept(r)
        assert r.content == "hello"

    def test_exact_length_unchanged(self):
        f = TruncateFilter(max_length=5)
        r = _rec("12345")
        f.accept(r)
        assert r.content == "12345"

    def test_always_returns_true(self):
        f = TruncateFilter(max_length=3)
        assert f.accept(_rec("toolong")) is True

    def test_defaults(self):
        f = TruncateFilter()
        assert f.max_length == 1000


class TestPrefixFilter:
    def test_adds_prefix(self):
        f = PrefixFilter(prefix="PRE: ")
        r = _rec("text")
        f.accept(r)
        assert r.content == "PRE: text"

    def test_empty_prefix_no_change(self):
        f = PrefixFilter(prefix="")
        r = _rec("text")
        f.accept(r)
        assert r.content == "text"

    def test_always_returns_true(self):
        f = PrefixFilter(prefix="x")
        assert f.accept(_rec("y")) is True

    def test_defaults(self):
        f = PrefixFilter()
        assert f.prefix == ""


class TestMetadataFilter:
    def test_include_mode_match(self):
        f = MetadataFilter(key="source", values=["web", "api"], mode="include")
        assert f.accept(_rec("c", {"source": "web"})) is True

    def test_include_mode_no_match(self):
        f = MetadataFilter(key="source", values=["web", "api"], mode="include")
        assert f.accept(_rec("c", {"source": "db"})) is False

    def test_exclude_mode_match(self):
        f = MetadataFilter(key="type", values=["spam"], mode="exclude")
        assert f.accept(_rec("c", {"type": "spam"})) is False

    def test_exclude_mode_no_match(self):
        f = MetadataFilter(key="type", values=["spam"], mode="exclude")
        assert f.accept(_rec("c", {"type": "good"})) is True

    def test_empty_key_always_true(self):
        f = MetadataFilter(key="", values=["a"], mode="include")
        assert f.accept(_rec("c", {"k": "v"})) is True

    def test_empty_values_always_true(self):
        f = MetadataFilter(key="source", values=[], mode="include")
        assert f.accept(_rec("c", {"source": "web"})) is True

    def test_missing_metadata_key(self):
        f = MetadataFilter(key="missing", values=["a"], mode="include")
        assert f.accept(_rec("c")) is False

    def test_non_string_value_converted(self):
        f = MetadataFilter(key="count", values=["42"], mode="include")
        assert f.accept(_rec("c", {"count": 42})) is True

    def test_defaults(self):
        f = MetadataFilter()
        assert f.key == ""
        assert f.values == []
        assert f.mode == "include"


class TestFilterProtocolCompliance:
    def test_all_filters_are_filter_protocol(self):
        from domains.collections.filters import Filter
        instances = [
            LengthFilter(),
            DedupFilter(),
            KeywordFilter(),
            RegexFilter(),
            LanguageFilter(),
            SamplerFilter(),
            TransformFilter(),
            TruncateFilter(),
            PrefixFilter(),
            MetadataFilter(),
        ]
        for inst in instances:
            assert isinstance(inst, Filter), f"{type(inst).__name__} does not implement Filter"
