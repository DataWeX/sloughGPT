"""Tests for domains.collections.filters — LengthFilter, DedupFilter, KeywordFilter, RegexFilter, LanguageFilter, SamplerFilter, TransformFilter, TruncateFilter, PrefixFilter, MetadataFilter, FilterChain; domains.collections.sources — Record."""

from domains.collections.filters import (
    LengthFilter, DedupFilter, KeywordFilter, RegexFilter, LanguageFilter,
    SamplerFilter, TransformFilter, TruncateFilter, PrefixFilter,
    MetadataFilter, FilterChain, Filter,
)
from domains.collections.sources import Record


def _record(content: str, metadata=None) -> Record:
    if metadata is not None:
        return Record(content=content, metadata=metadata)
    return Record(content=content)


# ── Record ────────────────────────────────────────────────────────────────────

class TestRecord:
    def test_defaults(self):
        r = Record(content="hello")
        assert r.content == "hello"
        assert "timestamp" in r.metadata

    def test_custom_metadata(self):
        r = Record(content="hi", metadata={"author": "test"})
        assert r.metadata["author"] == "test"

    def test_to_dict(self):
        r = Record(content="hello", metadata={"k": "v"})
        d = r.to_dict()
        assert d["content"] == "hello"
        assert d["metadata"]["k"] == "v"

    def test_to_dict_has_timestamp(self):
        r = Record(content="x")
        d = r.to_dict()
        assert "timestamp" in d["metadata"]

    def test_empty_content(self):
        r = Record(content="")
        assert r.content == ""

    def test_metadata_preserved(self):
        r = Record(content="x", metadata={"a": 1, "b": 2})
        assert r.metadata["a"] == 1
        assert r.metadata["b"] == 2

    def test_metadata_not_shared(self):
        r1 = Record(content="a")
        r2 = Record(content="b")
        r1.metadata["custom"] = True
        assert "custom" not in r2.metadata


# ── Filter protocol ──────────────────────────────────────────────────────────

class TestFilterProtocol:
    def test_length_filter_is_filter(self):
        assert isinstance(LengthFilter(), Filter)

    def test_dedup_filter_is_filter(self):
        assert isinstance(DedupFilter(), Filter)

    def test_keyword_filter_is_filter(self):
        assert isinstance(KeywordFilter(), Filter)

    def test_regex_filter_is_filter(self):
        assert isinstance(RegexFilter(), Filter)


# ── LengthFilter ──────────────────────────────────────────────────────────────

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

    def test_exact_min(self):
        lf = LengthFilter(min_length=5, max_length=10)
        assert lf.accept(_record("hello")) is True

    def test_exact_max(self):
        lf = LengthFilter(min_length=1, max_length=5)
        assert lf.accept(_record("hello")) is True

    def test_one_below_min(self):
        lf = LengthFilter(min_length=6)
        assert lf.accept(_record("hello")) is False

    def test_one_above_max(self):
        lf = LengthFilter(max_length=4)
        assert lf.accept(_record("hello")) is False

    def test_empty_string(self):
        lf = LengthFilter(min_length=0)
        assert lf.accept(_record("")) is True

    def test_default_min(self):
        lf = LengthFilter(max_length=100)
        assert lf.accept(_record("a" * 10)) is True

    def test_large_content(self):
        lf = LengthFilter(min_length=1000, max_length=100000)
        assert lf.accept(_record("x" * 1000)) is True

    def test_custom_range(self):
        lf = LengthFilter(min_length=50, max_length=60)
        assert lf.accept(_record("x" * 55)) is True
        assert lf.accept(_record("x" * 49)) is False
        assert lf.accept(_record("x" * 61)) is False


# ── DedupFilter ───────────────────────────────────────────────────────────────

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

    def test_multiple_unique(self):
        df = DedupFilter()
        assert df.accept(_record("a")) is True
        assert df.accept(_record("b")) is True
        assert df.accept(_record("c")) is True

    def test_multiple_duplicates(self):
        df = DedupFilter()
        assert df.accept(_record("a")) is True
        assert df.accept(_record("a")) is False
        assert df.accept(_record("a")) is False

    def test_similar_content(self):
        df = DedupFilter()
        assert df.accept(_record("hello")) is True
        assert df.accept(_record("Hello")) is True  # different hash due to case

    def test_empty_string_dedup(self):
        df = DedupFilter()
        assert df.accept(_record("")) is True
        assert df.accept(_record("")) is False

    def test_long_content_dedup(self):
        df = DedupFilter()
        long_text = "x" * 10000
        assert df.accept(_record(long_text)) is True
        assert df.accept(_record(long_text)) is False


# ── KeywordFilter ─────────────────────────────────────────────────────────────

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

    def test_include_case_insensitive(self):
        kf = KeywordFilter(keywords=["python"], mode="include")
        assert kf.accept(_record("PYTHON")) is True
        assert kf.accept(_record("Python")) is True

    def test_exclude_case_insensitive(self):
        kf = KeywordFilter(keywords=["spam"], mode="exclude")
        assert kf.accept(_record("SPAM")) is False

    def test_multiple_keywords_any_match(self):
        kf = KeywordFilter(keywords=["cat", "dog"], mode="include")
        assert kf.accept(_record("I have a cat")) is True
        assert kf.accept(_record("I have a dog")) is True
        assert kf.accept(_record("I have a fish")) is False

    def test_exclude_multiple_keywords(self):
        kf = KeywordFilter(keywords=["spam", "ads"], mode="exclude")
        assert kf.accept(_record("hello")) is True
        assert kf.accept(_record("this is spam")) is False
        assert kf.accept(_record("this is ads")) is False

    def test_single_keyword(self):
        kf = KeywordFilter(keywords=["test"], mode="include")
        assert kf.accept(_record("this is a test")) is True
        assert kf.accept(_record("no match")) is False

    def test_keyword_in_middle(self):
        kf = KeywordFilter(keywords=["bug"], mode="include")
        assert kf.accept(_record("there is a bug here")) is True

    def test_keyword_substring(self):
        kf = KeywordFilter(keywords=["cat"], mode="include")
        assert kf.accept(_record("category")) is True  # "cat" is substring of "category"


# ── RegexFilter ───────────────────────────────────────────────────────────────

class TestRegexFilter:
    def test_include(self):
        rf = RegexFilter(pattern=r"\b\d{3}\b", mode="include")
        assert rf.accept(_record("code 123")) is True
        assert rf.accept(_record("no numbers")) is False

    def test_exclude(self):
        rf = RegexFilter(pattern=r"test", mode="exclude")
        assert rf.accept(_record("hello")) is True
        assert rf.accept(_record("test case")) is False

    def test_empty_pattern(self):
        rf = RegexFilter(pattern="")
        assert rf.accept(_record("anything")) is True

    def test_include_case_insensitive(self):
        rf = RegexFilter(pattern=r"hello", mode="include")
        assert rf.accept(_record("HELLO")) is True
        assert rf.accept(_record("Hello World")) is True

    def test_exclude_case_insensitive(self):
        rf = RegexFilter(pattern=r"spam", mode="exclude")
        assert rf.accept(_record("SPAM")) is False

    def test_email_pattern(self):
        rf = RegexFilter(pattern=r"\b[\w.]+@[\w.]+\.\w+\b", mode="include")
        assert rf.accept(_record("contact me at test@example.com")) is True
        assert rf.accept(_record("no email here")) is False

    def test_complex_pattern(self):
        rf = RegexFilter(pattern=r"^\d{4}-\d{2}-\d{2}$", mode="include")
        assert rf.accept(_record("2024-01-15")) is True
        assert rf.accept(_record("not a date")) is False

    def test_alternation(self):
        rf = RegexFilter(pattern=r"cat|dog", mode="include")
        assert rf.accept(_record("I have a cat")) is True
        assert rf.accept(_record("I have a dog")) is True
        assert rf.accept(_record("I have a fish")) is False


# ── LanguageFilter ────────────────────────────────────────────────────────────

class TestLanguageFilter:
    def test_ascii_text(self):
        lf = LanguageFilter()
        assert lf.accept(_record("hello world")) is True

    def test_non_ascii_heavy(self):
        lf = LanguageFilter(allowed_chars_ratio=0.8)
        assert lf.accept(_record("日本語のテキスト")) is False

    def test_mixed_ascii_non_ascii(self):
        lf = LanguageFilter(allowed_chars_ratio=0.5)
        assert lf.accept(_record("hello 世界")) is True

    def test_empty_content(self):
        lf = LanguageFilter()
        assert lf.accept(_record("")) is False

    def test_all_ascii_punctuation(self):
        lf = LanguageFilter()
        assert lf.accept(_record("!@#$%^&*()")) is True

    def test_custom_ratio(self):
        lf = LanguageFilter(allowed_chars_ratio=0.3)
        assert lf.accept(_record("abc日本語日本語")) is True

    def test_pure_latin(self):
        lf = LanguageFilter(allowed_chars_ratio=1.0)
        assert lf.accept(_record("abcdef")) is True


# ── SamplerFilter ─────────────────────────────────────────────────────────────

class TestSamplerFilter:
    def test_rate_1_always_accepts(self):
        sf = SamplerFilter(rate=1.0)
        for _ in range(100):
            assert sf.accept(_record("text")) is True

    def test_rate_0_never_accepts(self):
        sf = SamplerFilter(rate=0.0)
        for _ in range(100):
            assert sf.accept(_record("text")) is False

    def test_rate_05_approximately_half(self):
        sf = SamplerFilter(rate=0.5)
        accepted = sum(1 for _ in range(1000) if sf.accept(_record("x")))
        assert 300 < accepted < 700

    def test_deterministic_sequence(self):
        sf1 = SamplerFilter(rate=0.5)
        sf2 = SamplerFilter(rate=0.5)
        results1 = [sf1.accept(_record("x")) for _ in range(10)]
        results2 = [sf2.accept(_record("x")) for _ in range(10)]
        assert results1 == results2


# ── TransformFilter ──────────────────────────────────────────────────────────

class TestTransformFilter:
    def test_always_accepts(self):
        tf = TransformFilter()
        assert tf.accept(_record("text")) is True

    def test_identity_transform(self):
        tf = TransformFilter()
        r = _record("hello")
        result = tf.transform(r)
        assert result.content == "hello"

    def test_custom_transform(self):
        tf = TransformFilter(transform_fn=lambda r: Record(content=r.content.upper()))
        result = tf.transform(_record("hello"))
        assert result.content == "HELLO"

    def test_string_return_becomes_record(self):
        tf = TransformFilter(transform_fn=lambda r: "transformed")
        result = tf.transform(_record("original"))
        assert isinstance(result, Record)
        assert result.content == "transformed"

    def test_metadata_preserved(self):
        tf = TransformFilter(transform_fn=lambda r: Record(content="new", metadata=r.metadata))
        result = tf.transform(_record("old", metadata={"k": "v"}))
        assert result.metadata["k"] == "v"

    def test_transform_fn_receives_record(self):
        received = []
        tf = TransformFilter(transform_fn=lambda r: (received.append(r), r)[1])
        r = _record("test")
        tf.transform(r)
        assert received[0] is r


# ── TruncateFilter ───────────────────────────────────────────────────────────

class TestTruncateFilter:
    def test_within_limit(self):
        tf = TruncateFilter(max_length=100)
        r = _record("hello")
        assert tf.accept(r) is True
        assert r.content == "hello"

    def test_truncates(self):
        tf = TruncateFilter(max_length=3)
        r = _record("hello")
        assert tf.accept(r) is True
        assert r.content == "hel"

    def test_exact_limit(self):
        tf = TruncateFilter(max_length=5)
        r = _record("hello")
        assert tf.accept(r) is True
        assert r.content == "hello"

    def test_empty_string(self):
        tf = TruncateFilter(max_length=10)
        r = _record("")
        assert tf.accept(r) is True
        assert r.content == ""

    def test_always_returns_true(self):
        tf = TruncateFilter(max_length=1)
        assert tf.accept(_record("long text")) is True


# ── PrefixFilter ─────────────────────────────────────────────────────────────

class TestPrefixFilter:
    def test_adds_prefix(self):
        pf = PrefixFilter(prefix="[PFX] ")
        r = _record("hello")
        assert pf.accept(r) is True
        assert r.content == "[PFX] hello"

    def test_empty_prefix(self):
        pf = PrefixFilter(prefix="")
        r = _record("hello")
        assert pf.accept(r) is True
        assert r.content == "hello"

    def test_always_returns_true(self):
        pf = PrefixFilter(prefix="x")
        assert pf.accept(_record("any")) is True

    def test_metadata_preserved(self):
        pf = PrefixFilter(prefix="PRE: ")
        r = _record("text", metadata={"k": "v"})
        pf.accept(r)
        assert r.metadata["k"] == "v"

    def test_chained_prefixes(self):
        pf1 = PrefixFilter(prefix="A")
        pf2 = PrefixFilter(prefix="B")
        r = _record("C")
        pf1.accept(r)
        pf2.accept(r)
        assert r.content == "BAC"


# ── MetadataFilter ───────────────────────────────────────────────────────────

class TestMetadataFilter:
    def test_include_match(self):
        mf = MetadataFilter(key="author", values=["alice", "bob"], mode="include")
        assert mf.accept(_record("text", metadata={"author": "alice"})) is True

    def test_include_no_match(self):
        mf = MetadataFilter(key="author", values=["alice"], mode="include")
        assert mf.accept(_record("text", metadata={"author": "charlie"})) is False

    def test_exclude_match(self):
        mf = MetadataFilter(key="author", values=["alice"], mode="exclude")
        assert mf.accept(_record("text", metadata={"author": "alice"})) is False

    def test_exclude_no_match(self):
        mf = MetadataFilter(key="author", values=["alice"], mode="exclude")
        assert mf.accept(_record("text", metadata={"author": "bob"})) is True

    def test_missing_key(self):
        mf = MetadataFilter(key="missing", values=["x"], mode="include")
        assert mf.accept(_record("text")) is False

    def test_empty_key(self):
        mf = MetadataFilter(key="", values=["x"], mode="include")
        assert mf.accept(_record("text")) is True

    def test_empty_values(self):
        mf = MetadataFilter(key="k", values=[], mode="include")
        assert mf.accept(_record("text")) is True

    def test_value_type_coercion(self):
        mf = MetadataFilter(key="count", values=["42"], mode="include")
        assert mf.accept(_record("text", metadata={"count": 42})) is True

    def test_multiple_values(self):
        mf = MetadataFilter(key="tag", values=["a", "b", "c"], mode="include")
        assert mf.accept(_record("x", metadata={"tag": "b"})) is True
        assert mf.accept(_record("x", metadata={"tag": "z"})) is False


# ── FilterChain ──────────────────────────────────────────────────────────────

class TestFilterChain:
    def test_chain(self):
        fc = FilterChain([LengthFilter(min_length=3), KeywordFilter(keywords=["hi"], mode="include")])
        assert fc.accept(_record("hi there")) is True
        assert fc.accept(_record("lo")) is False

    def test_empty_chain(self):
        fc = FilterChain()
        assert fc.accept(_record("anything")) is True

    def test_stats(self):
        fc = FilterChain([LengthFilter(min_length=10)])
        fc.accept(_record("short"))
        fc.accept(_record("a]long enough text"))
        assert fc.stats["rejected"] == 1
        assert fc.stats["accepted"] == 1

    def test_add_filter(self):
        fc = FilterChain()
        fc.add(LengthFilter(min_length=5))
        assert len(fc.filters) == 1
        fc.add(KeywordFilter(keywords=["test"], mode="include"))
        assert len(fc.filters) == 2

    def test_add_returns_self(self):
        fc = FilterChain()
        result = fc.add(LengthFilter())
        assert result is fc

    def test_filter_records(self):
        fc = FilterChain([LengthFilter(min_length=5)])
        records = [_record("hi"), _record("hello"), _record("ok"), _record("world")]
        filtered = fc.filter_records(records)
        assert len(filtered) == 2
        assert all(r.content in ("hello", "world") for r in filtered)

    def test_chain_short_circuits(self):
        """First failing filter should reject without checking later filters."""
        checked = []

        class CheckFilter:
            def accept(self, record):
                checked.append(True)
                return True

        fc = FilterChain([LengthFilter(min_length=100), CheckFilter()])
        fc.accept(_record("short"))
        assert len(checked) == 0

    def test_chain_all_pass(self):
        fc = FilterChain([
            LengthFilter(min_length=1),
            KeywordFilter(keywords=["hello"], mode="include"),
        ])
        assert fc.accept(_record("hello world")) is True

    def test_chain_reject_first(self):
        fc = FilterChain([
            LengthFilter(min_length=100),
            KeywordFilter(keywords=["hello"], mode="include"),
        ])
        assert fc.accept(_record("hi")) is False

    def test_chain_reject_second(self):
        fc = FilterChain([
            LengthFilter(min_length=1),
            KeywordFilter(keywords=["spam"], mode="exclude"),
        ])
        assert fc.accept(_record("this is spam")) is False

    def test_filter_records_empty(self):
        fc = FilterChain([LengthFilter(min_length=1)])
        assert fc.filter_records([]) == []

    def test_stats_accumulate(self):
        fc = FilterChain([LengthFilter(min_length=3)])
        fc.accept(_record("ab"))
        fc.accept(_record("abc"))
        fc.accept(_record("abcd"))
        assert fc.stats["rejected"] == 1
        assert fc.stats["accepted"] == 2
