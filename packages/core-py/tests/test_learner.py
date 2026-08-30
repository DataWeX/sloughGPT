"""Tests for learner domain — pure logic functions, dataclasses, helpers, and chunking strategies."""

import hashlib
import math
import re
import time
from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ── continual.py: tokenize / detokenize ─────────────────────────────────

from domains.learner.continual import (
    _tokenize,
    _detokenize,
    CHAR_SET,
    STOI,
    ITOS,
    UNK,
    VOCAB,
    TRAIN_SEQ_LEN,
    BUFFER_CAPACITY,
    INGEST_THRESHOLD,
)


class TestTokenize:
    def test_simple_text(self):
        ids = _tokenize("hello")
        assert all(isinstance(i, int) for i in ids)
        assert len(ids) == 5

    def test_all_known_chars(self):
        for c in CHAR_SET:
            ids = _tokenize(c)
            assert len(ids) == 1

    def test_unknown_chars_mapped_to_unk(self):
        ids = _tokenize("@#$%")
        assert all(i == UNK for i in ids)

    def test_uppercase_to_lowercase(self):
        ids = _tokenize("ABC")
        lower_ids = _tokenize("abc")
        assert ids == lower_ids

    def test_empty_string(self):
        assert _tokenize("") == []

    def test_vocab_size(self):
        assert VOCAB == len(CHAR_SET)

    def test_stoi_itos_consistent(self):
        for i, c in enumerate(CHAR_SET):
            assert STOI[c] == i
            assert ITOS[i] == c


class TestDetokenize:
    def test_roundtrip(self):
        original = "hello world"
        ids = _tokenize(original)
        result = _detokenize(ids)
        assert result == original

    def test_unknown_id_replaced(self):
        result = _detokenize([9999])
        assert result == "?"

    def test_empty_ids(self):
        assert _detokenize([]) == ""

    def test_all_chars_recoverable(self):
        for c in CHAR_SET:
            ids = [STOI[c]]
            assert _detokenize(ids) == c


# ── knowledge.py: dataclasses ───────────────────────────────────────────

from domains.learner.knowledge import (
    KnowledgeFact,
    FeedSubscription,
    _topic_slug,
    chunk_by_fixed_size,
    chunk_by_paragraph,
    chunk_by_heading,
    chunk_by_semantic,
    chunk_text,
    _extract_facts_from_text,
    _extract_topics,
    DEFAULT_FEED_POLL_INTERVAL,
)


class TestKnowledgeFact:
    def test_defaults(self):
        fact = KnowledgeFact(content="test content")
        assert fact.content == "test content"
        assert fact.topic == "general"
        assert fact.source == "manual"
        assert fact.url == ""
        assert fact.timestamp == 0.0
        assert fact.importance == 0.5

    def test_custom_fields(self):
        fact = KnowledgeFact(
            content="AI is advancing",
            topic="technology",
            source="web",
            url="https://example.com",
            timestamp=1234567890.0,
            importance=0.9,
        )
        assert fact.topic == "technology"
        assert fact.importance == 0.9

    def test_asdict(self):
        fact = KnowledgeFact(content="test")
        d = asdict(fact)
        assert isinstance(d, dict)
        assert d["content"] == "test"

    def test_immutable_by_value(self):
        fact = KnowledgeFact(content="test")
        fact2 = KnowledgeFact(content="test")
        assert fact.content == fact2.content


class TestFeedSubscription:
    def test_defaults(self):
        feed = FeedSubscription(url="https://example.com/rss")
        assert feed.url == "https://example.com/rss"
        assert feed.title == ""
        assert feed.last_fetched == 0.0
        assert feed.poll_interval == DEFAULT_FEED_POLL_INTERVAL
        assert feed.enabled is True

    def test_custom_fields(self):
        feed = FeedSubscription(
            url="https://example.com/rss",
            title="My Feed",
            last_fetched=100.0,
            poll_interval=600,
            enabled=False,
        )
        assert feed.title == "My Feed"
        assert feed.poll_interval == 600
        assert feed.enabled is False

    def test_asdict(self):
        feed = FeedSubscription(url="https://example.com/rss")
        d = asdict(feed)
        assert "url" in d
        assert d["url"] == "https://example.com/rss"


class TestTopicSlug:
    def test_simple(self):
        assert _topic_slug("Machine Learning") == "machine_learning"

    def test_special_chars(self):
        result = _topic_slug("AI/ML & Deep Learning!")
        assert result.isalnum() or "_" in result

    def test_truncation(self):
        long_topic = "a" * 100
        result = _topic_slug(long_topic)
        assert len(result) <= 64

    def test_empty_string(self):
        result = _topic_slug("")
        assert isinstance(result, str)

    def test_whitespace_stripped(self):
        result = _topic_slug("  AI  ")
        assert result == "ai"


# ── knowledge.py: chunking strategies ────────────────────────────────────

class TestChunkByFixedSize:
    def test_empty_text(self):
        assert chunk_by_fixed_size("") == []
        assert chunk_by_fixed_size("   ") == []

    def test_short_text(self):
        result = chunk_by_fixed_size("hello", chunk_size=100)
        assert result == ["hello"]

    def test_exact_boundary(self):
        text = "a" * 100
        result = chunk_by_fixed_size(text, chunk_size=100)
        assert len(result) == 1

    def test_longer_than_chunk(self):
        text = "a" * 150
        result = chunk_by_fixed_size(text, chunk_size=100)
        assert len(result) >= 2

    def test_overlap(self):
        text = "a" * 200
        result = chunk_by_fixed_size(text, chunk_size=100, overlap=50)
        assert len(result) >= 2

    def test_whitespace_chunks_filtered(self):
        text = "hello   world   foo"
        result = chunk_by_fixed_size(text, chunk_size=5)
        for chunk in result:
            assert chunk.strip()

    def test_no_overlap(self):
        text = "abcdefghij" * 10
        result = chunk_by_fixed_size(text, chunk_size=10, overlap=0)
        assert len(result) == 10


class TestChunkByParagraph:
    def test_empty_text(self):
        assert chunk_by_paragraph("") == []
        assert chunk_by_paragraph("   ") == []

    def test_single_paragraph(self):
        result = chunk_by_paragraph("hello world")
        assert len(result) == 1
        assert "hello world" in result[0]

    def test_multiple_paragraphs(self):
        text = "para1\n\npara2\n\npara3"
        result = chunk_by_paragraph(text)
        assert len(result) >= 1

    def test_long_paragraphs_merge(self):
        text = ("short\n\n" * 20).strip()
        result = chunk_by_paragraph(text, max_chunk_size=200)
        assert len(result) >= 1

    def test_long_paragraphs_split(self):
        text = ("a" * 100 + "\n\n") * 5
        result = chunk_by_paragraph(text, max_chunk_size=150)
        assert len(result) >= 2


class TestChunkByHeading:
    def test_empty_text(self):
        assert chunk_by_heading("") == []
        assert chunk_by_heading("   ") == []

    def test_no_headings(self):
        text = "just some text without headings"
        result = chunk_by_heading(text)
        assert len(result) >= 1

    def test_with_headings(self):
        text = "# Title\nContent under title\n## Subtitle\nMore content"
        result = chunk_by_heading(text)
        assert len(result) >= 1

    def test_heading_based_split(self):
        sec1 = "Content one sentence. " * 80
        sec2 = "Content two sentence. " * 80
        text = f"# Section One\n{sec1}\n\n# Section Two\n{sec2}"
        result = chunk_by_heading(text, max_chunk_size=500)
        assert len(result) >= 2

    def test_merge_small_sections(self):
        text = "# A\nshort\n# B\nshort\n# C\nshort"
        result = chunk_by_heading(text, max_chunk_size=1000)
        assert len(result) >= 1


class TestChunkBySemantic:
    def test_empty_text(self):
        assert chunk_by_semantic("") == []

    def test_short_text(self):
        result = chunk_by_semantic("hello world")
        assert len(result) == 1

    def test_multiple_sentences(self):
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        result = chunk_by_semantic(text)
        assert len(result) >= 1

    def test_sentence_grouping(self):
        sentences = [f"This is sentence number {i} with some content." for i in range(10)]
        text = " ".join(sentences)
        result = chunk_by_semantic(text, max_chunk_size=200, min_chunk_size=50)
        assert len(result) >= 1

    def test_topic_shift_splits(self):
        text = (
            "Machine learning is a subset of artificial intelligence. "
            "Neural networks are inspired by biological neurons. "
            "Cooking requires heat and ingredients. "
            "Recipes often include detailed instructions for preparation."
        )
        result = chunk_by_semantic(text, min_chunk_size=30)
        assert len(result) >= 1


class TestChunkText:
    def test_auto_strategy_heading(self):
        text = "# Title\nContent here"
        result = chunk_text(text, strategy="auto")
        assert len(result) >= 1

    def test_auto_strategy_paragraph(self):
        text = "\n\n".join(["Paragraph " * 20] * 5)
        result = chunk_text(text, strategy="auto")
        assert len(result) >= 1

    def test_fixed_strategy(self):
        text = "a" * 200
        result = chunk_text(text, strategy="fixed", chunk_size=50)
        assert len(result) >= 3

    def test_paragraph_strategy(self):
        text = "para1\n\npara2"
        result = chunk_text(text, strategy="paragraph")
        assert len(result) >= 1

    def test_heading_strategy(self):
        text = "# H1\nContent\n# H2\nMore"
        result = chunk_text(text, strategy="heading")
        assert len(result) >= 1

    def test_semantic_strategy(self):
        sentences = [f"Sentence {i} with some meaningful content here." for i in range(10)]
        text = " ".join(sentences)
        result = chunk_text(text, strategy="semantic")
        assert len(result) >= 1

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            chunk_text("hello", strategy="unknown")


# ── knowledge.py: fact extraction heuristics ─────────────────────────────

class TestExtractFactsFromText:
    def test_empty_text(self):
        assert _extract_facts_from_text("") == []

    def test_short_text(self):
        assert _extract_facts_from_text("too short") == []

    def test_question_excluded(self):
        text = "What is the meaning of life? " * 5
        assert _extract_facts_from_text(text) == []

    def test_declarative_fact(self):
        text = "The Eiffel Tower is a famous landmark in Paris, France. It was built in 1889 for the World Fair exhibition."
        facts = _extract_facts_from_text(text)
        assert len(facts) >= 1

    def test_numbers_indicate_facts(self):
        text = "The population exceeds 10 million people and the GDP grew by 3.5 percent annually."
        facts = _extract_facts_from_text(text)
        assert len(facts) >= 1

    def test_imperative_excluded(self):
        text = "Try to make sure you remember the steps for this process carefully. " * 3
        facts = _extract_facts_from_text(text)
        assert len(facts) == 0

    def test_exclamatory_excluded(self):
        text = "This is amazing news! " * 5
        facts = _extract_facts_from_text(text)
        assert len(facts) == 0


class TestExtractTopics:
    def test_empty_text(self):
        assert _extract_topics("") == []

    def test_single_topic(self):
        topics = _extract_topics("machine learning algorithms and neural networks")
        assert len(topics) <= 5
        assert all(isinstance(t, str) for t in topics)

    def test_stopwords_excluded(self):
        topics = _extract_topics("the and for are but not you all can")
        for t in topics:
            assert t not in {"the", "and", "for", "are", "but", "not", "you", "all", "can"}

    def test_max_topics(self):
        text = " ".join([f"topic{i}word{i}" for i in range(20)])
        topics = _extract_topics(text, max_topics=3)
        assert len(topics) <= 3


# ── data_filter.py: helper functions ─────────────────────────────────────

from domains.learner.data_filter import (
    _score_quality,
    _score_relevance,
    _matches_blacklist,
    _matches_whitelist,
    _hashed,
    DEFAULT_CONFIG,
)


class TestScoreQuality:
    def test_empty_text(self):
        assert _score_quality("") == 0.0

    def test_short_text(self):
        assert _score_quality("hi") == 0.0

    def test_good_prose(self):
        text = (
            "The quick brown fox jumps over the lazy dog. "
            "This is a well-structured sentence with proper punctuation. "
            "Another sentence follows with more detail and context. "
            "And a fourth sentence to complete the paragraph nicely."
        )
        score = _score_quality(text)
        assert 0.0 <= score <= 1.0
        assert score > 0.3

    def test_caps_heavy_penalized(self):
        text = "THIS IS ALL CAPS TEXT SHOUTING LOUDLY " * 10
        score = _score_quality(text)
        assert score < 0.5

    def test_listicle_penalized(self):
        lines = "\n".join([f"item {i}" for i in range(20)])
        score = _score_quality(lines)
        assert score < 0.8

    def test_score_range(self):
        text = "This is a normal sentence with some content. " * 10
        score = _score_quality(text)
        assert 0.0 <= score <= 1.0


class TestScoreRelevance:
    def test_no_whitelist(self):
        assert _score_relevance("any text", []) == 1.0

    def test_matching_whitelist(self):
        text = "machine learning and deep learning are related"
        score = _score_relevance(text, ["machine learning"])
        assert score > 0.0

    def test_no_match(self):
        text = "cooking recipes are delicious"
        score = _score_relevance(text, ["quantum physics"])
        assert score == 0.0

    def test_multiple_matches(self):
        text = "AI and machine learning with deep learning"
        score = _score_relevance(text, ["machine learning", "deep learning"])
        assert score > 0.5


class TestMatchesBlacklist:
    def test_no_match(self):
        assert _matches_blacklist("hello world", ["porn", "gambling"]) is False

    def test_match(self):
        assert _matches_blacklist("visit our gambling site", ["gambling"]) is True

    def test_case_insensitive(self):
        assert _matches_blacklist("GAMBLING", ["gambling"]) is True

    def test_empty_blacklist(self):
        assert _matches_blacklist("anything", []) is False


class TestMatchesWhitelist:
    def test_empty_whitelist(self):
        assert _matches_whitelist("anything", []) is True

    def test_match(self):
        assert _matches_whitelist("machine learning is great", ["machine learning"]) is True

    def test_no_match(self):
        assert _matches_whitelist("cooking is fun", ["quantum physics"]) is False

    def test_case_insensitive(self):
        assert _matches_whitelist("MACHINE LEARNING", ["machine learning"]) is True


class TestHashed:
    def test_deterministic(self):
        assert _hashed(1.0) == _hashed(1.0)

    def test_different_inputs_different_outputs(self):
        assert _hashed(1.0) != _hashed(2.0)

    def test_output_range(self):
        for n in range(100):
            result = _hashed(float(n))
            assert 0.0 <= result < 1.0


class TestDefaultConfig:
    def test_has_required_keys(self):
        required = [
            "min_content_length", "min_quality_score", "min_relevance_score",
            "topic_whitelist", "topic_blacklist", "whitelist_is_hard_gate",
            "dup_similarity_threshold", "enabled",
        ]
        for key in required:
            assert key in DEFAULT_CONFIG

    def test_sensible_defaults(self):
        assert DEFAULT_CONFIG["min_content_length"] > 0
        assert 0 <= DEFAULT_CONFIG["min_quality_score"] <= 1
        assert 0 <= DEFAULT_CONFIG["min_relevance_score"] <= 1
        assert DEFAULT_CONFIG["enabled"] is True


# ── knowledge_augmenter.py: helper functions ──────────────────────────────

from domains.learner.knowledge_augmenter import (
    _is_casual_small_talk,
    _needs_web_search,
    _content_tokens,
    _topically_related,
    MIN_RELEVANCE_SCORE,
    _MIN_CONTENT_TOKEN_LEN,
    _QUERY_SIGNALS,
)


class TestIsCasualSmallTalk:
    def test_hello(self):
        assert _is_casual_small_talk("Hello!") is True

    def test_hi(self):
        assert _is_casual_small_talk("Hi") is True

    def test_hey(self):
        assert _is_casual_small_talk("Hey there") is True

    def test_how_are_you(self):
        assert _is_casual_small_talk("How are you?") is True

    def test_good_morning(self):
        assert _is_casual_small_talk("Good morning!") is True

    def test_nice_to_meet(self):
        assert _is_casual_small_talk("Nice to meet you") is True

    def test_not_casual(self):
        assert _is_casual_small_talk("What is machine learning?") is False

    def test_empty_string(self):
        assert _is_casual_small_talk("") is False

    def test_whats_up(self):
        assert _is_casual_small_talk("What's up?") is True


class TestNeedsWebSearch:
    def test_what_query(self):
        assert _needs_web_search("What is the latest news?") is True

    def test_how_query(self):
        assert _needs_web_search("How do I train a model?") is True

    def test_latest_signal(self):
        assert _needs_web_search("Latest developments in AI") is True

    def test_casual_no_search(self):
        assert _needs_web_search("Hello") is False

    def test_general_text_no_search(self):
        assert _needs_web_search("I like cats") is False

    def test_compare_signal(self):
        assert _needs_web_search("Compare React and Vue") is True


class TestContentTokens:
    def test_basic(self):
        tokens = _content_tokens("the quick brown fox jumps")
        assert "quick" in tokens
        assert "brown" in tokens

    def test_short_words_excluded(self):
        tokens = _content_tokens("is the of a to")
        assert len(tokens) == 0

    def test_mixed_lengths(self):
        tokens = _content_tokens("is machine learning advanced")
        assert "machine" in tokens
        assert "learning" in tokens
        assert "is" not in tokens

    def test_empty(self):
        assert _content_tokens("") == set()

    def test_case_insensitive(self):
        tokens = _content_tokens("MACHINE Learning")
        assert "machine" in tokens
        assert "learning" in tokens


class TestTopicallyRelated:
    def test_related(self):
        assert _topically_related("train a neural network", "neural network training") is True

    def test_unrelated(self):
        assert _topically_related("what color is the sky", "cooking pasta recipes") is False

    def test_partial_overlap(self):
        assert _topically_related("deep learning models", "learning algorithms") is True

    def test_no_content_tokens(self):
        assert _topically_related("is the a", "of to for") is False


class TestConstants:
    def test_min_relevance_score(self):
        assert 0.0 <= MIN_RELEVANCE_SCORE <= 1.0

    def test_min_content_token_len(self):
        assert _MIN_CONTENT_TOKEN_LEN >= 1

    def test_query_signals_populated(self):
        assert len(_QUERY_SIGNALS) > 0
        assert all(isinstance(s, str) for s in _QUERY_SIGNALS)


# ── knowledge_ops.py: DuplicateDetector ──────────────────────────────────

from domains.learner.knowledge_ops import (
    DuplicateDetector,
    SmartContextInjector,
    KnowledgeGapDetector,
)


class TestDuplicateDetector:
    def test_no_store(self):
        dup = DuplicateDetector(threshold=0.85)
        is_dup, match, score = dup.check("test text")
        assert is_dup is False
        assert match is None
        assert score == 0.0

    def test_threshold_default(self):
        dup = DuplicateDetector()
        assert dup._threshold == 0.85

    def test_custom_threshold(self):
        dup = DuplicateDetector(threshold=0.9)
        assert dup._threshold == 0.9


class TestSmartContextInjector:
    def test_no_memory(self):
        injector = SmartContextInjector(knowledge_memory=None)
        context = injector.get_context("test message")
        assert context == ""

    def test_should_inject_no_memory(self):
        injector = SmartContextInjector(knowledge_memory=None)
        assert injector.should_inject("test") is False

    def test_get_context_for_system_no_memory(self):
        injector = SmartContextInjector(knowledge_memory=None)
        result = injector.get_context_for_system("test", "system prompt")
        assert result == "system prompt"

    def test_config(self):
        injector = SmartContextInjector(min_score=0.5, max_facts=3)
        assert injector._min_score == 0.5
        assert injector._max_facts == 3


class TestKnowledgeGapDetector:
    def test_no_store(self):
        gap = KnowledgeGapDetector()
        gaps = gap.find_gaps(seed_topics=["security"])
        assert gaps == []

    def test_empty_topic_counts(self):
        gap = KnowledgeGapDetector()
        gap._store = MagicMock()
        gaps = gap.find_gaps()
        assert gaps == []


# ── data_filter.py: DataFilter class (init with mock DB) ──────────────────

from domains.learner.data_filter import DataFilter, set_data_filter_db, reset_data_filter_db


@pytest.fixture(autouse=True)
def isolate_data_filter(tmp_path):
    """Use a temp DB path for each test to avoid polluting real DB."""
    db_path = str(tmp_path / "test_filter_db")
    set_data_filter_db(db_path)
    yield
    reset_data_filter_db()


class TestDataFilter:
    def test_init(self):
        f = DataFilter()
        assert f.config is not None
        assert "enabled" in f.config

    def test_get_config(self):
        f = DataFilter()
        cfg = f.get_config()
        assert isinstance(cfg, dict)
        assert cfg["enabled"] is True

    def test_get_stats(self):
        f = DataFilter()
        stats = f.get_stats()
        assert "total_seen" in stats
        assert stats["total_seen"] == 0

    def test_filter_short_content(self):
        f = DataFilter()
        passed, reason = f.filter_article("http://x.com", "title", "short")
        assert passed is False
        assert reason == "too_short"

    def test_filter_low_quality(self):
        f = DataFilter()
        content = "\n".join(["x"] * 300)
        passed, reason = f.filter_article("http://x.com", "title", content)
        assert passed is False
        assert "low_quality" in reason

    def test_filter_disabled(self):
        f = DataFilter(config={"enabled": False})
        passed, reason = f.filter_article("http://x.com", "title", "short")
        assert passed is True

    def test_filter_blacklisted(self):
        f = DataFilter(config={"min_content_length": 10, "min_quality_score": 0.0})
        content = "This is about gambling and casino games. " * 20
        passed, reason = f.filter_article("http://x.com", "gambling article", content)
        assert passed is False
        assert reason == "blacklisted"

    def test_filter_near_duplicate(self):
        f = DataFilter(config={
            "min_content_length": 50,
            "min_quality_score": 0.0,
            "dup_similarity_threshold": 0.5,
        })
        content = "Machine learning is a branch of artificial intelligence that focuses on algorithms. " * 5
        existing = ["Machine learning is a branch of artificial intelligence that focuses on algorithms. " * 5]
        passed, reason = f.filter_article("http://x.com", "ML", content, existing_facts=existing)
        assert passed is False
        assert reason == "near_duplicate"

    def test_filter_chunk_disabled(self):
        f = DataFilter(config={"enabled": False})
        assert f.filter_chunk("any chunk", "any topic") is True

    def test_filter_chunk_blacklisted(self):
        f = DataFilter()
        assert f.filter_chunk("this is about gambling", "topic") is False

    def test_filter_chunk_passes(self):
        f = DataFilter()
        assert f.filter_chunk("normal content about technology", "tech") is True

    def test_stats_tracked(self):
        f = DataFilter(config={"min_content_length": 10, "min_quality_score": 0.0})
        f.filter_article("http://x.com", "t", "short")
        stats = f.get_stats()
        assert stats["total_seen"] == 1
        assert stats["rejected"] == 1
        assert stats["rejected_short"] == 1

    def test_update_config(self):
        f = DataFilter()
        f.update_config(min_content_length=500)
        assert f.config["min_content_length"] == 500


# ── continual.py: constants ──────────────────────────────────────────────

class TestContinualConstants:
    def test_buffer_capacity(self):
        assert BUFFER_CAPACITY == 10000

    def test_ingest_threshold(self):
        assert INGEST_THRESHOLD == 512

    def test_train_seq_len(self):
        assert TRAIN_SEQ_LEN == 32
