"""Tests for domains.learner.data_filter: quality gating and topic filtering."""

import json

import pytest

from domains.learner import data_filter as df

GOOD_TEXT = (
    "The quick brown fox jumps over the lazy dog. It runs through the forest "
    "and into the meadow. The weather is pleasant today. Birds sing their songs. "
    "Farmers harvest the golden wheat in the warm afternoon sun. Children play "
    "near the sparkling stream while their parents watch from the porch. "
    "Everyone enjoys the peaceful afternoon in the countryside."
)
BAD_TEXT = "ZZZZZZZZZZ\n" * 10


@pytest.fixture
def tmp_config(tmp_path, monkeypatch):
    path = tmp_path / "filter_config.json"
    monkeypatch.setattr(df, "FILTER_CONFIG_PATH", path)
    return path


@pytest.fixture
def filt(tmp_config):
    return df.DataFilter()


class TestScoreQuality:
    def test_short_text_zero(self):
        assert df._score_quality("short") == 0.0

    def test_empty_zero(self):
        assert df._score_quality("") == 0.0

    def test_good_prose_scores_high(self):
        assert df._score_quality(GOOD_TEXT) >= 0.5

    def test_repetitive_text_scores_low(self):
        assert df._score_quality(BAD_TEXT) < 0.3

    def test_varied_beats_repetitive(self):
        assert df._score_quality(GOOD_TEXT) > df._score_quality("aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa " * 5)

    def test_mixed_case_beats_all_caps(self):
        mixed = "This is a normal sentence. Another one here. A third sentence follows."
        caps = "THIS IS A NORMAL SENTENCE. ANOTHER ONE HERE. A THIRD SENTENCE FOLLOWS."
        assert df._score_quality(mixed) > df._score_quality(caps)


class TestScoreRelevance:
    def test_empty_whitelist_is_max(self):
        assert df._score_relevance("anything", []) == 1.0

    def test_no_match_is_zero(self):
        assert df._score_relevance("nothing relevant here", ["space"]) == 0.0

    def test_many_matches_caps_at_one(self):
        text = "ai " * 60
        assert df._score_relevance(text, ["ai"]) == 1.0

    def test_longer_topics_weigh_more(self):
        one = df._score_relevance("machine learning is machine learning", ["machine learning"])
        two = df._score_relevance("machine learning is machine learning", ["machine"])
        assert one > two

    def test_case_insensitive(self):
        assert df._score_relevance("AI at the conference", ["ai"]) > 0.0


class TestMatchesBlacklist:
    def test_blacklisted_term(self):
        assert df._matches_blacklist("buy viagra now", ["viagra"]) is True

    def test_multiword_term(self):
        assert df._matches_blacklist("you won free money", ["free money"]) is True

    def test_no_match(self):
        assert df._matches_blacklist("clean article text", ["porn"]) is False

    def test_case_insensitive(self):
        assert df._matches_blacklist("PORN content", ["porn"]) is True


class TestMatchesWhitelist:
    def test_empty_whitelist_passes(self):
        assert df._matches_whitelist("anything", []) is True

    def test_match(self):
        assert df._matches_whitelist("about artificial intelligence", ["artificial"]) is True

    def test_no_match(self):
        assert df._matches_whitelist("about cooking", ["artificial"]) is False


class TestConfig:
    def test_load_default_when_missing(self, tmp_config):
        assert df._load_config() == df.DEFAULT_CONFIG

    def test_load_from_file(self, tmp_config):
        tmp_config.write_text(json.dumps({"min_content_length": 123}))
        cfg = df._load_config()
        assert cfg == {"min_content_length": 123}

    def test_load_corrupt_falls_back_to_default(self, tmp_config):
        tmp_config.write_text("{not json")
        assert df._load_config() == df.DEFAULT_CONFIG

    def test_save_config_writes_file(self, tmp_config):
        df._save_config({"a": 1})
        assert json.loads(tmp_config.read_text()) == {"a": 1}

    def test_init_merges_and_persists(self, tmp_config):
        d = df.DataFilter(config={"min_content_length": 50, "min_quality_score": 0.5})
        assert d.config["min_content_length"] == 50
        assert d.config["min_quality_score"] == 0.5
        assert d.config["enabled"] is True
        assert json.loads(tmp_config.read_text())["min_content_length"] == 50

    def test_get_config_returns_copy(self, filt):
        cfg = filt.get_config()
        cfg["enabled"] = False
        assert filt.config["enabled"] is True

    def test_update_config(self, filt, tmp_config):
        filt.update_config(min_content_length=999)
        assert filt.config["min_content_length"] == 999
        assert json.loads(tmp_config.read_text())["min_content_length"] == 999


class TestStats:
    def test_initial_stats(self, filt):
        s = filt.get_stats()
        assert s["total_seen"] == 0
        assert s["passed"] == 0
        assert s["rejected"] == 0

    def test_get_stats_returns_copy(self, filt):
        s = filt.get_stats()
        s["passed"] = 99
        assert filt.stats["passed"] == 0


class TestFilterArticle:
    def test_disabled_passes_without_counting_seen(self, filt):
        filt.update_config(enabled=False)
        ok, reason = filt.filter_article("http://x", "t", "short")
        assert (ok, reason) == (True, "")
        assert filt.stats["passed"] == 1
        assert filt.stats["total_seen"] == 0

    def test_too_short(self, filt):
        ok, reason = filt.filter_article("http://x", "t", "way too short")
        assert (ok, reason) == (False, "too_short")
        assert filt.stats["rejected_short"] == 1

    def test_low_quality(self, filt):
        filt.update_config(min_content_length=0)
        ok, reason = filt.filter_article("http://x", "t", BAD_TEXT)
        assert ok is False
        assert reason.startswith("low_quality")
        assert filt.stats["rejected_quality"] == 1

    def test_blacklisted(self, filt):
        filt.update_config(min_content_length=0)
        ok, reason = filt.filter_article("http://x", "spam", "gambling offers inside " * 20)
        assert ok is False
        assert reason == "blacklisted"
        assert filt.stats["rejected_blacklist"] == 1

    def test_whitelist_hard_gate_rejects(self, filt):
        filt.update_config(min_content_length=0, topic_whitelist=["artificial"], whitelist_is_hard_gate=True)
        ok, reason = filt.filter_article("http://x", "cooking", "how to cook pasta properly today " * 20)
        assert ok is False
        assert reason == "not_in_whitelist"
        assert filt.stats["rejected_whitelist"] == 1

    def test_whitelist_not_hard_gate_passes(self, filt):
        filt.update_config(min_content_length=0, topic_whitelist=["artificial"], whitelist_is_hard_gate=False)
        ok, reason = filt.filter_article("http://x", "cooking", "how to cook pasta properly today " * 20)
        assert (ok, reason) == (True, "")

    def test_whitelist_match_passes(self, filt):
        filt.update_config(min_content_length=0, topic_whitelist=["artificial"], whitelist_is_hard_gate=True)
        ok, reason = filt.filter_article("http://x", "ai news", "artificial intelligence advances today " * 20)
        assert (ok, reason) == (True, "")

    def test_near_duplicate_rejected(self, filt):
        filt.update_config(min_content_length=0)
        ok, reason = filt.filter_article("http://x", "t", GOOD_TEXT, existing_facts=[GOOD_TEXT])
        assert ok is False
        assert reason == "near_duplicate"
        assert filt.stats["rejected_dup"] == 1

    def test_distinct_content_passes_dup_check(self, filt):
        filt.update_config(min_content_length=0)
        other = "Completely different text about astronomy and planets today. " * 20
        ok, reason = filt.filter_article("http://x", "t", GOOD_TEXT, existing_facts=[other])
        assert (ok, reason) == (True, "")

    def test_good_article_passes(self, filt):
        ok, reason = filt.filter_article("http://x", "t", GOOD_TEXT)
        assert (ok, reason) == (True, "")
        assert filt.stats["passed"] == 1
        assert filt.stats["total_seen"] == 1

    def test_stats_rejected_incremented(self, filt):
        filt.filter_article("http://x", "t", "short")
        assert filt.stats["rejected"] == 1


class TestFilterChunk:
    def test_disabled(self, filt):
        filt.update_config(enabled=False)
        assert filt.filter_chunk("anything", "topic") is True

    def test_blacklist_rejects(self, filt):
        assert filt.filter_chunk("this mentions gambling", "topic") is False

    def test_hard_gate_whitelist_rejects(self, filt):
        filt.update_config(topic_whitelist=["space"], whitelist_is_hard_gate=True)
        assert filt.filter_chunk("about cooking pasta", "cooking") is False

    def test_hard_gate_whitelist_pass(self, filt):
        filt.update_config(topic_whitelist=["space"], whitelist_is_hard_gate=True)
        assert filt.filter_chunk("about space travel", "space") is True

    def test_no_gates_passes(self, filt):
        assert filt.filter_chunk("normal chunk of text here", "general") is True


class TestHashed:
    def test_deterministic(self):
        assert df._hashed(0.5) == df._hashed(0.5)

    def test_in_unit_interval(self):
        for n in [0.0, 1.0, 0.1, 42.0, 3.14159]:
            v = df._hashed(n)
            assert 0.0 <= v < 1.0

    def test_distinct_inputs_differ(self):
        assert df._hashed(0.1) != df._hashed(0.2)


class TestSingleton:
    def test_get_data_filter_singleton(self, monkeypatch, tmp_config):
        monkeypatch.setattr(df, "_filter", None)
        f1 = df.get_data_filter()
        f2 = df.get_data_filter()
        assert f1 is f2
        monkeypatch.setattr(df, "_filter", None)
