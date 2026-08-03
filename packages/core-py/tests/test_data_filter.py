"""Tests for domains/learner/data_filter.py — quality scoring, config, gates."""

import json

import pytest

from domains import learner
from domains.learner import data_filter as df
from domains.learner.data_filter import (
    DataFilter,
    DEFAULT_CONFIG,
    _hashed,
    _score_quality,
    _score_relevance,
    _matches_blacklist,
    _matches_whitelist,
    _load_config,
    _save_config,
    get_data_filter,
)


GOOD_ARTICLE = (
    "The oak stood by the river, its limbs bare. Farmers picked ripe apples, "
    "careful with soft fruit. Children played by the stream, laughing at "
    "splashes. Writers built long stories, weaving rich detail. Scientists "
    "watched ocean currents, noting each anomaly. Travelers searched distant "
    "lands, hunting treasure."
)


@pytest.fixture
def config_path(tmp_path, monkeypatch):
    path = tmp_path / "filter_config.json"
    monkeypatch.setattr(df, "FILTER_CONFIG_PATH", path)
    return path


@pytest.fixture
def fresh_singleton(monkeypatch):
    monkeypatch.setattr(df, "_filter", None)


# ── helpers ────────────────────────────────────────────────────────────


class TestHashed:
    def test_deterministic(self):
        assert _hashed(3.14) == _hashed(3.14)

    def test_in_unit_range(self):
        assert 0.0 <= _hashed(42.0) < 1.0


class TestScoreQuality:
    def test_short_text_is_zero(self):
        assert _score_quality("hi") == 0.0

    def test_good_prose_scores_high(self):
        assert _score_quality(GOOD_ARTICLE) > 0.7

    def test_caps_heavy_scores_low(self):
        text = (
            "THIS IS ALL CAPS TEXT WITH MANY UPPERCASE LETTERS. THIS SHOULD "
            "GET A LOW QUALITY SCORE. BECAUSE IT LOOKS LIKE SHOUTING AT THE "
            "READER ALL THE TIME WITHOUT ANY REASON AT ALL."
        )
        assert _score_quality(text) < _score_quality(GOOD_ARTICLE)

    def test_listicle_short_lines_penalized(self):
        text = "Nav\nMenu\nHome\nAbout\nPricing\nContact\nBlog\nShop\nCart\nLogin\nSign up\n" + GOOD_ARTICLE
        assert _score_quality(text) < _score_quality(GOOD_ARTICLE)

    def test_long_words_penalized(self):
        text = (
            "Pneumonoultramicroscopicsilicovolcanoconiosis "
            "supercalifragilisticexpialidocious antidisestablishmentarianism "
            "floccinaucinihilipilification honorificabilitudinitatibus"
        )
        assert _score_quality(text) < 0.6

    def test_mid_avg_word_length_scores_half(self):
        text = "the cat and the dog ran up the big red hill for fun " * 3
        assert _score_quality(text) < _score_quality(GOOD_ARTICLE)

    def test_mid_caps_ratio_scores_half(self):
        text = "ABCDEFGHIJ ABCDEFGHIJ abcdefgh " + "abcd " * 12
        assert _score_quality(text) < _score_quality(GOOD_ARTICLE)

    def test_mid_unique_ratio_scores_half(self):
        text = "aaaa bbbb cccc dddd eeee " * 3
        assert _score_quality(text) < _score_quality(GOOD_ARTICLE)

    def test_low_unique_ratio_penalized(self):
        text = "aaaa bbbb aaaa bbbb aaaa bbbb aaaa bbbb aaaa bbbb aaaa bbbb aaaa bbbb aaaa bbbb"
        assert _score_quality(text) < 0.6


class TestScoreRelevance:
    def test_no_whitelist_means_fully_relevant(self):
        assert _score_relevance("anything at all", []) == 1.0

    def test_matches_weight_by_length(self):
        text = (
            "Artificial Intelligence is changing python programming. "
            "Artificial Intelligence research advances daily in the field."
        )
        score = _score_relevance(text, ["artificial intelligence", "python"])
        assert score == 1.0

    def test_partial_match_scores_less(self):
        score = _score_relevance("Artificial Intelligence is a fascinating field of study", ["artificial intelligence"])
        assert 0.0 < score < 1.0

    def test_no_match_scores_zero(self):
        assert _score_relevance("the sky is blue today", ["gambling"]) == 0.0


class TestBlacklistWhitelist:
    def test_blacklist_hit(self):
        assert _matches_blacklist("free money casino bonus", ["gambling", "casino"]) is True

    def test_blacklist_miss(self):
        assert _matches_blacklist("the weather is nice today", ["gambling"]) is False

    def test_whitelist_empty_always_matches(self):
        assert _matches_whitelist("anything", []) is True

    def test_whitelist_hit(self):
        assert _matches_whitelist("deep learning advances", ["learning"]) is True

    def test_whitelist_miss(self):
        assert _matches_whitelist("cooking pasta at home", ["artificial intelligence"]) is False


class TestConfigIO:
    def test_load_missing_returns_defaults(self, config_path):
        cfg = _load_config()
        assert cfg == DEFAULT_CONFIG
        assert cfg["enabled"] is True

    def test_save_then_load_roundtrip(self, config_path):
        cfg = dict(DEFAULT_CONFIG)
        cfg["min_quality_score"] = 0.9
        _save_config(cfg)
        assert _load_config()["min_quality_score"] == 0.9

    def test_load_invalid_json_returns_defaults(self, config_path, caplog):
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text("{not valid json")
        assert _load_config() == DEFAULT_CONFIG
        assert any("Failed to load filter config" in r.message for r in caplog.records)


# ── DataFilter ─────────────────────────────────────────────────────────


class TestDataFilter:
    def test_default_config_loaded(self, config_path):
        f = DataFilter()
        assert f.get_config() == DEFAULT_CONFIG

    def test_custom_config_merged_and_saved(self, config_path):
        f = DataFilter({"min_quality_score": 0.8, "min_content_length": 500})
        assert f.get_config()["min_quality_score"] == 0.8
        assert f.get_config()["min_content_length"] == 500
        assert config_path.exists()

    def test_update_config_persists(self, config_path):
        f = DataFilter()
        f.update_config(enabled=False)
        assert f.get_config()["enabled"] is False
        assert _load_config()["enabled"] is False

    def test_get_stats_returns_copy(self, config_path):
        f = DataFilter()
        stats = f.get_stats()
        stats["passed"] = 999
        assert f.get_stats()["passed"] == 0

    def test_disabled_always_passes(self, config_path):
        f = DataFilter({"enabled": False})
        ok, reason = f.filter_article("u", "t", "short")
        assert ok is True
        assert reason == ""
        assert f.get_stats()["passed"] == 1

    def test_too_short_rejected(self, config_path):
        f = DataFilter()
        ok, reason = f.filter_article("u", "t", "tiny")
        assert ok is False
        assert reason == "too_short"
        assert f.get_stats()["rejected_short"] == 1

    def test_low_quality_rejected(self, config_path):
        f = DataFilter()
        content = "SUPERCALIFRAGILISTICEXPIALIDOCIOUS " * 6
        ok, reason = f.filter_article("u", "t", content)
        assert ok is False
        assert reason.startswith("low_quality_")
        assert f.get_stats()["rejected_quality"] == 1

    def test_blacklist_rejected(self, config_path):
        f = DataFilter()
        content = GOOD_ARTICLE + " A nearby casino offers free money gambling bonuses."
        ok, reason = f.filter_article("u", "t", content)
        assert ok is False
        assert reason == "blacklisted"
        assert f.get_stats()["rejected_blacklist"] == 1

    def test_whitelist_hard_gate_rejects(self, config_path):
        f = DataFilter({"topic_whitelist": ["artificial intelligence"], "whitelist_is_hard_gate": True})
        ok, reason = f.filter_article("u", "t", GOOD_ARTICLE)
        assert ok is False
        assert reason == "not_in_whitelist"
        assert f.get_stats()["rejected_whitelist"] == 1

    def test_whitelist_hard_gate_passes_with_match(self, config_path):
        f = DataFilter({"topic_whitelist": ["river"], "whitelist_is_hard_gate": True})
        ok, reason = f.filter_article("u", "About the river", GOOD_ARTICLE)
        assert ok is True
        assert reason == ""

    def test_whitelist_without_hard_gate_does_not_block(self, config_path):
        f = DataFilter({"topic_whitelist": ["artificial intelligence"], "whitelist_is_hard_gate": False})
        ok, reason = f.filter_article("u", "t", GOOD_ARTICLE)
        assert ok is True
        assert reason == ""

    def test_near_duplicate_rejected(self, config_path):
        f = DataFilter()
        ok, reason = f.filter_article("u", "t", GOOD_ARTICLE, existing_facts=[GOOD_ARTICLE])
        assert ok is False
        assert reason == "near_duplicate"
        assert f.get_stats()["rejected_dup"] == 1

    def test_passes_good_article(self, config_path):
        f = DataFilter()
        ok, reason = f.filter_article("u", "t", GOOD_ARTICLE)
        assert ok is True
        assert reason == ""
        assert f.get_stats()["total_seen"] == 1
        assert f.get_stats()["passed"] == 1

    def test_filter_chunk_blacklist(self, config_path):
        f = DataFilter()
        assert f.filter_chunk("win free money at this casino", "promo") is False

    def test_filter_chunk_whitelist_hard_gate(self, config_path):
        f = DataFilter({"topic_whitelist": ["river"], "whitelist_is_hard_gate": True})
        assert f.filter_chunk("oak tree by the stream", "nature") is False
        assert f.filter_chunk("river flow in spring", "nature") is True

    def test_filter_chunk_whitelist_soft_gate(self, config_path):
        f = DataFilter({"topic_whitelist": ["river"], "whitelist_is_hard_gate": False})
        assert f.filter_chunk("anything at all", "topic") is True

    def test_filter_chunk_disabled(self, config_path):
        f = DataFilter({"enabled": False})
        assert f.filter_chunk("gambling casino", "promo") is True


class TestGetDataFilter:
    def test_singleton_shared(self, config_path, fresh_singleton):
        a = get_data_filter()
        b = get_data_filter()
        assert a is b
        assert isinstance(a, DataFilter)

    def test_singleton_created_once(self, config_path, fresh_singleton, monkeypatch):
        calls = []
        monkeypatch.setattr(df, "DataFilter", lambda *a, **k: calls.append(1) or object())
        get_data_filter()
        get_data_filter()
        assert len(calls) == 1
