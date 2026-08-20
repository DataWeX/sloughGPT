"""
Tests for DataFilter (domains/learner/data_filter.py).
"""

import os
import json
from pathlib import Path

import pytest

from domains.learner.data_filter import (
    DataFilter,
    _score_quality,
    _score_relevance,
    _matches_blacklist,
    _matches_whitelist,
    DEFAULT_CONFIG,
    set_data_filter_db,
    reset_data_filter_db,
)


@pytest.fixture(autouse=True)
def _temp_mogdb(tmp_path):
    """Redirect filter config to a temp MogDB so tests don't interfere."""
    db_path = str(tmp_path / "test_data_filter")
    set_data_filter_db(db_path)
    yield
    reset_data_filter_db()


class TestScoreQuality:
    def test_empty_text(self):
        assert _score_quality("") == 0.0
        assert _score_quality("short") == 0.0

    def test_good_prose_scores_high(self):
        text = (
            "Python is a versatile programming language. "
            "It supports multiple paradigms including object-oriented programming. "
            "The language is widely used in data science and web development. "
            "Its readable syntax makes it accessible for beginners. "
        )
        assert _score_quality(text) >= 0.3


class TestMatchesBlacklist:
    def test_detects_blacklisted(self):
        assert _matches_blacklist("this is about gambling", ["gambling", "casino"])

    def test_clean_text_not_flagged(self):
        assert not _matches_blacklist("python programming", ["gambling", "casino"])

    def test_case_insensitive(self):
        assert _matches_blacklist("Gambling is bad", ["gambling"])


class TestMatchesWhitelist:
    def test_no_whitelist_matches_all(self):
        assert _matches_whitelist("anything", [])

    def test_matching_topic(self):
        assert _matches_whitelist("python programming guide", ["python"])

    def test_non_matching_topic(self):
        assert not _matches_whitelist("cooking recipes", ["python"])


class TestDataFilter:
    def test_default_config(self):
        f = DataFilter()
        assert f.config["min_content_length"] == 200
        assert f.config["enabled"] is True

    def test_disabled_passes_everything(self):
        f = DataFilter({"enabled": False, "min_content_length": 999999})
        ok, _ = f.filter_article("http://x.com", "", "")
        assert ok is True

    def test_rejects_short_content(self):
        f = DataFilter({"enabled": True})
        ok, reason = f.filter_article("http://x.com", "", "short")
        assert ok is False
        assert reason == "too_short"

    def test_rejects_blacklisted(self):
        f = DataFilter({"enabled": True, "min_quality_score": 0, "min_content_length": 0})
        content = "valid content " * 30
        title = "casino gambling"
        ok, reason = f.filter_article("http://x.com", title, content)
        assert ok is False
        assert reason == "blacklisted"

    def test_passes_good_content(self):
        f = DataFilter({"enabled": True, "min_quality_score": 0, "min_content_length": 0})
        content = (
            "Machine learning is a fascinating field. "
            + "It enables computers to learn from data without explicit programming. "
            + "Deep learning uses neural networks with many layers. "
            + "Natural language processing has seen remarkable progress. "
            + "Transformers revolutionized sequence modeling. "
            + "Attention mechanisms allow focus on relevant context. "
            + "Reinforcement learning teaches agents through trial and error. "
        )
        ok, reason = f.filter_article("http://x.com", "AI advances", content)
        assert ok is True
        assert reason == ""

    def test_whitelist_hard_gate_rejects_non_match(self):
        f = DataFilter({
            "enabled": True,
            "min_quality_score": 0,
            "min_content_length": 0,
            "topic_whitelist": ["python"],
            "whitelist_is_hard_gate": True,
        })
        content = "Cooking recipes for dinner. Pasta is delicious. " * 15
        ok, reason = f.filter_article("http://x.com", "cooking", content)
        assert ok is False
        assert reason == "not_in_whitelist"

    def test_whitelist_hard_gate_allows_match(self):
        f = DataFilter({
            "enabled": True,
            "min_quality_score": 0,
            "min_content_length": 0,
            "topic_whitelist": ["python"],
            "whitelist_is_hard_gate": True,
        })
        content = "Python programming tips. Learn to code. " * 15
        ok, reason = f.filter_article("http://x.com", "python", content)
        assert ok is True

    def test_near_dup_rejected(self):
        text = "Python is a great programming language for beginners. " * 20
        f = DataFilter({"enabled": True, "min_quality_score": 0, "min_content_length": 0, "dup_similarity_threshold": 0.9})
        ok, reason = f.filter_article("http://x.com", "Python", text, existing_facts=[text])
        assert ok is False
        assert reason == "near_duplicate"

    def test_stats_tracking(self):
        f = DataFilter({"enabled": True})
        f.filter_article("http://a.com", "", "short")
        good = "Python is versatile. It supports many paradigms. " * 15
        f.filter_article("http://b.com", "Python", good)
        assert f.stats["total_seen"] == 2
        assert f.stats["rejected"] == 1
        assert f.stats["passed"] == 1
        assert f.stats["rejected_short"] == 1

    def test_chunk_filter(self):
        f = DataFilter({"enabled": True})
        assert f.filter_chunk("python is great", "coding") is True
        assert f.filter_chunk("click here for casino", "ads") is False

    def test_update_config(self):
        f = DataFilter({"enabled": True})
        f.update_config(min_content_length=500)
        assert f.config["min_content_length"] == 500

    def test_get_config_returns_copy(self):
        f = DataFilter({"enabled": True})
        c = f.get_config()
        c["min_content_length"] = 999
        assert f.config["min_content_length"] != 999
