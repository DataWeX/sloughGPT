"""Tests for memory/consolidation.py — near-duplicate fact deduplication."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from domain.memory._internal.consolidation import plan_consolidation


# ── Helpers ──────────────────────────────────────────────────────────────────

def _fact(fid: str, content: str, topic: str = "general") -> dict:
    return {"id": fid, "content": content, "topic": topic}


# ── Edge cases ───────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_input(self):
        result = plan_consolidation([])
        assert result == {"keep_ids": [], "remove_ids": [], "groups": [], "removed_count": 0}

    def test_single_fact(self):
        result = plan_consolidation([_fact("a", "hello world")])
        assert result["keep_ids"] == ["a"]
        assert result["remove_ids"] == []
        assert result["removed_count"] == 0

    def test_two_distinct_facts(self):
        with patch("domain.memory._internal.consolidation._cosine_similarity", return_value=0.3):
            result = plan_consolidation([
                _fact("a", "cats are animals"),
                _fact("b", "quantum physics is complex"),
            ])
        assert result["keep_ids"] == ["a", "b"]
        assert result["remove_ids"] == []
        assert result["removed_count"] == 0

    def test_missing_id_skipped(self):
        with patch("domain.memory._internal.consolidation._cosine_similarity", return_value=0.0):
            result = plan_consolidation([{"content": "no id field"}])
        assert result["keep_ids"] == []
        assert result["removed_count"] == 0


# ── Duplicate detection ──────────────────────────────────────────────────────

class TestDuplicateDetection:
    def test_two_near_duplicates(self):
        with patch("domain.memory._internal.consolidation._cosine_similarity", return_value=0.95):
            result = plan_consolidation([
                _fact("a", "The user prefers Zed over VS Code"),
                _fact("b", "User prefers the editor Zed"),
            ])
        assert "a" in result["keep_ids"] or "b" in result["keep_ids"]
        assert result["removed_count"] == 1
        assert len(result["groups"]) == 1
        group = result["groups"][0]
        assert len(group["duplicates"]) == 1

    def test_keeps_longest_fact(self):
        short = _fact("a", "short")
        long = _fact("b", "this is a much longer fact with more detail")
        with patch("domain.memory._internal.consolidation._cosine_similarity", return_value=0.95):
            result = plan_consolidation([short, long])
        assert result["keep_ids"] == ["b"]
        assert result["remove_ids"] == ["a"]

    def test_three_duplicates_cluster(self):
        with patch("domain.memory._internal.consolidation._cosine_similarity", return_value=0.90):
            result = plan_consolidation([
                _fact("a", "fact one"),
                _fact("b", "fact one rephrased"),
                _fact("c", "fact one restated again"),
            ])
        assert result["removed_count"] == 2
        assert len(result["groups"]) == 1
        assert len(result["groups"][0]["duplicates"]) == 2

    def test_transitive_clustering(self):
        """A~B and B~C but A~C below threshold — all should cluster via transitivity."""
        import numpy as np

        call_count = [0]
        def mock_sim(va, vb):
            call_count[0] += 1
            # First call (a,b) = 0.95, second (a,c) = 0.40, third (b,c) = 0.95
            n = call_count[0]
            if n == 1:
                return 0.95  # a,b
            elif n == 2:
                return 0.40  # a,c
            else:
                return 0.95  # b,c

        with patch("domain.memory._internal.consolidation._cosine_similarity", side_effect=mock_sim):
            result = plan_consolidation([
                _fact("a", "alpha"),
                _fact("b", "beta"),
                _fact("c", "charlie"),
            ])
        assert result["removed_count"] == 2
        assert len(result["groups"]) == 1


# ── Topic isolation ──────────────────────────────────────────────────────────

class TestTopicIsolation:
    def test_different_topics_not_clustered(self):
        with patch("domain.memory._internal.consolidation._cosine_similarity", return_value=0.99):
            result = plan_consolidation([
                _fact("a", "same text", topic="cooking"),
                _fact("b", "same text", topic="coding"),
            ])
        assert result["removed_count"] == 0
        assert result["keep_ids"] == ["a", "b"]

    def test_same_topic_clustered(self):
        with patch("domain.memory._internal.consolidation._cosine_similarity", return_value=0.99):
            result = plan_consolidation([
                _fact("a", "same text", topic="cooking"),
                _fact("b", "same text", topic="cooking"),
            ])
        assert result["removed_count"] == 1

    def test_none_topic_defaults_to_general(self):
        with patch("domain.memory._internal.consolidation._cosine_similarity", return_value=0.99):
            result = plan_consolidation([
                _fact("a", "text", topic=None),
                _fact("b", "text", topic=None),
            ])
        assert result["removed_count"] == 1


# ── Threshold ────────────────────────────────────────────────────────────────

class TestThreshold:
    def test_exact_threshold_included(self):
        with patch("domain.memory._internal.consolidation._cosine_similarity", return_value=0.80):
            result = plan_consolidation(
                [_fact("a", "x"), _fact("b", "y")],
                threshold=0.80,
            )
        assert result["removed_count"] == 1

    def test_below_threshold_excluded(self):
        with patch("domain.memory._internal.consolidation._cosine_similarity", return_value=0.79):
            result = plan_consolidation(
                [_fact("a", "x"), _fact("b", "y")],
                threshold=0.80,
            )
        assert result["removed_count"] == 0

    def test_custom_threshold(self):
        with patch("domain.memory._internal.consolidation._cosine_similarity", return_value=0.50):
            result = plan_consolidation(
                [_fact("a", "x"), _fact("b", "y")],
                threshold=0.50,
            )
        assert result["removed_count"] == 1


# ── Multiple clusters ────────────────────────────────────────────────────────

class TestMultipleClusters:
    def test_two_separate_clusters(self):
        call_count = [0]
        def mock_sim(va, vb):
            call_count[0] += 1
            n = call_count[0]
            # Pairs in order: (a,b), (a,c), (a,d), (b,c), (b,d), (c,d)
            if n in (1, 6):  # (a,b) and (c,d)
                return 0.95
            return 0.20

        with patch("domain.memory._internal.consolidation._cosine_similarity", side_effect=mock_sim):
            result = plan_consolidation([
                _fact("a", "cooking tip one"),
                _fact("b", "cooking tip one rephrased"),
                _fact("c", "coding tip one"),
                _fact("d", "coding tip one rephrased"),
            ])
        assert result["removed_count"] == 2
        assert len(result["groups"]) == 2
