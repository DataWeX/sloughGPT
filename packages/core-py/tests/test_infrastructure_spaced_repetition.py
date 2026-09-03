"""Tests for SpacedRepetitionScheduler — spaced repetition learning."""
from __future__ import annotations

import time

from domains.infrastructure.spaced_repetition_engine import SpacedRepetitionScheduler


class TestSpacedRepetitionScheduler:
    def test_schedule_review(self):
        sched = SpacedRepetitionScheduler()
        before = time.time()
        next_t = sched.schedule_review("doc1", 0.9)
        assert next_t > before

    def test_high_performance_long_interval(self):
        sched = SpacedRepetitionScheduler()
        t = sched.schedule_review("doc1", 0.95)
        # Should be ~30 days out
        days = (t - time.time()) / (24 * 3600)
        assert 29 < days < 31

    def test_medium_performance_week_interval(self):
        sched = SpacedRepetitionScheduler()
        t = sched.schedule_review("doc1", 0.85)
        days = (t - time.time()) / (24 * 3600)
        assert 6 < days < 8

    def test_low_performance_day_interval(self):
        sched = SpacedRepetitionScheduler()
        t = sched.schedule_review("doc1", 0.4)
        days = (t - time.time()) / (24 * 3600)
        assert 0.9 < days < 1.1

    def test_get_due_reviews(self):
        sched = SpacedRepetitionScheduler()
        sched.review_schedule["old"] = time.time() - 100
        sched.review_schedule["future"] = time.time() + 100000
        due = sched.get_due_reviews()
        assert "old" in due
        assert "future" not in due

    def test_get_next_review_time(self):
        sched = SpacedRepetitionScheduler()
        t = sched.schedule_review("doc1", 0.7)
        assert sched.get_next_review_time("doc1") == t
        assert sched.get_next_review_time("missing") is None

    def test_review_stats(self):
        sched = SpacedRepetitionScheduler()
        sched.schedule_review("d1", 0.5)
        stats = sched.get_review_stats()
        assert stats["total_scheduled"] == 1
        assert "due_count" in stats

    def test_performance_averages(self):
        sched = SpacedRepetitionScheduler()
        sched.schedule_review("doc1", 0.5)  # low → 1 day
        sched.schedule_review("doc1", 0.95)  # average is now 0.725 → 3 days
        t = sched.get_next_review_time("doc1")
        days = (t - time.time()) / (24 * 3600)
        assert 2.5 < days < 3.5
