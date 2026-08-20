"""Tests for domains.infrastructure.spaced_repetition_engine."""

import time
from domains.infrastructure.spaced_repetition_engine import SpacedRepetitionScheduler


class TestSpacedRepetitionScheduler:
    def test_init(self):
        sched = SpacedRepetitionScheduler()
        assert sched.review_schedule == {}

    def test_schedule_review_returns_future(self):
        sched = SpacedRepetitionScheduler()
        ts = sched.schedule_review("doc1", 0.8)
        assert ts > time.time()

    def test_good_performance_longer_interval(self):
        sched = SpacedRepetitionScheduler()
        ts_high = sched.schedule_review("doc1", 0.95)
        ts_low = sched.schedule_review("doc2", 0.3)
        assert ts_high > ts_low

    def test_get_next_review_time(self):
        sched = SpacedRepetitionScheduler()
        sched.schedule_review("doc1", 0.8)
        nxt = sched.get_next_review_time("doc1")
        assert nxt is not None
        assert nxt > time.time()

    def test_get_next_review_time_unknown(self):
        sched = SpacedRepetitionScheduler()
        assert sched.get_next_review_time("unknown") is None

    def test_get_due_reviews_empty(self):
        sched = SpacedRepetitionScheduler()
        assert sched.get_due_reviews() == []

    def test_get_due_reviews_with_past(self):
        sched = SpacedRepetitionScheduler()
        sched.review_schedule["doc1"] = time.time() - 10
        due = sched.get_due_reviews()
        assert "doc1" in due

    def test_get_review_stats(self):
        sched = SpacedRepetitionScheduler()
        sched.schedule_review("doc1", 0.8)
        stats = sched.get_review_stats()
        assert stats["total_scheduled"] == 1

    def test_performance_history_recorded(self):
        sched = SpacedRepetitionScheduler()
        sched.schedule_review("doc1", 0.7)
        sched.schedule_review("doc1", 0.9)
        assert len(sched.performance_history["doc1"]) == 2

    def test_month_interval(self):
        sched = SpacedRepetitionScheduler()
        ts = sched.schedule_review("doc1", 0.95)
        diff = ts - time.time()
        assert diff >= 29 * 86400
