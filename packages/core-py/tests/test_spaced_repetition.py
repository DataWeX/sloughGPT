"""Tests for Spaced Repetition Scheduler — review scheduling, due detection, stats.

Covers:
  - Scheduling reviews with different performance levels
  - Interval calculation based on average performance
  - Due review detection
  - Review statistics
"""

import time
import pytest
from domains.infrastructure.spaced_repetition_engine import SpacedRepetitionScheduler


class TestSpacedRepetitionScheduler:
    def test_init(self):
        s = SpacedRepetitionScheduler()
        assert len(s.review_schedule) == 0
        assert len(s.performance_history) == 0

    def test_schedule_review_excellent(self):
        s = SpacedRepetitionScheduler()
        next_time = s.schedule_review("doc1", 0.95)
        assert next_time > time.time()
        # ≥0.9 → month interval (~30 days)
        interval = next_time - time.time()
        assert 29 * 86400 < interval < 31 * 86400

    def test_schedule_review_good(self):
        s = SpacedRepetitionScheduler()
        next_time = s.schedule_review("doc1", 0.85)
        interval = next_time - time.time()
        # 0.8-0.9 → week interval (~7 days)
        assert 6 * 86400 < interval < 8 * 86400

    def test_schedule_review_fair(self):
        s = SpacedRepetitionScheduler()
        next_time = s.schedule_review("doc1", 0.7)
        interval = next_time - time.time()
        # 0.6-0.8 → 3 days
        assert 2 * 86400 < interval < 4 * 86400

    def test_schedule_review_poor(self):
        s = SpacedRepetitionScheduler()
        next_time = s.schedule_review("doc1", 0.3)
        interval = next_time - time.time()
        # <0.6 → 1 day
        assert 0.5 * 86400 < interval < 1.5 * 86400

    def test_average_performance(self):
        s = SpacedRepetitionScheduler()
        s.schedule_review("doc1", 0.95)
        s.schedule_review("doc1", 0.5)
        # Average: (0.95+0.5)/2 = 0.725 → 3-day interval
        next_time = s.schedule_review("doc1", 0.5)
        # Third call uses avg of [0.95, 0.5, 0.5] = 0.65 → 3-day
        interval = next_time - time.time()
        assert 2 * 86400 < interval < 4 * 86400

    def test_performance_history_accumulates(self):
        s = SpacedRepetitionScheduler()
        s.schedule_review("doc1", 0.8)
        s.schedule_review("doc1", 0.9)
        assert s.performance_history["doc1"] == [0.8, 0.9]

    def test_get_due_reviews_empty(self):
        s = SpacedRepetitionScheduler()
        assert s.get_due_reviews() == []

    def test_get_due_reviews_with_future(self):
        s = SpacedRepetitionScheduler()
        s.schedule_review("doc1", 0.9)
        due = s.get_due_reviews()
        assert "doc1" not in due  # scheduled for month from now

    def test_get_next_review_time(self):
        s = SpacedRepetitionScheduler()
        s.schedule_review("doc1", 0.8)
        t = s.get_next_review_time("doc1")
        assert t is not None
        assert t > time.time()

    def test_get_next_review_time_missing(self):
        s = SpacedRepetitionScheduler()
        assert s.get_next_review_time("nonexistent") is None

    def test_review_stats(self):
        s = SpacedRepetitionScheduler()
        s.schedule_review("doc1", 0.9)
        s.schedule_review("doc2", 0.5)
        stats = s.get_review_stats()
        assert stats["total_scheduled"] == 2
        assert stats["due_count"] == 0
        assert "doc1" in stats["upcoming_reviews"]
        assert "doc2" in stats["upcoming_reviews"]

    def test_multiple_documents(self):
        s = SpacedRepetitionScheduler()
        s.schedule_review("a", 0.95)
        s.schedule_review("b", 0.3)
        s.schedule_review("c", 0.7)
        assert len(s.review_schedule) == 3
        # Each gets its own interval
        intervals = {}
        now = time.time()
        for doc_id in ["a", "b", "c"]:
            intervals[doc_id] = s.review_schedule[doc_id] - now
        assert intervals["a"] > intervals["c"] > intervals["b"]
