"""
Tests for domains/infrastructure/spaced_repetition_engine.py — spaced repetition scheduler.

Covers:
    - Review scheduling with different performance levels
    - Due review detection
    - Performance history tracking
    - Review stats
    - Interval tier thresholds
"""

import time
import sys
from pathlib import Path
import pytest

_CORE_PY = Path(__file__).resolve().parents[1]
if str(_CORE_PY) not in sys.path:
    sys.path.insert(0, str(_CORE_PY))

from domains.infrastructure.spaced_repetition_engine import SpacedRepetitionScheduler


class TestSpacedRepetitionScheduler:
    def test_schedule_review_high_performance(self):
        s = SpacedRepetitionScheduler()
        next_t = s.schedule_review("doc1", 0.95)
        assert next_t > time.time()
        # High performance → month interval (30 days)
        delta = next_t - time.time()
        assert delta > 29 * 86400

    def test_schedule_review_medium_performance(self):
        s = SpacedRepetitionScheduler()
        next_t = s.schedule_review("doc1", 0.85)
        delta = next_t - time.time()
        assert 6 * 86400 < delta < 8 * 86400  # ~1 week

    def test_schedule_review_low_performance(self):
        s = SpacedRepetitionScheduler()
        next_t = s.schedule_review("doc1", 0.5)
        delta = next_t - time.time()
        assert 0.9 * 86400 < delta < 1.1 * 86400  # ~1 day

    def test_schedule_review_terrible_performance(self):
        s = SpacedRepetitionScheduler()
        next_t = s.schedule_review("doc1", 0.3)
        delta = next_t - time.time()
        assert 0.9 * 86400 < delta < 1.1 * 86400  # ~1 day

    def test_performance_history_accumulates(self):
        s = SpacedRepetitionScheduler()
        s.schedule_review("doc1", 0.9)
        s.schedule_review("doc1", 0.7)
        assert len(s.performance_history["doc1"]) == 2
        assert s.performance_history["doc1"] == [0.9, 0.7]

    def test_average_performance_drives_interval(self):
        s = SpacedRepetitionScheduler()
        # Two reviews: avg = 0.75 → 3-day interval
        s.schedule_review("doc1", 0.9)
        next_t = s.schedule_review("doc1", 0.6)
        delta = next_t - time.time()
        assert 2.5 * 86400 < delta < 3.5 * 86400

    def test_get_due_reviews_empty(self):
        s = SpacedRepetitionScheduler()
        assert s.get_due_reviews() == []

    def test_get_due_reviews_future(self):
        s = SpacedRepetitionScheduler()
        s.schedule_review("doc1", 0.9)
        assert s.get_due_reviews() == []

    def test_get_due_reviews_past(self):
        s = SpacedRepetitionScheduler()
        s.review_schedule["doc1"] = time.time() - 100
        due = s.get_due_reviews()
        assert "doc1" in due

    def test_get_next_review_time(self):
        s = SpacedRepetitionScheduler()
        t = s.schedule_review("doc1", 0.8)
        assert s.get_next_review_time("doc1") == t

    def test_get_next_review_time_unknown(self):
        s = SpacedRepetitionScheduler()
        assert s.get_next_review_time("nonexistent") is None

    def test_get_review_stats(self):
        s = SpacedRepetitionScheduler()
        s.review_schedule["past"] = time.time() - 100
        s.review_schedule["future"] = time.time() + 86400
        stats = s.get_review_stats()
        assert stats["due_count"] == 1
        assert "past" in stats["due_documents"]
        assert stats["total_scheduled"] == 2
        assert "future" in stats["upcoming_reviews"]

    def test_multiple_documents(self):
        s = SpacedRepetitionScheduler()
        s.schedule_review("a", 0.95)
        s.schedule_review("b", 0.5)
        s.schedule_review("c", 0.8)
        assert len(s.review_schedule) == 3
        assert len(s.performance_history) == 3

    def test_boundary_exactly_08(self):
        s = SpacedRepetitionScheduler()
        next_t = s.schedule_review("doc1", 0.8)
        delta = next_t - time.time()
        assert 6 * 86400 < delta < 8 * 86400  # exactly 0.8 → week

    def test_boundary_exactly_06(self):
        s = SpacedRepetitionScheduler()
        next_t = s.schedule_review("doc1", 0.6)
        delta = next_t - time.time()
        assert 2.5 * 86400 < delta < 3.5 * 86400  # exactly 0.6 → 3 days
