"""Tests for SpacedRepetitionScheduler — performance-based review scheduling."""

import pytest

import domains.infrastructure.spaced_repetition_engine as sre
from domains.infrastructure.spaced_repetition_engine import SpacedRepetitionScheduler

DAY = 24 * 3600
WEEK = 7 * DAY
MONTH = 30 * DAY


class FakeClock:
    def __init__(self, now):
        self.now = now

    def time(self):
        return self.now


@pytest.fixture
def scheduler(monkeypatch):
    s = SpacedRepetitionScheduler()
    clock = FakeClock(1000.0)
    monkeypatch.setattr(sre, "time_module", clock)
    return s, clock


class TestScheduleReview:
    def test_excellent_performance_month_interval(self, scheduler):
        s, clock = scheduler
        next_review = s.schedule_review("doc1", 0.95)
        assert next_review == pytest.approx(1000.0 + MONTH)

    def test_good_performance_week_interval(self, scheduler):
        s, clock = scheduler
        next_review = s.schedule_review("doc1", 0.85)
        assert next_review == pytest.approx(1000.0 + WEEK)

    def test_average_performance_three_days(self, scheduler):
        s, clock = scheduler
        next_review = s.schedule_review("doc1", 0.7)
        assert next_review == pytest.approx(1000.0 + 3 * DAY)

    def test_poor_performance_day_interval(self, scheduler):
        s, clock = scheduler
        next_review = s.schedule_review("doc1", 0.4)
        assert next_review == pytest.approx(1000.0 + DAY)

    def test_records_performance_history(self, scheduler):
        s, clock = scheduler
        s.schedule_review("doc1", 0.8)
        s.schedule_review("doc1", 0.9)
        assert s.performance_history["doc1"] == [0.8, 0.9]

    def test_interval_uses_average_performance(self, scheduler):
        s, clock = scheduler
        s.schedule_review("doc1", 0.5)
        s.schedule_review("doc1", 1.0)
        next_review = s.schedule_review("doc1", 0.6)
        avg = (0.5 + 1.0 + 0.6) / 3
        assert 0.6 <= avg < 0.8
        assert next_review == pytest.approx(1000.0 + 3 * DAY)

    def test_returns_next_review_timestamp(self, scheduler):
        s, clock = scheduler
        next_review = s.schedule_review("doc1", 0.9)
        assert next_review > clock.now

    def test_multiple_docs_independent(self, scheduler):
        s, clock = scheduler
        s.schedule_review("a", 0.95)
        s.schedule_review("b", 0.4)
        assert s.review_schedule["a"] == pytest.approx(1000.0 + MONTH)
        assert s.review_schedule["b"] == pytest.approx(1000.0 + DAY)


class TestGetDueReviews:
    def test_not_due_before_review_time(self, scheduler):
        s, clock = scheduler
        s.schedule_review("doc1", 0.4)
        assert s.get_due_reviews() == []

    def test_due_when_time_reaches_review(self, scheduler):
        s, clock = scheduler
        s.schedule_review("doc1", 0.4)
        clock.now = 1000.0 + DAY
        assert s.get_due_reviews() == ["doc1"]

    def test_due_after_review_time(self, scheduler):
        s, clock = scheduler
        s.schedule_review("doc1", 0.4)
        clock.now = 1000.0 + DAY + 100
        assert s.get_due_reviews() == ["doc1"]

    def test_only_scheduled_docs_are_due(self, scheduler):
        s, clock = scheduler
        s.schedule_review("due", 0.4)
        s.schedule_review("later", 0.95)
        clock.now = 1000.0 + DAY + 1
        assert s.get_due_reviews() == ["due"]

    def test_empty_schedule(self, scheduler):
        s, clock = scheduler
        assert s.get_due_reviews() == []


class TestGetNextReviewTime:
    def test_returns_scheduled_time(self, scheduler):
        s, clock = scheduler
        next_review = s.schedule_review("doc1", 0.9)
        assert s.get_next_review_time("doc1") == next_review

    def test_unknown_doc_returns_none(self, scheduler):
        s, clock = scheduler
        assert s.get_next_review_time("nope") is None


class TestGetReviewStats:
    def test_stats_counts(self, scheduler):
        s, clock = scheduler
        s.schedule_review("due", 0.4)
        s.schedule_review("later", 0.95)
        clock.now = 1000.0 + DAY + 1
        stats = s.get_review_stats()
        assert stats["due_count"] == 1
        assert stats["due_documents"] == ["due"]
        assert stats["total_scheduled"] == 2

    def test_upcoming_reviews_in_days(self, scheduler):
        s, clock = scheduler
        s.schedule_review("later", 0.95)
        stats = s.get_review_stats()
        assert "later" in stats["upcoming_reviews"]
        assert stats["upcoming_reviews"]["later"] == pytest.approx(30.0, abs=0.1)

    def test_empty_schedule_stats(self, scheduler):
        s, clock = scheduler
        stats = s.get_review_stats()
        assert stats["due_count"] == 0
        assert stats["due_documents"] == []
        assert stats["total_scheduled"] == 0
        assert stats["upcoming_reviews"] == {}


class TestDefaultIntervals:
    def test_interval_definitions(self):
        s = SpacedRepetitionScheduler()
        assert s.intervals["day"] == DAY
        assert s.intervals["week"] == WEEK
        assert s.intervals["month"] == MONTH
