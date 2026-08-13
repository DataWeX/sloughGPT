"""Tests for SpacedRepetitionScheduler — review scheduling and stats."""

import time
import pytest
from domains.infrastructure.spaced_repetition_engine import SpacedRepetitionScheduler


@pytest.fixture
def scheduler():
    return SpacedRepetitionScheduler()


class TestScheduleReview:
    def test_returns_future_timestamp(self, scheduler):
        result = scheduler.schedule_review("doc1", 0.8)
        assert result > time.time()

    def test_stores_in_schedule(self, scheduler):
        scheduler.schedule_review("doc1", 0.8)
        assert "doc1" in scheduler.review_schedule

    def test_stores_performance(self, scheduler):
        scheduler.schedule_review("doc1", 0.9)
        assert 0.9 in scheduler.performance_history["doc1"]

    def test_excellent_performance_long_interval(self, scheduler):
        t0 = time.time()
        result = scheduler.schedule_review("doc1", 0.95)
        interval = result - t0
        assert interval >= 25 * 24 * 3600  # ~month

    def test_good_performance_week_interval(self, scheduler):
        t0 = time.time()
        result = scheduler.schedule_review("doc1", 0.85)
        interval = result - t0
        assert 5 * 24 * 3600 <= interval <= 9 * 24 * 3600  # ~week

    def test_moderate_performance_3day_interval(self, scheduler):
        t0 = time.time()
        result = scheduler.schedule_review("doc1", 0.7)
        interval = result - t0
        assert 2 * 24 * 3600 <= interval <= 4 * 24 * 3600  # 3 days

    def test_poor_performance_day_interval(self, scheduler):
        t0 = time.time()
        result = scheduler.schedule_review("doc1", 0.3)
        interval = result - t0
        assert 0.5 * 24 * 3600 <= interval <= 1.5 * 24 * 3600  # ~1 day

    def test_averages_multiple_reviews(self, scheduler):
        scheduler.schedule_review("doc1", 0.95)
        scheduler.schedule_review("doc1", 0.65)
        avg = sum(scheduler.performance_history["doc1"]) / 2
        assert avg == pytest.approx(0.8, abs=0.01)

    def test_separate_docs(self, scheduler):
        scheduler.schedule_review("doc1", 0.9)
        scheduler.schedule_review("doc2", 0.3)
        assert "doc1" in scheduler.review_schedule
        assert "doc2" in scheduler.review_schedule

    def test_multiple_reviews_same_doc(self, scheduler):
        scheduler.schedule_review("doc1", 0.5)
        scheduler.schedule_review("doc1", 0.5)
        scheduler.schedule_review("doc1", 0.5)
        assert len(scheduler.performance_history["doc1"]) == 3


class TestGetDueReviews:
    def test_no_reviews_returns_empty(self, scheduler):
        assert scheduler.get_due_reviews() == []

    def test_past_review_is_due(self, scheduler):
        scheduler.review_schedule["doc1"] = time.time() - 100
        assert "doc1" in scheduler.get_due_reviews()

    def test_future_review_not_due(self, scheduler):
        scheduler.review_schedule["doc1"] = time.time() + 86400
        assert "doc1" not in scheduler.get_due_reviews()

    def test_mixed_due_and_not(self, scheduler):
        scheduler.review_schedule["past"] = time.time() - 100
        scheduler.review_schedule["future"] = time.time() + 86400
        due = scheduler.get_due_reviews()
        assert "past" in due
        assert "future" not in due


class TestGetNextReviewTime:
    def test_returns_none_for_unknown(self, scheduler):
        assert scheduler.get_next_review_time("nonexistent") is None

    def test_returns_stored_time(self, scheduler):
        ts = scheduler.schedule_review("doc1", 0.8)
        assert scheduler.get_next_review_time("doc1") == ts


class TestGetReviewStats:
    def test_empty_stats(self, scheduler):
        stats = scheduler.get_review_stats()
        assert stats["due_count"] == 0
        assert stats["total_scheduled"] == 0
        assert stats["due_documents"] == []
        assert stats["upcoming_reviews"] == {}

    def test_stats_with_scheduled(self, scheduler):
        scheduler.schedule_review("doc1", 0.8)
        stats = scheduler.get_review_stats()
        assert stats["total_scheduled"] == 1
        assert "doc1" in stats["upcoming_reviews"]

    def test_stats_with_due(self, scheduler):
        scheduler.review_schedule["doc1"] = time.time() - 10
        stats = scheduler.get_review_stats()
        assert stats["due_count"] == 1
        assert "doc1" in stats["due_documents"]

    def test_upcoming_shows_days(self, scheduler):
        scheduler.schedule_review("doc1", 0.9)
        stats = scheduler.get_review_stats()
        days = stats["upcoming_reviews"]["doc1"]
        assert isinstance(days, float)
        assert days >= 0
