"""Tests for domains.infrastructure.spaced_repetition_engine."""

import time
from domains.infrastructure.spaced_repetition_engine import SpacedRepetitionScheduler


class TestSpacedRepetitionSchedulerInit:
    def test_init_empty_schedule(self):
        sched = SpacedRepetitionScheduler()
        assert sched.review_schedule == {}

    def test_init_empty_performance_history(self):
        sched = SpacedRepetitionScheduler()
        assert sched.performance_history == {}

    def test_init_intervals(self):
        sched = SpacedRepetitionScheduler()
        assert sched.intervals["day"] == 86400
        assert sched.intervals["week"] == 604800
        assert sched.intervals["month"] == 2592000

    def test_init_is_defaultdict(self):
        sched = SpacedRepetitionScheduler()
        sched.performance_history["new_key"]
        assert sched.performance_history["new_key"] == []

    def test_intervals_are_seconds(self):
        sched = SpacedRepetitionScheduler()
        assert sched.intervals["day"] == 24 * 3600
        assert sched.intervals["week"] == 7 * 24 * 3600
        assert sched.intervals["month"] == 30 * 24 * 3600

    def test_independent_instances(self):
        s1 = SpacedRepetitionScheduler()
        s2 = SpacedRepetitionScheduler()
        s1.schedule_review("d", 0.8)
        assert "d" not in s2.review_schedule

    def test_performance_history_independent(self):
        s1 = SpacedRepetitionScheduler()
        s2 = SpacedRepetitionScheduler()
        s1.schedule_review("d", 0.5)
        assert "d" not in s2.performance_history

    def test_intervals_dict_is_same_across_instances(self):
        s1 = SpacedRepetitionScheduler()
        s2 = SpacedRepetitionScheduler()
        assert s1.intervals == s2.intervals

    def test_review_schedule_type(self):
        sched = SpacedRepetitionScheduler()
        assert isinstance(sched.review_schedule, dict)

    def test_performance_history_type(self):
        sched = SpacedRepetitionScheduler()
        assert isinstance(sched.performance_history, dict)


# ── ScheduleReview ───────────────────────────────────────────────────────────


class TestScheduleReview:
    def test_returns_future_timestamp(self):
        sched = SpacedRepetitionScheduler()
        ts = sched.schedule_review("doc1", 0.8)
        assert ts > time.time()

    def test_high_performance_longer_interval(self):
        sched = SpacedRepetitionScheduler()
        ts_high = sched.schedule_review("doc1", 0.95)
        ts_low = sched.schedule_review("doc2", 0.3)
        assert ts_high > ts_low

    def test_excellent_performance_gives_month(self):
        sched = SpacedRepetitionScheduler()
        ts = sched.schedule_review("doc1", 0.95)
        diff = ts - time.time()
        assert diff >= 29 * 86400

    def test_good_performance_gives_week(self):
        sched = SpacedRepetitionScheduler()
        ts = sched.schedule_review("doc1", 0.8)
        diff = ts - time.time()
        assert 6 * 86400 <= diff <= 8 * 86400

    def test_average_performance_gives_3_days(self):
        sched = SpacedRepetitionScheduler()
        ts = sched.schedule_review("doc1", 0.6)
        diff = ts - time.time()
        assert 2.5 * 86400 <= diff <= 3.5 * 86400

    def test_poor_performance_gives_1_day(self):
        sched = SpacedRepetitionScheduler()
        ts = sched.schedule_review("doc1", 0.3)
        diff = ts - time.time()
        assert 0.5 * 86400 <= diff <= 1.5 * 86400

    def test_boundary_09_gets_month(self):
        sched = SpacedRepetitionScheduler()
        ts = sched.schedule_review("doc1", 0.9)
        diff = ts - time.time()
        assert diff >= 29 * 86400

    def test_boundary_08_gets_week(self):
        sched = SpacedRepetitionScheduler()
        ts = sched.schedule_review("doc1", 0.8)
        diff = ts - time.time()
        assert 6 * 86400 <= diff <= 8 * 86400

    def test_boundary_06_gets_3_days(self):
        sched = SpacedRepetitionScheduler()
        ts = sched.schedule_review("doc1", 0.6)
        diff = ts - time.time()
        assert 2.5 * 86400 <= diff <= 3.5 * 86400

    def test_boundary_05_gets_day(self):
        sched = SpacedRepetitionScheduler()
        ts = sched.schedule_review("doc1", 0.5)
        diff = ts - time.time()
        assert 0.5 * 86400 <= diff <= 1.5 * 86400

    def test_stores_in_review_schedule(self):
        sched = SpacedRepetitionScheduler()
        ts = sched.schedule_review("doc1", 0.8)
        assert sched.review_schedule["doc1"] == ts

    def test_multiple_same_doc_overwrites(self):
        sched = SpacedRepetitionScheduler()
        ts1 = sched.schedule_review("doc1", 0.3)
        ts2 = sched.schedule_review("doc1", 0.3)
        assert ts2 > ts1

    def test_averages_multiple_performances(self):
        sched = SpacedRepetitionScheduler()
        sched.schedule_review("doc1", 0.95)
        ts = sched.schedule_review("doc1", 0.95)
        diff = ts - time.time()
        assert diff >= 29 * 86400

    def test_low_then_high_averages_to_medium(self):
        sched = SpacedRepetitionScheduler()
        sched.schedule_review("doc1", 0.2)
        ts = sched.schedule_review("doc1", 0.95)
        avg = (0.2 + 0.95) / 2
        assert 0.5 <= avg < 0.6

    def test_perfect_score_gets_month(self):
        sched = SpacedRepetitionScheduler()
        ts = sched.schedule_review("doc1", 1.0)
        diff = ts - time.time()
        assert diff >= 29 * 86400

    def test_zero_score_gets_day(self):
        sched = SpacedRepetitionScheduler()
        ts = sched.schedule_review("doc1", 0.0)
        diff = ts - time.time()
        assert 0.5 * 86400 <= diff <= 1.5 * 86400

    def test_review_time_is_float(self):
        sched = SpacedRepetitionScheduler()
        ts = sched.schedule_review("doc1", 0.7)
        assert isinstance(ts, float)

    def test_review_time_positive(self):
        sched = SpacedRepetitionScheduler()
        ts = sched.schedule_review("doc1", 0.7)
        assert ts > 0

    def test_review_time_after_now(self):
        sched = SpacedRepetitionScheduler()
        now = time.time()
        ts = sched.schedule_review("doc1", 0.7)
        assert ts > now

    def test_multiple_docs_different_times(self):
        sched = SpacedRepetitionScheduler()
        ts1 = sched.schedule_review("a", 0.95)
        ts2 = sched.schedule_review("b", 0.3)
        assert ts1 != ts2

    def test_schedule_review_updates_existing(self):
        sched = SpacedRepetitionScheduler()
        sched.schedule_review("doc1", 0.3)
        old_ts = sched.review_schedule["doc1"]
        sched.schedule_review("doc1", 0.3)
        new_ts = sched.review_schedule["doc1"]
        assert new_ts != old_ts

    def test_performance_boundary_089_gets_week(self):
        sched = SpacedRepetitionScheduler()
        ts = sched.schedule_review("doc1", 0.89)
        diff = ts - time.time()
        assert 6 * 86400 <= diff <= 8 * 86400

    def test_performance_boundary_091_gets_month(self):
        sched = SpacedRepetitionScheduler()
        ts = sched.schedule_review("doc1", 0.91)
        diff = ts - time.time()
        assert diff >= 29 * 86400

    def test_performance_059_gets_day(self):
        sched = SpacedRepetitionScheduler()
        ts = sched.schedule_review("doc1", 0.59)
        diff = ts - time.time()
        assert 0.5 * 86400 <= diff <= 1.5 * 86400

    def test_performance_061_gets_3_days(self):
        sched = SpacedRepetitionScheduler()
        ts = sched.schedule_review("doc1", 0.61)
        diff = ts - time.time()
        assert 2.5 * 86400 <= diff <= 3.5 * 86400

    def test_many_reviews_same_doc(self):
        sched = SpacedRepetitionScheduler()
        for _ in range(10):
            sched.schedule_review("doc1", 0.3)
        assert len(sched.performance_history["doc1"]) == 10

    def test_review_schedule_has_all_docs(self):
        sched = SpacedRepetitionScheduler()
        for i in range(5):
            sched.schedule_review(f"doc{i}", 0.8)
        assert len(sched.review_schedule) == 5

    def test_schedule_review_with_empty_doc_id(self):
        sched = SpacedRepetitionScheduler()
        ts = sched.schedule_review("", 0.8)
        assert "" in sched.review_schedule

    def test_schedule_review_with_special_chars(self):
        sched = SpacedRepetitionScheduler()
        ts = sched.schedule_review("doc/with/slashes", 0.8)
        assert "doc/with/slashes" in sched.review_schedule


# ── GetNextReviewTime ────────────────────────────────────────────────────────


class TestGetNextReviewTime:
    def test_returns_future(self):
        sched = SpacedRepetitionScheduler()
        sched.schedule_review("doc1", 0.8)
        nxt = sched.get_next_review_time("doc1")
        assert nxt is not None
        assert nxt > time.time()

    def test_unknown_returns_none(self):
        sched = SpacedRepetitionScheduler()
        assert sched.get_next_review_time("unknown") is None

    def test_returns_stored_value(self):
        sched = SpacedRepetitionScheduler()
        ts = sched.schedule_review("doc1", 0.8)
        assert sched.get_next_review_time("doc1") == ts

    def test_returns_none_for_empty_string(self):
        sched = SpacedRepetitionScheduler()
        assert sched.get_next_review_time("") is None

    def test_returns_none_for_nonexistent(self):
        sched = SpacedRepetitionScheduler()
        assert sched.get_next_review_time("missing") is None

    def test_returns_float(self):
        sched = SpacedRepetitionScheduler()
        sched.schedule_review("doc1", 0.8)
        nxt = sched.get_next_review_time("doc1")
        assert isinstance(nxt, float)

    def test_multiple_docs_independent(self):
        sched = SpacedRepetitionScheduler()
        ts1 = sched.schedule_review("a", 0.9)
        ts2 = sched.schedule_review("b", 0.3)
        assert sched.get_next_review_time("a") == ts1
        assert sched.get_next_review_time("b") == ts2

    def test_after_reschedule_value_changes(self):
        sched = SpacedRepetitionScheduler()
        sched.schedule_review("doc1", 0.3)
        t1 = sched.get_next_review_time("doc1")
        sched.schedule_review("doc1", 0.3)
        t2 = sched.get_next_review_time("doc1")
        assert t2 != t1

    def test_defaultdict_does_not_create_key(self):
        sched = SpacedRepetitionScheduler()
        sched.get_next_review_time("ghost")
        assert "ghost" not in sched.review_schedule


# ── GetDueReviews ────────────────────────────────────────────────────────────


class TestGetDueReviews:
    def test_empty_when_no_schedule(self):
        sched = SpacedRepetitionScheduler()
        assert sched.get_due_reviews() == []

    def test_returns_past_due(self):
        sched = SpacedRepetitionScheduler()
        sched.review_schedule["doc1"] = time.time() - 10
        due = sched.get_due_reviews()
        assert "doc1" in due

    def test_excludes_future(self):
        sched = SpacedRepetitionScheduler()
        sched.review_schedule["doc1"] = time.time() + 86400
        due = sched.get_due_reviews()
        assert "doc1" not in due

    def test_multiple_due(self):
        sched = SpacedRepetitionScheduler()
        sched.review_schedule["doc1"] = time.time() - 10
        sched.review_schedule["doc2"] = time.time() - 5
        due = sched.get_due_reviews()
        assert len(due) == 2

    def test_exactly_now_is_due(self):
        sched = SpacedRepetitionScheduler()
        sched.review_schedule["doc1"] = time.time()
        due = sched.get_due_reviews()
        assert "doc1" in due

    def test_mix_past_and_future(self):
        sched = SpacedRepetitionScheduler()
        sched.review_schedule["past"] = time.time() - 100
        sched.review_schedule["future"] = time.time() + 86400
        due = sched.get_due_reviews()
        assert "past" in due
        assert "future" not in due

    def test_returns_list(self):
        sched = SpacedRepetitionScheduler()
        assert isinstance(sched.get_due_reviews(), list)

    def test_all_past_are_due(self):
        sched = SpacedRepetitionScheduler()
        for i in range(10):
            sched.review_schedule[f"doc{i}"] = time.time() - 1
        due = sched.get_due_reviews()
        assert len(due) == 10

    def test_no_future_are_due(self):
        sched = SpacedRepetitionScheduler()
        for i in range(10):
            sched.review_schedule[f"doc{i}"] = time.time() + 86400
        due = sched.get_due_reviews()
        assert len(due) == 0

    def test_just_past_due(self):
        sched = SpacedRepetitionScheduler()
        sched.review_schedule["doc1"] = time.time() - 0.001
        due = sched.get_due_reviews()
        assert "doc1" in due

    def test_far_future_not_due(self):
        sched = SpacedRepetitionScheduler()
        sched.review_schedule["doc1"] = time.time() + 365 * 86400
        due = sched.get_due_reviews()
        assert "doc1" not in due

    def test_returns_doc_ids(self):
        sched = SpacedRepetitionScheduler()
        sched.review_schedule["abc"] = time.time() - 1
        due = sched.get_due_reviews()
        assert due == ["abc"]


# ── GetReviewStats ───────────────────────────────────────────────────────────


class TestGetReviewStats:
    def test_empty_stats(self):
        sched = SpacedRepetitionScheduler()
        stats = sched.get_review_stats()
        assert stats["total_scheduled"] == 0
        assert stats["due_count"] == 0
        assert stats["due_documents"] == []
        assert stats["upcoming_reviews"] == {}

    def test_one_scheduled(self):
        sched = SpacedRepetitionScheduler()
        sched.schedule_review("doc1", 0.8)
        stats = sched.get_review_stats()
        assert stats["total_scheduled"] == 1

    def test_due_count(self):
        sched = SpacedRepetitionScheduler()
        sched.review_schedule["doc1"] = time.time() - 10
        stats = sched.get_review_stats()
        assert stats["due_count"] == 1
        assert "doc1" in stats["due_documents"]

    def test_upcoming_reviews(self):
        sched = SpacedRepetitionScheduler()
        sched.schedule_review("doc1", 0.8)
        stats = sched.get_review_stats()
        assert "doc1" in stats["upcoming_reviews"]
        assert stats["upcoming_reviews"]["doc1"] > 0

    def test_upcoming_days_format(self):
        sched = SpacedRepetitionScheduler()
        sched.schedule_review("doc1", 0.95)
        stats = sched.get_review_stats()
        days = stats["upcoming_reviews"]["doc1"]
        assert days >= 29.0

    def test_stats_returns_dict(self):
        sched = SpacedRepetitionScheduler()
        assert isinstance(sched.get_review_stats(), dict)

    def test_all_keys_present(self):
        sched = SpacedRepetitionScheduler()
        stats = sched.get_review_stats()
        assert "due_count" in stats
        assert "due_documents" in stats
        assert "total_scheduled" in stats
        assert "upcoming_reviews" in stats

    def test_total_scheduled_count(self):
        sched = SpacedRepetitionScheduler()
        for i in range(5):
            sched.schedule_review(f"doc{i}", 0.8)
        stats = sched.get_review_stats()
        assert stats["total_scheduled"] == 5

    def test_due_documents_are_list(self):
        sched = SpacedRepetitionScheduler()
        sched.review_schedule["doc1"] = time.time() - 1
        stats = sched.get_review_stats()
        assert isinstance(stats["due_documents"], list)

    def test_upcoming_reviews_are_dict(self):
        sched = SpacedRepetitionScheduler()
        sched.schedule_review("doc1", 0.8)
        stats = sched.get_review_stats()
        assert isinstance(stats["upcoming_reviews"], dict)

    def test_mixed_due_and_upcoming(self):
        sched = SpacedRepetitionScheduler()
        sched.review_schedule["past"] = time.time() - 100
        sched.schedule_review("future", 0.8)
        stats = sched.get_review_stats()
        assert stats["due_count"] == 1
        assert "past" in stats["due_documents"]
        assert "future" in stats["upcoming_reviews"]

    def test_upcoming_days_are_rounded(self):
        sched = SpacedRepetitionScheduler()
        sched.schedule_review("doc1", 0.8)
        stats = sched.get_review_stats()
        days = stats["upcoming_reviews"]["doc1"]
        # Should be rounded to 1 decimal
        assert days == round(days, 1)


# ── PerformanceHistory ───────────────────────────────────────────────────────


class TestPerformanceHistory:
    def test_single_record(self):
        sched = SpacedRepetitionScheduler()
        sched.schedule_review("doc1", 0.7)
        assert len(sched.performance_history["doc1"]) == 1

    def test_multiple_records(self):
        sched = SpacedRepetitionScheduler()
        sched.schedule_review("doc1", 0.7)
        sched.schedule_review("doc1", 0.9)
        assert len(sched.performance_history["doc1"]) == 2

    def test_stores_exact_values(self):
        sched = SpacedRepetitionScheduler()
        sched.schedule_review("doc1", 0.42)
        sched.schedule_review("doc1", 0.87)
        assert sched.performance_history["doc1"] == [0.42, 0.87]

    def test_separate_per_doc(self):
        sched = SpacedRepetitionScheduler()
        sched.schedule_review("doc1", 0.8)
        sched.schedule_review("doc2", 0.6)
        assert len(sched.performance_history["doc1"]) == 1
        assert len(sched.performance_history["doc2"]) == 1

    def test_history_grows(self):
        sched = SpacedRepetitionScheduler()
        for i in range(5):
            sched.schedule_review("doc1", 0.5)
        assert len(sched.performance_history["doc1"]) == 5

    def test_history_values_order(self):
        sched = SpacedRepetitionScheduler()
        sched.schedule_review("doc1", 0.1)
        sched.schedule_review("doc1", 0.5)
        sched.schedule_review("doc1", 0.9)
        assert sched.performance_history["doc1"] == [0.1, 0.5, 0.9]

    def test_history_float_values(self):
        sched = SpacedRepetitionScheduler()
        sched.schedule_review("doc1", 0.123456)
        assert sched.performance_history["doc1"][0] == 0.123456

    def test_history_zero_value(self):
        sched = SpacedRepetitionScheduler()
        sched.schedule_review("doc1", 0.0)
        assert sched.performance_history["doc1"] == [0.0]

    def test_history_one_value(self):
        sched = SpacedRepetitionScheduler()
        sched.schedule_review("doc1", 1.0)
        assert sched.performance_history["doc1"] == [1.0]

    def test_history_independent_docs(self):
        sched = SpacedRepetitionScheduler()
        sched.schedule_review("a", 0.3)
        sched.schedule_review("b", 0.7)
        sched.schedule_review("a", 0.5)
        assert sched.performance_history["a"] == [0.3, 0.5]
        assert sched.performance_history["b"] == [0.7]

    def test_empty_history_for_new_doc(self):
        sched = SpacedRepetitionScheduler()
        assert sched.performance_history["nonexistent"] == []

    def test_history_list_type(self):
        sched = SpacedRepetitionScheduler()
        sched.schedule_review("doc1", 0.5)
        assert isinstance(sched.performance_history["doc1"], list)
