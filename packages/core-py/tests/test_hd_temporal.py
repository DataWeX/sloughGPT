"""Tests for domains.soul.quantum — HyperdimensionalProcessor, TemporalReasoningEngine."""

import pytest

from domains.soul.quantum import HyperdimensionalProcessor, TemporalReasoningEngine


# ── HyperdimensionalProcessor construction ──────────────────────────────────

class TestHyperdimensionalProcessorConstruction:
    def test_default_dim(self):
        hp = HyperdimensionalProcessor()
        assert hp.dim == 10000

    def test_custom_dim(self):
        hp = HyperdimensionalProcessor(dim=500)
        assert hp.dim == 500

    def test_small_dim(self):
        hp = HyperdimensionalProcessor(dim=10)
        assert hp.dim == 10

    def test_vectors_starts_empty(self):
        hp = HyperdimensionalProcessor()
        assert hp.vectors == {}


# ── Encode ───────────────────────────────────────────────────────────────────

class TestEncode:
    def test_encode_returns_list(self):
        hp = HyperdimensionalProcessor(dim=100)
        v = hp.encode("hello")
        assert isinstance(v, list)

    def test_encode_correct_length(self):
        hp = HyperdimensionalProcessor(dim=100)
        v = hp.encode("hello")
        assert len(v) == 100

    def test_encode_binary_values(self):
        hp = HyperdimensionalProcessor(dim=100)
        v = hp.encode("hello")
        assert all(x in (-1, 1) for x in v)

    def test_encode_caching(self):
        hp = HyperdimensionalProcessor(dim=100)
        v1 = hp.encode("hello")
        v2 = hp.encode("hello")
        assert v1 is v2

    def test_encode_different_symbols_different_vectors(self):
        hp = HyperdimensionalProcessor(dim=100)
        v1 = hp.encode("alpha")
        v2 = hp.encode("beta")
        assert v1 != v2

    def test_encode_stores_in_dict(self):
        hp = HyperdimensionalProcessor(dim=50)
        hp.encode("stored")
        assert "stored" in hp.vectors
        assert len(hp.vectors["stored"]) == 50

    def test_encode_empty_string(self):
        hp = HyperdimensionalProcessor(dim=32)
        v = hp.encode("")
        assert len(v) == 32
        assert all(x in (-1, 1) for x in v)

    def test_encode_long_string(self):
        hp = HyperdimensionalProcessor(dim=64)
        v = hp.encode("a" * 1000)
        assert len(v) == 64

    def test_encode_many_symbols(self):
        hp = HyperdimensionalProcessor(dim=32)
        for i in range(50):
            hp.encode(f"sym_{i}")
        assert len(hp.vectors) == 50


# ── Encode Text ──────────────────────────────────────────────────────────────

class TestEncodeText:
    def test_encode_text_single_word(self):
        hp = HyperdimensionalProcessor(dim=100)
        v = hp.encode_text("hello")
        assert len(v) == 100

    def test_encode_text_multi_word(self):
        hp = HyperdimensionalProcessor(dim=100)
        v = hp.encode_text("hello world")
        assert len(v) == 100

    def test_encode_text_bundled(self):
        hp = HyperdimensionalProcessor(dim=100)
        v1 = hp.encode("hello")
        v2 = hp.encode("world")
        v_text = hp.encode_text("hello world")
        assert len(v_text) == 100
        assert v_text != v1

    def test_encode_text_empty(self):
        hp = HyperdimensionalProcessor(dim=32)
        v = hp.encode_text("")
        assert len(v) == 32
        assert all(x == 0 for x in v)

    def test_encode_text_long(self):
        hp = HyperdimensionalProcessor(dim=64)
        text = " ".join(["word"] * 50)
        v = hp.encode_text(text)
        assert len(v) == 64


# ── Bundle ───────────────────────────────────────────────────────────────────

class TestBundle:
    def test_bundle_returns_correct_dim(self):
        hp = HyperdimensionalProcessor(dim=100)
        v1 = hp.encode("a")
        v2 = hp.encode("b")
        result = hp.bundle([v1, v2])
        assert len(result) == 100

    def test_bundle_binary_output(self):
        hp = HyperdimensionalProcessor(dim=100)
        v1 = hp.encode("a")
        v2 = hp.encode("b")
        result = hp.bundle([v1, v2])
        assert all(x in (-1, 1) for x in result)

    def test_bundle_single_vector(self):
        hp = HyperdimensionalProcessor(dim=50)
        v = hp.encode("only")
        result = hp.bundle([v])
        assert len(result) == 50

    def test_bundle_empty_list(self):
        hp = HyperdimensionalProcessor(dim=32)
        result = hp.bundle([])
        assert len(result) == 32
        assert all(x == 0 for x in result)

    def test_bundle_many_vectors(self):
        hp = HyperdimensionalProcessor(dim=64)
        vectors = [hp.encode(f"v{i}") for i in range(20)]
        result = hp.bundle(vectors)
        assert len(result) == 64
        assert all(x in (-1, 1) for x in result)

    def test_bundle_result_differs_from_inputs(self):
        hp = HyperdimensionalProcessor(dim=100)
        v1 = hp.encode("a")
        v2 = hp.encode("b")
        result = hp.bundle([v1, v2])
        assert result != v1
        assert result != v2


# ── Bind ─────────────────────────────────────────────────────────────────────

class TestBind:
    def test_bind_returns_correct_dim(self):
        hp = HyperdimensionalProcessor(dim=100)
        v1 = hp.encode("a")
        v2 = hp.encode("b")
        bound = hp.bind(v1, v2)
        assert len(bound) == 100

    def test_bind_binary_output(self):
        hp = HyperdimensionalProcessor(dim=100)
        v1 = hp.encode("a")
        v2 = hp.encode("b")
        bound = hp.bind(v1, v2)
        assert all(x in (-1, 1) for x in bound)

    def test_bind_symmetric(self):
        hp = HyperdimensionalProcessor(dim=100)
        v1 = hp.encode("a")
        v2 = hp.encode("b")
        assert hp.bind(v1, v2) == hp.bind(v2, v1)

    def test_bind_self_produces_ones(self):
        hp = HyperdimensionalProcessor(dim=100)
        v = hp.encode("self")
        bound = hp.bind(v, v)
        assert all(x == 1.0 for x in bound)

    def test_bind_dimension_mismatch_raises(self):
        hp = HyperdimensionalProcessor(dim=100)
        v1 = hp.encode("a")
        with pytest.raises(ValueError, match="same dimension"):
            hp.bind(v1, [1, 2, 3])

    def test_bind_result_differs_from_inputs(self):
        hp = HyperdimensionalProcessor(dim=100)
        v1 = hp.encode("x")
        v2 = hp.encode("y")
        bound = hp.bind(v1, v2)
        assert bound != v1
        assert bound != v2


# ── Similarity ───────────────────────────────────────────────────────────────

class TestSimilarity:
    def test_self_similarity_near_one(self):
        hp = HyperdimensionalProcessor(dim=1000)
        v = hp.encode("hello")
        sim = hp.similarity(v, v)
        assert sim > 0.99

    def test_similarity_range(self):
        hp = HyperdimensionalProcessor(dim=1000)
        v1 = hp.encode("alpha")
        v2 = hp.encode("beta")
        sim = hp.similarity(v1, v2)
        assert -1.0 <= sim <= 1.0

    def test_different_vectors_lower_similarity(self):
        hp = HyperdimensionalProcessor(dim=1000)
        v = hp.encode("hello")
        sim_self = hp.similarity(v, v)
        sim_diff = hp.similarity(v, hp.encode("world"))
        assert sim_diff < sim_self

    def test_similarity_dimension_mismatch_raises(self):
        hp = HyperdimensionalProcessor(dim=100)
        v = hp.encode("a")
        with pytest.raises(ValueError, match="same dimension"):
            hp.similarity(v, [1, 2, 3])


# ── Unbind ───────────────────────────────────────────────────────────────────

class TestUnbind:
    def test_unbind_inverse_of_bind(self):
        hp = HyperdimensionalProcessor(dim=200)
        v1 = hp.encode("a")
        v2 = hp.encode("b")
        bound = hp.bind(v1, v2)
        unbound = hp.unbind(bound, v1)
        assert unbound == hp.bind(bound, v1)

    def test_unbind_returns_correct_dim(self):
        hp = HyperdimensionalProcessor(dim=100)
        v1 = hp.encode("x")
        v2 = hp.encode("y")
        bound = hp.bind(v1, v2)
        result = hp.unbind(bound, v1)
        assert len(result) == 100


# ── Cleanup ──────────────────────────────────────────────────────────────────

class TestCleanup:
    def test_cleanup_removes_zero_vectors(self):
        hp = HyperdimensionalProcessor(dim=32)
        hp.encode("keep")
        hp.vectors["zero"] = [0] * 32
        removed = hp.cleanup()
        assert removed == 1
        assert "zero" not in hp.vectors
        assert "keep" in hp.vectors

    def test_cleanup_nothing_to_remove(self):
        hp = HyperdimensionalProcessor(dim=32)
        hp.encode("a")
        hp.encode("b")
        removed = hp.cleanup()
        assert removed == 0

    def test_cleanup_empty(self):
        hp = HyperdimensionalProcessor(dim=32)
        removed = hp.cleanup()
        assert removed == 0


# ── TemporalReasoningEngine construction ────────────────────────────────────

class TestTemporalReasoningEngineConstruction:
    def test_default_timeline_depth(self):
        tre = TemporalReasoningEngine()
        assert tre.timeline_depth == 5

    def test_custom_timeline_depth(self):
        tre = TemporalReasoningEngine(timeline_depth=10)
        assert tre.timeline_depth == 10

    def test_current_timeline_zero(self):
        tre = TemporalReasoningEngine()
        assert tre.current_timeline == 0

    def test_timelines_initialized_empty(self):
        tre = TemporalReasoningEngine()
        assert len(tre.timelines) == 5
        for tl in tre.timelines:
            assert tl == []

    def test_branch_points_empty(self):
        tre = TemporalReasoningEngine()
        assert tre.branch_points == []


# ── Add Event ────────────────────────────────────────────────────────────────

class TestAddEvent:
    def test_add_single_event(self):
        tre = TemporalReasoningEngine()
        tre.add_event({"type": "observation", "content": "saw a cat"})
        assert len(tre.timelines[0]) == 1

    def test_add_multiple_events(self):
        tre = TemporalReasoningEngine()
        tre.add_event({"type": "a", "content": "x"})
        tre.add_event({"type": "b", "content": "y"})
        assert len(tre.timelines[0]) == 2

    def test_event_has_timeline_field(self):
        tre = TemporalReasoningEngine()
        tre.add_event({"type": "obs"})
        assert tre.timelines[0][0]["timeline"] == 0

    def test_event_has_timestamp(self):
        tre = TemporalReasoningEngine()
        tre.add_event({"type": "obs"})
        assert "timestamp" in tre.timelines[0][0]

    def test_add_event_to_specific_timeline(self):
        tre = TemporalReasoningEngine()
        tre.add_event({"type": "obs"}, timeline=2)
        assert len(tre.timelines[2]) == 1
        assert len(tre.timelines[0]) == 0

    def test_add_event_out_of_range_ignored(self):
        tre = TemporalReasoningEngine()
        tre.add_event({"type": "obs"}, timeline=100)
        assert all(len(tl) == 0 for tl in tre.timelines)

    def test_add_event_negative_index_ignored(self):
        tre = TemporalReasoningEngine()
        tre.add_event({"type": "obs"}, timeline=-1)
        assert all(len(tl) == 0 for tl in tre.timelines)

    def test_event_preserves_original_fields(self):
        tre = TemporalReasoningEngine()
        tre.add_event({"type": "obs", "extra": "data"})
        event = tre.timelines[0][0]
        assert event["type"] == "obs"
        assert event["extra"] == "data"


# ── Branch ───────────────────────────────────────────────────────────────────

class TestBranch:
    def test_branch_returns_next_timeline(self):
        tre = TemporalReasoningEngine()
        new_tl = tre.branch("alternate path")
        assert new_tl == 1

    def test_branch_does_not_change_current(self):
        tre = TemporalReasoningEngine()
        tre.branch("alt")
        assert tre.current_timeline == 0

    def test_branch_copies_events(self):
        tre = TemporalReasoningEngine()
        tre.add_event({"type": "a"})
        tre.branch("alt")
        assert len(tre.timelines[1]) == 1

    def test_branch_records_branch_point(self):
        tre = TemporalReasoningEngine()
        tre.branch("reason")
        assert len(tre.branch_points) == 1
        bp = tre.branch_points[0]
        assert bp["from_timeline"] == 0
        assert bp["to_timeline"] == 1
        assert bp["condition"] == "reason"

    def test_branch_has_timestamp(self):
        tre = TemporalReasoningEngine()
        tre.branch("test")
        assert "timestamp" in tre.branch_points[0]

    def test_branch_wraps_around(self):
        tre = TemporalReasoningEngine(timeline_depth=3)
        tre.branch("b1")  # 0 -> 1
        tre.switch_timeline(1)
        tre.branch("b2")  # 1 -> 2
        tre.switch_timeline(2)
        tre.branch("b3")  # 2 -> 0
        assert tre.timelines[0] == tre.timelines[2]

    def test_branch_from_non_zero_timeline(self):
        tre = TemporalReasoningEngine()
        tre.switch_timeline(2)
        new_tl = tre.branch("from_2")
        assert new_tl == 3


# ── Switch Timeline ──────────────────────────────────────────────────────────

class TestSwitchTimeline:
    def test_switch_returns_true(self):
        tre = TemporalReasoningEngine()
        result = tre.switch_timeline(1)
        assert result is True

    def test_switch_updates_current(self):
        tre = TemporalReasoningEngine()
        tre.switch_timeline(3)
        assert tre.current_timeline == 3

    def test_switch_out_of_range_returns_false(self):
        tre = TemporalReasoningEngine()
        result = tre.switch_timeline(100)
        assert result is False

    def test_switch_negative_returns_false(self):
        tre = TemporalReasoningEngine()
        result = tre.switch_timeline(-1)
        assert result is False

    def test_switch_does_not_change_current_on_failure(self):
        tre = TemporalReasoningEngine()
        tre.switch_timeline(100)
        assert tre.current_timeline == 0

    def test_switch_to_zero(self):
        tre = TemporalReasoningEngine()
        tre.switch_timeline(3)
        tre.switch_timeline(0)
        assert tre.current_timeline == 0


# ── Get Current Events ──────────────────────────────────────────────────────

class TestGetCurrentEvents:
    def test_returns_list(self):
        tre = TemporalReasoningEngine()
        events = tre.get_current_events()
        assert isinstance(events, list)

    def test_returns_recent_n(self):
        tre = TemporalReasoningEngine()
        for i in range(10):
            tre.add_event({"type": "e", "i": i})
        events = tre.get_current_events(n=3)
        assert len(events) == 3

    def test_returns_all_when_fewer(self):
        tre = TemporalReasoningEngine()
        tre.add_event({"type": "a"})
        tre.add_event({"type": "b"})
        events = tre.get_current_events(n=10)
        assert len(events) == 2

    def test_default_n_is_10(self):
        tre = TemporalReasoningEngine()
        for i in range(20):
            tre.add_event({"type": "e", "i": i})
        events = tre.get_current_events()
        assert len(events) == 10

    def test_empty_timeline(self):
        tre = TemporalReasoningEngine()
        events = tre.get_current_events()
        assert events == []


# ── Merge Timelines ─────────────────────────────────────────────────────────

class TestMergeTimelines:
    def test_merge_empty_timelines(self):
        tre = TemporalReasoningEngine()
        merged = tre.merge_timelines(0, 1)
        assert merged == []

    def test_merge_one_empty_one_nonempty(self):
        tre = TemporalReasoningEngine()
        tre.add_event({"type": "a"})
        merged = tre.merge_timelines(0, 1)
        assert len(merged) == 1

    def test_merge_both_nonempty(self):
        tre = TemporalReasoningEngine()
        tre.add_event({"type": "a"})
        tre.branch("alt")
        tre.add_event({"type": "b"})
        merged = tre.merge_timelines(0, 1)
        assert len(merged) == 3

    def test_merge_sorted_by_timestamp(self):
        tre = TemporalReasoningEngine()
        tre.add_event({"type": "first"})
        tre.branch("alt")
        tre.add_event({"type": "second"})
        merged = tre.merge_timelines(0, 1)
        timestamps = [e["timestamp"] for e in merged]
        assert timestamps == sorted(timestamps)

    def test_merge_same_timeline(self):
        tre = TemporalReasoningEngine()
        tre.add_event({"type": "a"})
        tre.add_event({"type": "b"})
        merged = tre.merge_timelines(0, 0)
        assert len(merged) == 4
