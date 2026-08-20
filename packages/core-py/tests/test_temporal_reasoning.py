"""Meaningful tests for TemporalReasoningEngine — timeline management, branching, switching, merging."""

from domains.soul.quantum import TemporalReasoningEngine


class TestTemporalReasoningInit:
    def test_default_timelines(self):
        tre = TemporalReasoningEngine()
        assert tre.timeline_depth == 5
        assert len(tre.timelines) == 5
        assert tre.current_timeline == 0
        assert tre.branch_points == []


class TestAddEvent:
    def test_add_event_default_timeline(self):
        tre = TemporalReasoningEngine()
        tre.add_event({"type": "start"})
        assert len(tre.timelines[0]) == 1
        assert tre.timelines[0][0]["type"] == "start"
        assert "timestamp" in tre.timelines[0][0]

    def test_add_event_specific_timeline(self):
        tre = TemporalReasoningEngine()
        tre.add_event({"type": "A"}, timeline=2)
        assert len(tre.timelines[2]) == 1
        assert tre.timelines[2][0]["timeline"] == 2

    def test_add_event_out_of_range(self):
        tre = TemporalReasoningEngine()
        tre.add_event({"type": "A"}, timeline=10)
        # Should not crash, timeline 10 is out of range
        assert all(len(t) == 0 for t in tre.timelines)

    def test_add_multiple_events(self):
        tre = TemporalReasoningEngine()
        for i in range(5):
            tre.add_event({"step": i})
        assert len(tre.timelines[0]) == 5


class TestBranch:
    def test_branch_creates_new_timeline(self):
        tre = TemporalReasoningEngine()
        tre.add_event({"step": 0})
        new_tl = tre.branch("if user clicks")
        assert new_tl == 1
        assert len(tre.branch_points) == 1
        assert tre.branch_points[0]["condition"] == "if user clicks"

    def test_branch_copies_state(self):
        tre = TemporalReasoningEngine()
        tre.add_event({"step": 0})
        tre.add_event({"step": 1})
        new_tl = tre.branch("condition")
        # New timeline should have the same events
        assert len(tre.timelines[new_tl]) == 2

    def test_branch_wraps_around(self):
        tre = TemporalReasoningEngine(timeline_depth=3)
        tre.branch("b1")  # (0+1)%3 = 1
        tre.switch_timeline(2)
        tre.branch("b2")  # (2+1)%3 = 0
        assert tre.branch_points[1]["to_timeline"] == 0


class TestSwitchTimeline:
    def test_switch_valid(self):
        tre = TemporalReasoningEngine()
        result = tre.switch_timeline(3)
        assert result is True
        assert tre.current_timeline == 3

    def test_switch_invalid(self):
        tre = TemporalReasoningEngine()
        result = tre.switch_timeline(10)
        assert result is False
        assert tre.current_timeline == 0

    def test_switch_negative(self):
        tre = TemporalReasoningEngine()
        result = tre.switch_timeline(-1)
        assert result is False


class TestGetCurrentEvents:
    def test_get_recent_events(self):
        tre = TemporalReasoningEngine()
        for i in range(20):
            tre.add_event({"step": i})
        recent = tre.get_current_events(n=5)
        assert len(recent) == 5
        assert recent[0]["step"] == 15

    def test_get_all_events(self):
        tre = TemporalReasoningEngine()
        tre.add_event({"step": 0})
        tre.add_event({"step": 1})
        recent = tre.get_current_events(n=100)
        assert len(recent) == 2


class TestMergeTimelines:
    def test_merge(self):
        tre = TemporalReasoningEngine()
        tre.add_event({"step": 0, "source": "a"}, timeline=0)
        tre.add_event({"step": 1, "source": "b"}, timeline=1)
        merged = tre.merge_timelines(0, 1)
        assert len(merged) == 2
        # Should be sorted by timestamp
        assert merged[0]["timestamp"] <= merged[1]["timestamp"]

    def test_merge_empty(self):
        tre = TemporalReasoningEngine()
        merged = tre.merge_timelines(0, 1)
        assert merged == []
