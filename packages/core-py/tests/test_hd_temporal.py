"""Tests for domains.soul.quantum — HyperdimensionalProcessor, TemporalReasoningEngine."""

from domains.soul.quantum import HyperdimensionalProcessor, TemporalReasoningEngine


class TestHyperdimensionalProcessor:
    def test_init(self):
        hp = HyperdimensionalProcessor(dim=1000)
        assert hp.dim == 1000
        assert hp.vectors == {}

    def test_encode(self):
        hp = HyperdimensionalProcessor(dim=100)
        v = hp.encode("hello")
        assert len(v) == 100
        assert all(x in (-1, 1) for x in v)

    def test_encode_caching(self):
        hp = HyperdimensionalProcessor(dim=100)
        v1 = hp.encode("hello")
        v2 = hp.encode("hello")
        assert v1 is v2

    def test_encode_text(self):
        hp = HyperdimensionalProcessor(dim=100)
        v = hp.encode_text("hello world")
        assert len(v) == 100

    def test_bundle(self):
        hp = HyperdimensionalProcessor(dim=100)
        v1 = hp.encode("a")
        v2 = hp.encode("b")
        result = hp.bundle([v1, v2])
        assert len(result) == 100

    def test_bind(self):
        hp = HyperdimensionalProcessor(dim=100)
        v1 = hp.encode("a")
        v2 = hp.encode("b")
        bound = hp.bind(v1, v2)
        assert len(bound) == 100

    def test_similarity(self):
        hp = HyperdimensionalProcessor(dim=1000)
        v = hp.encode("hello")
        sim = hp.similarity(v, v)
        assert sim > 0.99


class TestTemporalReasoningEngine:
    def test_init(self):
        tre = TemporalReasoningEngine()
        assert tre.current_timeline == 0
        assert len(tre.timelines) == 5

    def test_add_event(self):
        tre = TemporalReasoningEngine()
        tre.add_event({"type": "observation", "content": "saw a cat"})
        assert len(tre.timelines[0]) == 1

    def test_branch(self):
        tre = TemporalReasoningEngine()
        tre.add_event({"type": "observation", "content": "test"})
        new_timeline = tre.branch("alternate path")
        assert new_timeline == 1
        assert tre.current_timeline == 0
        assert len(tre.timelines[1]) == 1

    def test_switch_timeline(self):
        tre = TemporalReasoningEngine()
        tre.add_event({"type": "observation", "content": "test"})
        tre.branch("alt")
        result = tre.switch_timeline(1)
        assert result is True
        assert tre.current_timeline == 1

    def test_get_current_events(self):
        tre = TemporalReasoningEngine()
        tre.add_event({"type": "a", "content": "x"})
        events = tre.get_current_events()
        assert len(events) == 1
