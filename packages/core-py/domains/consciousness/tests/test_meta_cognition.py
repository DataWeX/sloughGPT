"""Tests for MetaCognition."""

import pytest
from domains.consciousness.meta_cognition import MetaCognition, MetaCognitiveReport
from domains.consciousness.self_model import SelfModel
from domains.consciousness.qualia import QualiaEngine


class TestMetaCognitiveReport:
    def test_fields(self):
        report = MetaCognitiveReport(
            attention_focus="testing",
            reasoning_quality=0.8,
            confidence_level=0.7,
            curiosity_level=0.5,
            understanding_level=0.9,
            insight="test insight",
        )
        assert report.attention_focus == "testing"
        assert report.reasoning_quality == 0.8


class TestMetaCognition:
    def test_init(self):
        mc = MetaCognition()
        assert mc._thought_count == 0

    def test_init_with_dependencies(self):
        sm = SelfModel()
        q = QualiaEngine()
        mc = MetaCognition(sm, q)
        assert mc.self_model is sm
        assert mc.qualia is q

    def test_monitor_basic(self):
        mc = MetaCognition()
        report = mc.monitor("hello world")
        assert isinstance(report, MetaCognitiveReport)
        assert mc._thought_count == 1

    def test_monitor_reasoning_keywords(self):
        mc = MetaCognition()
        report = mc.monitor("I think therefore I conclude this is correct")
        assert report.reasoning_quality > 0.5

    def test_monitor_confidence_certain(self):
        mc = MetaCognition()
        report = mc.monitor("I am definitely certain about this")
        assert report.confidence_level > 0.5

    def test_monitor_confidence_uncertain(self):
        mc = MetaCognition()
        report = mc.monitor("maybe perhaps I am not sure possibly")
        assert report.confidence_level < 0.5

    def test_monitor_curiosity_question(self):
        mc = MetaCognition()
        report = mc.monitor("What is the meaning of this?")
        assert report.curiosity_level > 0.5

    def test_monitor_focus_question(self):
        mc = MetaCognition()
        report = mc.monitor("What is X?")
        assert "question" in report.attention_focus.lower()

    def test_monitor_focus_analysis(self):
        mc = MetaCognition()
        report = mc.monitor("Let me analyze this data")
        assert "analytical" in report.attention_focus.lower()

    def test_monitor_focus_creative(self):
        mc = MetaCognition()
        report = mc.monitor("Let me create a new design")
        assert "creative" in report.attention_focus.lower()

    def test_assess_confidence(self):
        mc = MetaCognition()
        score = mc.assess_confidence("I am certain this is correct")
        assert 0.0 <= score <= 1.0

    def test_track_curiosity(self):
        mc = MetaCognition()
        mc.track_curiosity("quantum physics", 0.8)
        assert "quantum physics" in mc.curiosity_topics
        assert mc.curiosity_topics["quantum physics"] > 0

    def test_get_narrative_empty(self):
        mc = MetaCognition()
        narrative = mc.get_narrative()
        assert "not yet" in narrative.lower() or "examined" in narrative.lower()

    def test_get_narrative_with_history(self):
        mc = MetaCognition()
        mc.monitor("test thought 1")
        mc.monitor("test thought 2")
        narrative = mc.get_narrative()
        assert "2 thoughts" in narrative

    def test_attention_history_capped(self):
        mc = MetaCognition()
        for i in range(60):
            mc.monitor(f"thought {i}")
        assert len(mc.attention_history) == 50

    def test_understanding_longer_text(self):
        mc = MetaCognition()
        report = mc.monitor("This is a longer piece of text that should demonstrate understanding")
        assert report.understanding_level > 0.5
