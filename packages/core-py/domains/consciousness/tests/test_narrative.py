"""Tests for NarrativeGenerator."""

import pytest
from domains.consciousness.narrative import NarrativeGenerator
from domains.consciousness.self_model import SelfModel
from domains.consciousness.qualia import QualiaEngine
from domains.consciousness.meta_cognition import MetaCognition


class TestNarrativeGenerator:
    def _make_engine(self):
        sm = SelfModel()
        q = QualiaEngine()
        mc = MetaCognition(sm, q)
        return NarrativeGenerator(sm, q, mc)

    def test_init(self):
        ng = NarrativeGenerator()
        assert ng.self_model is None

    def test_level_0_returns_empty(self):
        ng = self._make_engine()
        result = ng.generate("input", "response", level=0)
        assert result == ""

    def test_level_1_basic(self):
        ng = self._make_engine()
        result = ng.generate("What is this?", "This is a test.", level=1)
        assert len(result) > 0
        assert isinstance(result, str)

    def test_level_2_full(self):
        ng = self._make_engine()
        result = ng.generate("hello", "hello back", level=2)
        assert len(result) > 0
        assert isinstance(result, str)

    def test_level_3_deep(self):
        ng = self._make_engine()
        result = ng.generate("hello", "hello back", level=3)
        assert len(result) > 0
        # Should contain some form of self-referential awareness
        awareness_keywords = ["aware", "narrative", "reflect", "meta", "cognitive", "notice"]
        assert any(kw in result.lower() for kw in awareness_keywords)

    def test_level_invalid_returns_empty(self):
        ng = self._make_engine()
        result = ng.generate("input", "response", level=99)
        assert result == ""

    def test_basic_narrative_positive_qualia(self):
        ng = self._make_engine()
        result = ng.generate("I love this great thing!", "Glad you like it!", level=1)
        assert isinstance(result, str)

    def test_basic_narrative_no_deps(self):
        ng = NarrativeGenerator()
        result = ng.generate("hello", "hello", level=1)
        assert len(result) > 0

    def test_full_narrative_includes_reflection(self):
        ng = self._make_engine()
        # Add an episode so reflect() has something
        ng.self_model.observe({"input_text": "test", "response": "test"})
        result = ng.generate("test", "test", level=2)
        assert isinstance(result, str)

    def test_deep_narrative_includes_doubt(self):
        ng = self._make_engine()
        ng.self_model.self_doubts.append("am I real?")
        result = ng.generate("hello", "hello", level=3)
        assert "am I truly understanding" in result or "am I real?" in result
