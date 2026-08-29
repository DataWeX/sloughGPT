"""Tests for domains.cognitive.core — ThinkingMode, ReasoningType, ThoughtProcess, CreativeIdea, ReasoningChain."""

from domains.cognitive.core import (
    ThinkingMode, ReasoningType, ThoughtProcess, CreativeIdea, ReasoningChain,
)


class TestThinkingMode:
    def test_all_members(self):
        assert len(ThinkingMode) == 5
    def test_values(self):
        assert ThinkingMode.ANALYTICAL.value == "analytical"
        assert ThinkingMode.CREATIVE.value == "creative"


class TestReasoningType:
    def test_all_members(self):
        assert len(ReasoningType) == 5
    def test_values(self):
        assert ReasoningType.DEDUCTIVE.value == "deductive"
        assert ReasoningType.INDUCTIVE.value == "inductive"
        assert ReasoningType.ANALOGICAL.value == "analogical"


class TestThoughtProcess:
    def test_fields(self):
        tp = ThoughtProcess(
            id="t1", mode=ThinkingMode.ANALYTICAL,
            reasoning_type=ReasoningType.DEDUCTIVE,
            input_prompt="test", thought_content="thought",
            confidence=0.9, creativity_score=0.5, logical_score=0.8,
            timestamp=1.0, processing_time=0.5,
        )
        assert tp.id == "t1"
        assert tp.confidence == 0.9


class TestCreativeIdea:
    def test_fields(self):
        ci = CreativeIdea(
            id="i1", concept="test", description="desc",
            novelty_score=0.8, feasibility_score=0.7, creativity_score=0.9,
            category="tech", tags=["ai"], timestamp=1.0,
        )
        assert ci.id == "i1"
        assert ci.novelty_score == 0.8


class TestReasoningChain:
    def test_fields(self):
        rc = ReasoningChain(
            id="r1", question="why?", reasoning_steps=["step1"],
            conclusion="because", confidence=0.85,
            reasoning_type=ReasoningType.ANALOGICAL,
            evidence=["ev1"], timestamp=1.0,
        )
        assert rc.id == "r1"
        assert len(rc.reasoning_steps) == 1
