"""Tests for domains.cognitive.core — CognitiveCore, ThinkingMode, ReasoningType."""

import time
from domains.cognitive.core import (
    CognitiveCore, ThinkingMode, ReasoningType,
    ThoughtProcess, CreativeIdea, ReasoningChain,
)


class TestThinkingMode:
    def test_all_members(self):
        assert len(ThinkingMode) == 5
    def test_values(self):
        assert ThinkingMode.ANALYTICAL.value == "analytical"


class TestReasoningType:
    def test_all_members(self):
        assert len(ReasoningType) == 5


class TestThoughtProcess:
    def test_fields(self):
        tp = ThoughtProcess(
            id="t1", mode=ThinkingMode.ANALYTICAL,
            reasoning_type=ReasoningType.DEDUCTIVE,
            input_prompt="test", thought_content="thinking...",
            confidence=0.85, creativity_score=0.7, logical_score=0.8,
            timestamp=time.time(), processing_time=0.1,
        )
        assert tp.id == "t1"


class TestCognitiveCore:
    def test_think_analytical(self):
        core = CognitiveCore()
        t = core.think("prompt", ThinkingMode.ANALYTICAL)
        assert "Analysis" in t.thought_content
        assert len(core.thought_history) == 1

    def test_think_creative(self):
        core = CognitiveCore()
        t = core.think("prompt", ThinkingMode.CREATIVE)
        assert "Creative" in t.thought_content

    def test_think_critical(self):
        core = CognitiveCore()
        t = core.think("prompt", ThinkingMode.CRITICAL)
        assert "Critical" in t.thought_content

    def test_think_strategic(self):
        core = CognitiveCore()
        t = core.think("prompt", ThinkingMode.STRATEGIC)
        assert "Strategic" in t.thought_content

    def test_think_reflective(self):
        core = CognitiveCore()
        t = core.think("prompt", ThinkingMode.REFLECTIVE)
        assert "Reflection" in t.thought_content

    def test_generate_idea(self):
        core = CognitiveCore()
        idea = core.generate_idea("quantum", "tech")
        assert idea.concept == "quantum"
        assert len(core.ideas) == 1

    def test_reason(self):
        core = CognitiveCore()
        chain = core.reason("why?", ReasoningType.CAUSAL)
        assert chain.reasoning_type == ReasoningType.CAUSAL

    def test_get_recent_thoughts(self):
        core = CognitiveCore()
        for i in range(5):
            core.think(f"p{i}")
        assert len(core.get_recent_thoughts(3)) == 3

    def test_get_statistics(self):
        core = CognitiveCore()
        core.think("a")
        core.generate_idea("b")
        stats = core.get_statistics()
        assert stats["total_thoughts"] == 1
        assert stats["total_ideas"] == 1
