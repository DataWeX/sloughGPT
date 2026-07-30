"""Tests for CognitiveCore."""

import time
import pytest
from domains.cognitive.core import (
    CognitiveCore,
    ThinkingMode,
    ReasoningType,
    ThoughtProcess,
    CreativeIdea,
    ReasoningChain,
)


class TestThinkingMode:
    def test_values(self):
        assert ThinkingMode.ANALYTICAL.value == "analytical"
        assert ThinkingMode.CREATIVE.value == "creative"
        assert ThinkingMode.CRITICAL.value == "critical"
        assert ThinkingMode.STRATEGIC.value == "strategic"
        assert ThinkingMode.REFLECTIVE.value == "reflective"

    def test_members_count(self):
        assert len(ThinkingMode) == 5


class TestReasoningType:
    def test_values(self):
        assert ReasoningType.DEDUCTIVE.value == "deductive"
        assert ReasoningType.INDUCTIVE.value == "inductive"
        assert ReasoningType.ABDUCTIVE.value == "abductive"
        assert ReasoningType.CAUSAL.value == "causal"
        assert ReasoningType.ANALOGICAL.value == "analogical"

    def test_members_count(self):
        assert len(ReasoningType) == 5


class TestThoughtProcessDataclass:
    def test_fields(self):
        tp = ThoughtProcess(
            id="t1", mode=ThinkingMode.ANALYTICAL, reasoning_type=ReasoningType.DEDUCTIVE,
            input_prompt="test", thought_content="content", confidence=0.9,
            creativity_score=0.5, logical_score=0.8, timestamp=100.0, processing_time=0.1,
        )
        assert tp.id == "t1"
        assert tp.mode == ThinkingMode.ANALYTICAL
        assert tp.input_prompt == "test"

    def test_default_not_provided(self):
        with pytest.raises(TypeError):
            ThoughtProcess()


class TestCreativeIdeaDataclass:
    def test_fields(self):
        ci = CreativeIdea(
            id="i1", concept="c", description="d", novelty_score=0.8,
            feasibility_score=0.6, creativity_score=0.9, category="cat",
            tags=["a", "b"], timestamp=100.0,
        )
        assert ci.concept == "c"
        assert ci.tags == ["a", "b"]

    def test_missing_required(self):
        with pytest.raises(TypeError):
            CreativeIdea()


class TestReasoningChainDataclass:
    def test_fields(self):
        rc = ReasoningChain(
            id="r1", question="q", reasoning_steps=["s1"], conclusion="c",
            confidence=0.7, reasoning_type=ReasoningType.INDUCTIVE,
            evidence=["e1"], timestamp=200.0,
        )
        assert rc.question == "q"
        assert rc.reasoning_type == ReasoningType.INDUCTIVE
        assert rc.conclusion == "c"

    def test_missing_required(self):
        with pytest.raises(TypeError):
            ReasoningChain()


class TestCognitiveCore:
    def test_initial_state(self):
        cc = CognitiveCore()
        assert cc.thought_history == []
        assert cc.ideas == []
        assert cc.reasoning_chains == []
        assert cc.db_path == "slo_cognitive_core.db"
        assert cc.logger.name == "slo.cognitive_core"

    def test_think_returns_thought_process(self):
        cc = CognitiveCore()
        tp = cc.think("solve problem")
        assert isinstance(tp, ThoughtProcess)
        assert tp.input_prompt == "solve problem"
        assert tp.mode == ThinkingMode.ANALYTICAL
        assert "Analysis" in tp.thought_content

    def test_think_creative_mode(self):
        cc = CognitiveCore()
        tp = cc.think("new idea", ThinkingMode.CREATIVE)
        assert "Creative insight" in tp.thought_content

    def test_think_critical_mode(self):
        cc = CognitiveCore()
        tp = cc.think("review", ThinkingMode.CRITICAL)
        assert "Critical review" in tp.thought_content

    def test_think_strategic_mode(self):
        cc = CognitiveCore()
        tp = cc.think("plan", ThinkingMode.STRATEGIC)
        assert "Strategic planning" in tp.thought_content

    def test_think_reflective_mode(self):
        cc = CognitiveCore()
        tp = cc.think("reflect", ThinkingMode.REFLECTIVE)
        assert "Reflection" in tp.thought_content

    def test_think_appends_to_history(self):
        cc = CognitiveCore()
        cc.think("first")
        cc.think("second")
        assert len(cc.thought_history) == 2

    def test_generate_idea(self):
        cc = CognitiveCore()
        idea = cc.generate_idea("flying car", "transport")
        assert isinstance(idea, CreativeIdea)
        assert idea.concept == "flying car"
        assert idea.category == "transport"

    def test_generate_idea_appends(self):
        cc = CognitiveCore()
        cc.generate_idea("idea1")
        cc.generate_idea("idea2")
        assert len(cc.ideas) == 2

    def test_reason(self):
        cc = CognitiveCore()
        rc = cc.reason("why is the sky blue?")
        assert isinstance(rc, ReasoningChain)
        assert "why is the sky blue?" in rc.question
        assert rc.reasoning_type == ReasoningType.DEDUCTIVE
        assert len(rc.reasoning_steps) == 2
        assert len(rc.evidence) == 2

    def test_reason_inductive(self):
        cc = CognitiveCore()
        rc = cc.reason("test", ReasoningType.INDUCTIVE)
        assert rc.reasoning_type == ReasoningType.INDUCTIVE
        assert "Conclusion" in rc.conclusion

    def test_reason_appends(self):
        cc = CognitiveCore()
        cc.reason("q1")
        cc.reason("q2")
        assert len(cc.reasoning_chains) == 2

    def test_get_recent_thoughts_empty(self):
        cc = CognitiveCore()
        assert cc.get_recent_thoughts() == []

    def test_get_recent_thoughts_limit(self):
        cc = CognitiveCore()
        for i in range(5):
            cc.think(f"thought {i}")
        recent = cc.get_recent_thoughts(limit=3)
        assert len(recent) == 3
        assert recent[-1].input_prompt == "thought 4"

    def test_get_statistics_empty(self):
        cc = CognitiveCore()
        stats = cc.get_statistics()
        assert stats["total_thoughts"] == 0
        assert stats["total_ideas"] == 0
        assert stats["total_reasoning_chains"] == 0

    def test_get_statistics_after_activity(self):
        cc = CognitiveCore()
        cc.think("test")
        cc.generate_idea("idea")
        cc.reason("question")
        stats = cc.get_statistics()
        assert stats["total_thoughts"] == 1
        assert stats["total_ideas"] == 1
        assert stats["total_reasoning_chains"] == 1
        assert ThinkingMode.ANALYTICAL.value in stats["modes_used"]

    def test_think_tracks_processing_time(self):
        cc = CognitiveCore()
        tp = cc.think("test")
        assert tp.processing_time >= 0.0

    def test_think_sets_confidence(self):
        cc = CognitiveCore()
        tp = cc.think("test")
        assert tp.confidence == 0.85
        assert tp.creativity_score == 0.7
        assert tp.logical_score == 0.8

    def test_generate_idea_sets_scores(self):
        cc = CognitiveCore()
        idea = cc.generate_idea("test")
        assert idea.novelty_score == 0.75
        assert idea.feasibility_score == 0.8
        assert idea.creativity_score == 0.85

    def test_reason_sets_confidence(self):
        cc = CognitiveCore()
        rc = cc.reason("test")
        assert rc.confidence == 0.8
