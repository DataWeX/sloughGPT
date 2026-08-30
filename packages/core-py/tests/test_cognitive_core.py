"""Tests for domains.cognitive.core — ThinkingMode, ReasoningType, ThoughtProcess, CreativeIdea, ReasoningChain, CognitiveCore."""

import time
import pytest
from domains.cognitive.core import (
    ThinkingMode, ReasoningType, ThoughtProcess, CreativeIdea, ReasoningChain,
    CognitiveCore,
)


class TestThinkingMode:
    def test_all_members(self):
        assert len(ThinkingMode) == 5

    def test_values(self):
        assert ThinkingMode.ANALYTICAL.value == "analytical"
        assert ThinkingMode.CREATIVE.value == "creative"
        assert ThinkingMode.CRITICAL.value == "critical"
        assert ThinkingMode.STRATEGIC.value == "strategic"
        assert ThinkingMode.REFLECTIVE.value == "reflective"

    def test_member_names(self):
        names = [m.name for m in ThinkingMode]
        assert "ANALYTICAL" in names
        assert "CREATIVE" in names

    def test_string_repr(self):
        assert "ANALYTICAL" in repr(ThinkingMode.ANALYTICAL)

    def test_from_value(self):
        assert ThinkingMode("creative") == ThinkingMode.CREATIVE

    def test_invalid_value(self):
        with pytest.raises(ValueError):
            ThinkingMode("nonexistent")

    def test_iteration(self):
        modes = list(ThinkingMode)
        assert len(modes) == 5

    def test_comparison(self):
        assert ThinkingMode.ANALYTICAL == ThinkingMode.ANALYTICAL
        assert ThinkingMode.ANALYTICAL != ThinkingMode.CREATIVE

    def test_is_enum(self):
        from enum import Enum
        assert issubclass(ThinkingMode, Enum)


class TestReasoningType:
    def test_all_members(self):
        assert len(ReasoningType) == 5

    def test_values(self):
        assert ReasoningType.DEDUCTIVE.value == "deductive"
        assert ReasoningType.INDUCTIVE.value == "inductive"
        assert ReasoningType.ANALOGICAL.value == "analogical"
        assert ReasoningType.ABDUCTIVE.value == "abductive"
        assert ReasoningType.CAUSAL.value == "causal"

    def test_member_names(self):
        names = [m.name for m in ReasoningType]
        assert "DEDUCTIVE" in names
        assert "INDUCTIVE" in names
        assert "ABDUCTIVE" in names

    def test_from_value(self):
        assert ReasoningType("causal") == ReasoningType.CAUSAL

    def test_invalid_value(self):
        with pytest.raises(ValueError):
            ReasoningType("magic")

    def test_iteration(self):
        types = list(ReasoningType)
        assert len(types) == 5

    def test_distinct_values(self):
        values = {rt.value for rt in ReasoningType}
        assert len(values) == 5

    def test_is_enum(self):
        from enum import Enum
        assert issubclass(ReasoningType, Enum)


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
        assert tp.creativity_score == 0.5
        assert tp.logical_score == 0.8

    def test_all_fields_set(self):
        tp = ThoughtProcess(
            id="t2", mode=ThinkingMode.CREATIVE,
            reasoning_type=ReasoningType.INDUCTIVE,
            input_prompt="q", thought_content="t",
            confidence=0.7, creativity_score=0.9, logical_score=0.6,
            timestamp=2.0, processing_time=0.1,
        )
        assert tp.mode == ThinkingMode.CREATIVE
        assert tp.reasoning_type == ReasoningType.INDUCTIVE
        assert tp.input_prompt == "q"
        assert tp.thought_content == "t"

    def test_confidence_range(self):
        tp = ThoughtProcess(
            id="t", mode=ThinkingMode.CRITICAL,
            reasoning_type=ReasoningType.ABDUCTIVE,
            input_prompt="", thought_content="",
            confidence=0.0, creativity_score=0.0, logical_score=0.0,
            timestamp=0.0, processing_time=0.0,
        )
        assert tp.confidence == 0.0

    def test_high_confidence(self):
        tp = ThoughtProcess(
            id="t", mode=ThinkingMode.STRATEGIC,
            reasoning_type=ReasoningType.CAUSAL,
            input_prompt="", thought_content="",
            confidence=1.0, creativity_score=1.0, logical_score=1.0,
            timestamp=0.0, processing_time=0.0,
        )
        assert tp.confidence == 1.0

    def test_equality(self):
        kwargs = dict(
            id="t", mode=ThinkingMode.REFLECTIVE,
            reasoning_type=ReasoningType.ANALOGICAL,
            input_prompt="p", thought_content="c",
            confidence=0.5, creativity_score=0.5, logical_score=0.5,
            timestamp=0.0, processing_time=0.0,
        )
        a = ThoughtProcess(**kwargs)
        b = ThoughtProcess(**kwargs)
        assert a == b

    def test_repr(self):
        tp = ThoughtProcess(
            id="t3", mode=ThinkingMode.ANALYTICAL,
            reasoning_type=ReasoningType.DEDUCTIVE,
            input_prompt="test", thought_content="thought",
            confidence=0.9, creativity_score=0.5, logical_score=0.8,
            timestamp=1.0, processing_time=0.5,
        )
        r = repr(tp)
        assert "t3" in r

    def test_processing_time_zero(self):
        tp = ThoughtProcess(
            id="t", mode=ThinkingMode.ANALYTICAL,
            reasoning_type=ReasoningType.DEDUCTIVE,
            input_prompt="", thought_content="",
            confidence=0.0, creativity_score=0.0, logical_score=0.0,
            timestamp=0.0, processing_time=0.0,
        )
        assert tp.processing_time == 0.0


class TestCreativeIdea:
    def test_fields(self):
        ci = CreativeIdea(
            id="i1", concept="test", description="desc",
            novelty_score=0.8, feasibility_score=0.7, creativity_score=0.9,
            category="tech", tags=["ai"], timestamp=1.0,
        )
        assert ci.id == "i1"
        assert ci.novelty_score == 0.8

    def test_all_fields(self):
        ci = CreativeIdea(
            id="i2", concept="idea", description="new idea",
            novelty_score=0.5, feasibility_score=0.6, creativity_score=0.7,
            category="science", tags=["ml", "dl"], timestamp=2.0,
        )
        assert ci.concept == "idea"
        assert ci.description == "new idea"
        assert ci.category == "science"
        assert len(ci.tags) == 2

    def test_empty_tags(self):
        ci = CreativeIdea(
            id="i3", concept="c", description="d",
            novelty_score=0.0, feasibility_score=0.0, creativity_score=0.0,
            category="", tags=[], timestamp=0.0,
        )
        assert ci.tags == []

    def test_equality(self):
        kwargs = dict(
            id="i", concept="c", description="d",
            novelty_score=0.5, feasibility_score=0.5, creativity_score=0.5,
            category="cat", tags=[], timestamp=0.0,
        )
        assert CreativeIdea(**kwargs) == CreativeIdea(**kwargs)

    def test_scores_range(self):
        ci = CreativeIdea(
            id="i", concept="c", description="d",
            novelty_score=1.0, feasibility_score=1.0, creativity_score=1.0,
            category="", tags=[], timestamp=0.0,
        )
        assert ci.novelty_score == 1.0

    def test_repr(self):
        ci = CreativeIdea(
            id="i4", concept="concept", description="desc",
            novelty_score=0.8, feasibility_score=0.7, creativity_score=0.9,
            category="tech", tags=["ai"], timestamp=1.0,
        )
        r = repr(ci)
        assert "i4" in r

    def test_many_tags(self):
        tags = [f"tag{i}" for i in range(20)]
        ci = CreativeIdea(
            id="i", concept="c", description="d",
            novelty_score=0.5, feasibility_score=0.5, creativity_score=0.5,
            category="", tags=tags, timestamp=0.0,
        )
        assert len(ci.tags) == 20


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

    def test_all_fields(self):
        rc = ReasoningChain(
            id="r2", question="what?", reasoning_steps=["s1", "s2", "s3"],
            conclusion="answer", confidence=0.9,
            reasoning_type=ReasoningType.DEDUCTIVE,
            evidence=["e1", "e2"], timestamp=2.0,
        )
        assert rc.question == "what?"
        assert rc.conclusion == "answer"
        assert rc.confidence == 0.9
        assert len(rc.reasoning_steps) == 3
        assert len(rc.evidence) == 2

    def test_empty_steps(self):
        rc = ReasoningChain(
            id="r3", question="q", reasoning_steps=[],
            conclusion="c", confidence=0.0,
            reasoning_type=ReasoningType.INDUCTIVE,
            evidence=[], timestamp=0.0,
        )
        assert rc.reasoning_steps == []

    def test_equality(self):
        kwargs = dict(
            id="r", question="q", reasoning_steps=["s"],
            conclusion="c", confidence=0.5,
            reasoning_type=ReasoningType.ABDUCTIVE,
            evidence=["e"], timestamp=0.0,
        )
        assert ReasoningChain(**kwargs) == ReasoningChain(**kwargs)

    def test_repr(self):
        rc = ReasoningChain(
            id="r4", question="q", reasoning_steps=[],
            conclusion="c", confidence=0.5,
            reasoning_type=ReasoningType.CAUSAL,
            evidence=[], timestamp=0.0,
        )
        r = repr(rc)
        assert "r4" in r

    def test_many_steps(self):
        steps = [f"Step {i}" for i in range(50)]
        rc = ReasoningChain(
            id="r", question="q", reasoning_steps=steps,
            conclusion="c", confidence=0.5,
            reasoning_type=ReasoningType.DEDUCTIVE,
            evidence=[], timestamp=0.0,
        )
        assert len(rc.reasoning_steps) == 50


class TestCognitiveCore:
    def test_init(self):
        core = CognitiveCore()
        assert core.thought_history == []
        assert core.ideas == []
        assert core.reasoning_chains == []

    def test_think_analytical(self):
        core = CognitiveCore()
        tp = core.think("test prompt", ThinkingMode.ANALYTICAL)
        assert tp.mode == ThinkingMode.ANALYTICAL
        assert "Analysis" in tp.thought_content
        assert len(core.thought_history) == 1

    def test_think_creative(self):
        core = CognitiveCore()
        tp = core.think("test", ThinkingMode.CREATIVE)
        assert "Creative" in tp.thought_content

    def test_think_critical(self):
        core = CognitiveCore()
        tp = core.think("test", ThinkingMode.CRITICAL)
        assert "Critical" in tp.thought_content

    def test_think_strategic(self):
        core = CognitiveCore()
        tp = core.think("test", ThinkingMode.STRATEGIC)
        assert "Strategic" in tp.thought_content

    def test_think_reflective(self):
        core = CognitiveCore()
        tp = core.think("test", ThinkingMode.REFLECTIVE)
        assert "Reflection" in tp.thought_content

    def test_think_default_mode(self):
        core = CognitiveCore()
        tp = core.think("test")
        assert tp.mode == ThinkingMode.ANALYTICAL

    def test_think_assigns_id(self):
        core = CognitiveCore()
        t1 = core.think("a")
        t2 = core.think("b")
        assert t1.id != t2.id
        assert t1.id == "thought_0"
        assert t2.id == "thought_1"

    def test_think_confidence(self):
        core = CognitiveCore()
        tp = core.think("test")
        assert tp.confidence == 0.85

    def test_think_timestamp(self):
        core = CognitiveCore()
        before = time.time()
        tp = core.think("test")
        after = time.time()
        assert before <= tp.timestamp <= after

    def test_generate_idea(self):
        core = CognitiveCore()
        idea = core.generate_idea("new approach", "tech")
        assert idea.concept == "new approach"
        assert idea.category == "tech"
        assert len(core.ideas) == 1

    def test_generate_idea_default_category(self):
        core = CognitiveCore()
        idea = core.generate_idea("concept")
        assert idea.category == "general"

    def test_generate_idea_scores(self):
        core = CognitiveCore()
        idea = core.generate_idea("test")
        assert idea.novelty_score == 0.75
        assert idea.feasibility_score == 0.8
        assert idea.creativity_score == 0.85

    def test_generate_idea_tags(self):
        core = CognitiveCore()
        idea = core.generate_idea("test", "science")
        assert "science" in idea.tags

    def test_generate_idea_id(self):
        core = CognitiveCore()
        i1 = core.generate_idea("a")
        i2 = core.generate_idea("b")
        assert i1.id != i2.id

    def test_reason_deductive(self):
        core = CognitiveCore()
        rc = core.reason("why is sky blue?", ReasoningType.DEDUCTIVE)
        assert rc.reasoning_type == ReasoningType.DEDUCTIVE
        assert rc.question == "why is sky blue?"
        assert len(core.reasoning_chains) == 1

    def test_reason_default(self):
        core = CognitiveCore()
        rc = core.reason("test question")
        assert rc.reasoning_type == ReasoningType.DEDUCTIVE

    def test_reason_steps(self):
        core = CognitiveCore()
        rc = core.reason("q")
        assert len(rc.reasoning_steps) == 2

    def test_reason_evidence(self):
        core = CognitiveCore()
        rc = core.reason("q")
        assert len(rc.evidence) == 2

    def test_reason_confidence(self):
        core = CognitiveCore()
        rc = core.reason("q")
        assert rc.confidence == 0.8

    def test_reason_id(self):
        core = CognitiveCore()
        r1 = core.reason("q1")
        r2 = core.reason("q2")
        assert r1.id != r2.id

    def test_get_recent_thoughts(self):
        core = CognitiveCore()
        for i in range(5):
            core.think(f"prompt {i}")
        recent = core.get_recent_thoughts(3)
        assert len(recent) == 3

    def test_get_recent_thoughts_limit(self):
        core = CognitiveCore()
        for i in range(10):
            core.think(f"p{i}")
        recent = core.get_recent_thoughts(10)
        assert len(recent) == 10

    def test_get_recent_thoughts_empty(self):
        core = CognitiveCore()
        recent = core.get_recent_thoughts()
        assert recent == []

    def test_get_recent_thoughts_more_than_available(self):
        core = CognitiveCore()
        core.think("only one")
        recent = core.get_recent_thoughts(100)
        assert len(recent) == 1

    def test_get_statistics(self):
        core = CognitiveCore()
        stats = core.get_statistics()
        assert stats["total_thoughts"] == 0
        assert stats["total_ideas"] == 0
        assert stats["total_reasoning_chains"] == 0

    def test_get_statistics_with_data(self):
        core = CognitiveCore()
        core.think("a", ThinkingMode.ANALYTICAL)
        core.think("b", ThinkingMode.CREATIVE)
        core.generate_idea("concept")
        core.reason("question")
        stats = core.get_statistics()
        assert stats["total_thoughts"] == 2
        assert stats["total_ideas"] == 1
        assert stats["total_reasoning_chains"] == 1
        assert "analytical" in stats["modes_used"]
        assert "creative" in stats["modes_used"]

    def test_statistics_modes_unique(self):
        core = CognitiveCore()
        core.think("a")
        core.think("b")
        stats = core.get_statistics()
        assert len(stats["modes_used"]) == 1

    def test_multiple_modes_stats(self):
        core = CognitiveCore()
        core.think("a", ThinkingMode.ANALYTICAL)
        core.think("b", ThinkingMode.CRITICAL)
        core.think("c", ThinkingMode.STRATEGIC)
        stats = core.get_statistics()
        assert len(stats["modes_used"]) == 3

    def test_history_grows(self):
        core = CognitiveCore()
        assert len(core.thought_history) == 0
        core.think("a")
        assert len(core.thought_history) == 1
        core.think("b")
        assert len(core.thought_history) == 2

    def test_ideas_grow(self):
        core = CognitiveCore()
        core.generate_idea("a")
        core.generate_idea("b")
        assert len(core.ideas) == 2

    def test_chains_grow(self):
        core = CognitiveCore()
        core.reason("q1")
        core.reason("q2")
        assert len(core.reasoning_chains) == 2

    def test_custom_db_path(self):
        core = CognitiveCore(db_path="/tmp/test.db")
        assert core.db_path == "/tmp/test.db"
