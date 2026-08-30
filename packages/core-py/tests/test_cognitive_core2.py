"""Tests for domains.cognitive.core — CognitiveCore, enums, and dataclasses."""

import time
import pytest
from dataclasses import fields as dc_fields

from domains.cognitive.core import (
    ThinkingMode,
    ReasoningType,
    ThoughtProcess,
    CreativeIdea,
    ReasoningChain,
    CognitiveCore,
)


# ---------------------------------------------------------------------------
# ThinkingMode enum
# ---------------------------------------------------------------------------

class TestThinkingMode:
    def test_member_count(self):
        assert len(ThinkingMode) == 5

    def test_analytical_value(self):
        assert ThinkingMode.ANALYTICAL.value == "analytical"

    def test_creative_value(self):
        assert ThinkingMode.CREATIVE.value == "creative"

    def test_critical_value(self):
        assert ThinkingMode.CRITICAL.value == "critical"

    def test_strategic_value(self):
        assert ThinkingMode.STRATEGIC.value == "strategic"

    def test_reflective_value(self):
        assert ThinkingMode.REFLECTIVE.value == "reflective"

    def test_all_values_are_strings(self):
        for member in ThinkingMode:
            assert isinstance(member.value, str)

    def test_no_duplicates(self):
        values = [m.value for m in ThinkingMode]
        assert len(values) == len(set(values))


# ---------------------------------------------------------------------------
# ReasoningType enum
# ---------------------------------------------------------------------------

class TestReasoningType:
    def test_member_count(self):
        assert len(ReasoningType) == 5

    def test_deductive_value(self):
        assert ReasoningType.DEDUCTIVE.value == "deductive"

    def test_inductive_value(self):
        assert ReasoningType.INDUCTIVE.value == "inductive"

    def test_abductive_value(self):
        assert ReasoningType.ABDUCTIVE.value == "abductive"

    def test_causal_value(self):
        assert ReasoningType.CAUSAL.value == "causal"

    def test_analogical_value(self):
        assert ReasoningType.ANALOGICAL.value == "analogical"

    def test_no_duplicates(self):
        values = [m.value for m in ReasoningType]
        assert len(values) == len(set(values))


# ---------------------------------------------------------------------------
# ThoughtProcess dataclass
# ---------------------------------------------------------------------------

class TestThoughtProcess:
    def _make(self, **overrides):
        defaults = dict(
            id="t0",
            mode=ThinkingMode.ANALYTICAL,
            reasoning_type=ReasoningType.DEDUCTIVE,
            input_prompt="prompt",
            thought_content="content",
            confidence=0.9,
            creativity_score=0.5,
            logical_score=0.8,
            timestamp=1.0,
            processing_time=0.1,
        )
        defaults.update(overrides)
        return ThoughtProcess(**defaults)

    def test_field_count(self):
        assert len(dc_fields(ThoughtProcess)) == 10

    def test_all_fields_accessible(self):
        tp = self._make()
        assert tp.id == "t0"
        assert tp.mode == ThinkingMode.ANALYTICAL
        assert tp.reasoning_type == ReasoningType.DEDUCTIVE
        assert tp.input_prompt == "prompt"
        assert tp.thought_content == "content"
        assert tp.confidence == 0.9
        assert tp.creativity_score == 0.5
        assert tp.logical_score == 0.8
        assert tp.timestamp == 1.0
        assert tp.processing_time == 0.1

    def test_equality(self):
        a = self._make()
        b = self._make()
        assert a == b

    def test_inequality(self):
        a = self._make(id="a")
        b = self._make(id="b")
        assert a != b

    def test_creative_mode(self):
        tp = self._make(mode=ThinkingMode.CREATIVE)
        assert tp.mode == ThinkingMode.CREATIVE


# ---------------------------------------------------------------------------
# CreativeIdea dataclass
# ---------------------------------------------------------------------------

class TestCreativeIdea:
    def _make(self, **overrides):
        defaults = dict(
            id="i0",
            concept="test",
            description="desc",
            novelty_score=0.7,
            feasibility_score=0.8,
            creativity_score=0.9,
            category="tech",
            tags=["ai", "ml"],
            timestamp=2.0,
        )
        defaults.update(overrides)
        return CreativeIdea(**defaults)

    def test_field_count(self):
        assert len(dc_fields(CreativeIdea)) == 9

    def test_all_fields_accessible(self):
        ci = self._make()
        assert ci.id == "i0"
        assert ci.concept == "test"
        assert ci.description == "desc"
        assert ci.novelty_score == 0.7
        assert ci.feasibility_score == 0.8
        assert ci.creativity_score == 0.9
        assert ci.category == "tech"
        assert ci.tags == ["ai", "ml"]
        assert ci.timestamp == 2.0

    def test_empty_tags(self):
        ci = self._make(tags=[])
        assert ci.tags == []

    def test_equality(self):
        assert self._make() == self._make()


# ---------------------------------------------------------------------------
# ReasoningChain dataclass
# ---------------------------------------------------------------------------

class TestReasoningChain:
    def _make(self, **overrides):
        defaults = dict(
            id="r0",
            question="why?",
            reasoning_steps=["s1", "s2"],
            conclusion="because",
            confidence=0.85,
            reasoning_type=ReasoningType.ABDUCTIVE,
            evidence=["e1", "e2", "e3"],
            timestamp=3.0,
        )
        defaults.update(overrides)
        return ReasoningChain(**defaults)

    def test_field_count(self):
        assert len(dc_fields(ReasoningChain)) == 8

    def test_all_fields_accessible(self):
        rc = self._make()
        assert rc.id == "r0"
        assert rc.question == "why?"
        assert rc.reasoning_steps == ["s1", "s2"]
        assert rc.conclusion == "because"
        assert rc.confidence == 0.85
        assert rc.reasoning_type == ReasoningType.ABDUCTIVE
        assert rc.evidence == ["e1", "e2", "e3"]
        assert rc.timestamp == 3.0

    def test_single_step(self):
        rc = self._make(reasoning_steps=["only"])
        assert len(rc.reasoning_steps) == 1

    def test_equality(self):
        assert self._make() == self._make()


# ---------------------------------------------------------------------------
# CognitiveCore
# ---------------------------------------------------------------------------

class TestCognitiveCore:
    def setup_method(self):
        self.core = CognitiveCore(db_path=":memory:")

    def test_initial_state(self):
        assert self.core.thought_history == []
        assert self.core.ideas == []
        assert self.core.reasoning_chains == []

    def test_db_path(self):
        assert self.core.db_path == ":memory:"

    def test_default_db_path(self):
        c = CognitiveCore()
        assert c.db_path == "slo_cognitive_core.db"


# ---------------------------------------------------------------------------
# CognitiveCore.think
# ---------------------------------------------------------------------------

class TestCognitiveCoreThink:
    def setup_method(self):
        self.core = CognitiveCore()

    def test_returns_thought_process(self):
        result = self.core.think("test prompt")
        assert isinstance(result, ThoughtProcess)

    def test_analytical_content(self):
        result = self.core.think("X", mode=ThinkingMode.ANALYTICAL)
        assert "Analysis:" in result.thought_content
        assert "breaking down" in result.thought_content

    def test_creative_content(self):
        result = self.core.think("X", mode=ThinkingMode.CREATIVE)
        assert "Creative insight:" in result.thought_content
        assert "innovative" in result.thought_content

    def test_critical_content(self):
        result = self.core.think("X", mode=ThinkingMode.CRITICAL)
        assert "Critical review:" in result.thought_content
        assert "strengths" in result.thought_content

    def test_strategic_content(self):
        result = self.core.think("X", mode=ThinkingMode.STRATEGIC)
        assert "Strategic planning:" in result.thought_content
        assert "long-term" in result.thought_content

    def test_reflective_content(self):
        result = self.core.think("X", mode=ThinkingMode.REFLECTIVE)
        assert "Reflection:" in result.thought_content
        assert "learning" in result.thought_content

    def test_appends_to_history(self):
        self.core.think("a")
        self.core.think("b")
        assert len(self.core.thought_history) == 2

    def test_id_increments(self):
        a = self.core.think("a")
        b = self.core.think("b")
        assert a.id == "thought_0"
        assert b.id == "thought_1"

    def test_default_mode_is_analytical(self):
        result = self.core.think("prompt")
        assert result.mode == ThinkingMode.ANALYTICAL

    def test_default_reasoning_type_deductive(self):
        result = self.core.think("prompt")
        assert result.reasoning_type == ReasoningType.DEDUCTIVE

    def test_confidence_is_fixed(self):
        result = self.core.think("prompt")
        assert result.confidence == 0.85

    def test_timestamp_is_positive(self):
        before = time.time()
        result = self.core.think("prompt")
        after = time.time()
        assert before <= result.timestamp <= after

    def test_processing_time_non_negative(self):
        result = self.core.think("prompt")
        assert result.processing_time >= 0


# ---------------------------------------------------------------------------
# CognitiveCore.generate_idea
# ---------------------------------------------------------------------------

class TestCognitiveCoreGenerateIdea:
    def setup_method(self):
        self.core = CognitiveCore()

    def test_returns_creative_idea(self):
        result = self.core.generate_idea("test concept")
        assert isinstance(result, CreativeIdea)

    def test_default_category(self):
        result = self.core.generate_idea("concept")
        assert result.category == "general"
        assert result.tags == ["general"]

    def test_custom_category(self):
        result = self.core.generate_idea("concept", category="tech")
        assert result.category == "tech"
        assert result.tags == ["tech"]

    def test_appends_to_ideas(self):
        self.core.generate_idea("a")
        self.core.generate_idea("b")
        assert len(self.core.ideas) == 2

    def test_id_increments(self):
        a = self.core.generate_idea("a")
        b = self.core.generate_idea("b")
        assert a.id == "idea_0"
        assert b.id == "idea_1"

    def test_description_format(self):
        result = self.core.generate_idea("my concept")
        assert "my concept" in result.description

    def test_novelty_score(self):
        result = self.core.generate_idea("x")
        assert result.novelty_score == 0.75

    def test_feasibility_score(self):
        result = self.core.generate_idea("x")
        assert result.feasibility_score == 0.8

    def test_creativity_score(self):
        result = self.core.generate_idea("x")
        assert result.creativity_score == 0.85


# ---------------------------------------------------------------------------
# CognitiveCore.reason
# ---------------------------------------------------------------------------

class TestCognitiveCoreReason:
    def setup_method(self):
        self.core = CognitiveCore()

    def test_returns_reasoning_chain(self):
        result = self.core.reason("why?")
        assert isinstance(result, ReasoningChain)

    def test_default_deductive(self):
        result = self.core.reason("q")
        assert result.reasoning_type == ReasoningType.DEDUCTIVE

    def test_custom_reasoning_type(self):
        result = self.core.reason("q", reasoning_type=ReasoningType.INDUCTIVE)
        assert result.reasoning_type == ReasoningType.INDUCTIVE

    def test_question_preserved(self):
        result = self.core.reason("my question?")
        assert result.question == "my question?"

    def test_conclusion_format(self):
        result = self.core.reason("X?")
        assert "X?" in result.conclusion

    def test_has_evidence(self):
        result = self.core.reason("q")
        assert len(result.evidence) > 0

    def test_has_reasoning_steps(self):
        result = self.core.reason("q")
        assert len(result.reasoning_steps) == 2

    def test_appends_to_chains(self):
        self.core.reason("a")
        self.core.reason("b")
        assert len(self.core.reasoning_chains) == 2

    def test_id_increments(self):
        a = self.core.reason("a")
        b = self.core.reason("b")
        assert a.id == "chain_0"
        assert b.id == "chain_1"

    def test_confidence_is_fixed(self):
        result = self.core.reason("q")
        assert result.confidence == 0.8

    def test_analogical_type(self):
        result = self.core.reason("q", reasoning_type=ReasoningType.ANALOGICAL)
        assert result.reasoning_type == ReasoningType.ANALOGICAL

    def test_abductive_type(self):
        result = self.core.reason("q", reasoning_type=ReasoningType.ABDUCTIVE)
        assert result.reasoning_type == ReasoningType.ABDUCTIVE

    def test_causal_type(self):
        result = self.core.reason("q", reasoning_type=ReasoningType.CAUSAL)
        assert result.reasoning_type == ReasoningType.CAUSAL


# ---------------------------------------------------------------------------
# CognitiveCore.get_recent_thoughts
# ---------------------------------------------------------------------------

class TestCognitiveCoreGetRecentThoughts:
    def setup_method(self):
        self.core = CognitiveCore()

    def test_empty_when_no_history(self):
        assert self.core.get_recent_thoughts() == []

    def test_returns_all_when_under_limit(self):
        self.core.think("a")
        self.core.think("b")
        result = self.core.get_recent_thoughts(limit=10)
        assert len(result) == 2

    def test_respects_limit(self):
        for i in range(5):
            self.core.think(f"p{i}")
        result = self.core.get_recent_thoughts(limit=3)
        assert len(result) == 3

    def test_returns_last_n(self):
        for i in range(5):
            self.core.think(f"p{i}")
        result = self.core.get_recent_thoughts(limit=2)
        assert result[0].input_prompt == "p3"
        assert result[1].input_prompt == "p4"

    def test_limit_zero_returns_all(self):
        self.core.think("a")
        result = self.core.get_recent_thoughts(limit=0)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# CognitiveCore.get_statistics
# ---------------------------------------------------------------------------

class TestCognitiveCoreGetStatistics:
    def setup_method(self):
        self.core = CognitiveCore()

    def test_empty_stats(self):
        stats = self.core.get_statistics()
        assert stats["total_thoughts"] == 0
        assert stats["total_ideas"] == 0
        assert stats["total_reasoning_chains"] == 0
        assert stats["modes_used"] == []

    def test_thoughts_count(self):
        self.core.think("a")
        self.core.think("b")
        stats = self.core.get_statistics()
        assert stats["total_thoughts"] == 2

    def test_ideas_count(self):
        self.core.generate_idea("a")
        stats = self.core.get_statistics()
        assert stats["total_ideas"] == 1

    def test_reasoning_chains_count(self):
        self.core.reason("a")
        self.core.reason("b")
        self.core.reason("c")
        stats = self.core.get_statistics()
        assert stats["total_reasoning_chains"] == 3

    def test_modes_used_populated(self):
        self.core.think("a", mode=ThinkingMode.ANALYTICAL)
        self.core.think("b", mode=ThinkingMode.CREATIVE)
        stats = self.core.get_statistics()
        assert "analytical" in stats["modes_used"]
        assert "creative" in stats["modes_used"]

    def test_modes_used_deduped(self):
        self.core.think("a", mode=ThinkingMode.ANALYTICAL)
        self.core.think("b", mode=ThinkingMode.ANALYTICAL)
        stats = self.core.get_statistics()
        assert stats["modes_used"].count("analytical") == 1

    def test_full_lifecycle(self):
        self.core.think("a", ThinkingMode.CRITICAL)
        self.core.think("b", ThinkingMode.STRATEGIC)
        self.core.generate_idea("x")
        self.core.generate_idea("y")
        self.core.generate_idea("z")
        self.core.reason("q1")
        stats = self.core.get_statistics()
        assert stats["total_thoughts"] == 2
        assert stats["total_ideas"] == 3
        assert stats["total_reasoning_chains"] == 1
        assert set(stats["modes_used"]) == {"critical", "strategic"}
