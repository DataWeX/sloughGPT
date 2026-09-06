"""Tests for cognitive.core — CognitiveCore, thought processes, ideas, reasoning."""

from __future__ import annotations

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


# ── Enums ─────────────────────────────────────────────────────────────────


class TestEnums:

    def test_thinking_mode_values(self):
        assert ThinkingMode.ANALYTICAL.value == "analytical"
        assert ThinkingMode.CREATIVE.value == "creative"
        assert ThinkingMode.CRITICAL.value == "critical"
        assert ThinkingMode.STRATEGIC.value == "strategic"
        assert ThinkingMode.REFLECTIVE.value == "reflective"

    def test_reasoning_type_values(self):
        assert ReasoningType.DEDUCTIVE.value == "deductive"
        assert ReasoningType.INDUCTIVE.value == "inductive"
        assert ReasoningType.ABDUCTIVE.value == "abductive"
        assert ReasoningType.CAUSAL.value == "causal"
        assert ReasoningType.ANALOGICAL.value == "analogical"


# ── Dataclasses ───────────────────────────────────────────────────────────


class TestDataclasses:

    def test_thought_process_fields(self):
        tp = ThoughtProcess(
            id="t1", mode=ThinkingMode.ANALYTICAL,
            reasoning_type=ReasoningType.DEDUCTIVE,
            input_prompt="test", thought_content="thinking",
            confidence=0.9, creativity_score=0.5, logical_score=0.8,
            timestamp=1.0, processing_time=0.1,
        )
        assert tp.id == "t1"
        assert tp.confidence == 0.9

    def test_creative_idea_fields(self):
        idea = CreativeIdea(
            id="i1", concept="AI", description="An AI idea",
            novelty_score=0.8, feasibility_score=0.7,
            creativity_score=0.9, category="tech",
            tags=["ai", "ml"], timestamp=1.0,
        )
        assert idea.concept == "AI"
        assert len(idea.tags) == 2

    def test_reasoning_chain_fields(self):
        chain = ReasoningChain(
            id="c1", question="Why?",
            reasoning_steps=["step1", "step2"],
            conclusion="Because", confidence=0.85,
            reasoning_type=ReasoningType.DEDUCTIVE,
            evidence=["e1"], timestamp=1.0,
        )
        assert chain.conclusion == "Because"
        assert len(chain.reasoning_steps) == 2


# ── CognitiveCore ─────────────────────────────────────────────────────────


class TestCognitiveCore:

    def setup_method(self):
        self.core = CognitiveCore(db_path=":memory:")

    def test_init_defaults(self):
        core = CognitiveCore()
        assert core.thought_history == []
        assert core.ideas == []
        assert core.reasoning_chains == []

    def test_think_analytical(self):
        thought = self.core.think("What is 2+2?", ThinkingMode.ANALYTICAL)
        assert "Analysis" in thought.thought_content
        assert "2+2" in thought.thought_content
        assert len(self.core.thought_history) == 1

    def test_think_creative(self):
        thought = self.core.think("New idea", ThinkingMode.CREATIVE)
        assert "Creative insight" in thought.thought_content

    def test_think_critical(self):
        thought = self.core.think("Review this", ThinkingMode.CRITICAL)
        assert "Critical review" in thought.thought_content

    def test_think_strategic(self):
        thought = self.core.think("Plan ahead", ThinkingMode.STRATEGIC)
        assert "Strategic planning" in thought.thought_content

    def test_think_reflective(self):
        thought = self.core.think("Look back", ThinkingMode.REFLECTIVE)
        assert "Reflection" in thought.thought_content

    def test_think_default_mode(self):
        thought = self.core.think("Default prompt")
        assert thought.mode == ThinkingMode.ANALYTICAL

    def test_think_timestamp_reasonable(self):
        before = time.time()
        thought = self.core.think("test")
        after = time.time()
        assert before <= thought.timestamp <= after

    def test_generate_idea(self):
        idea = self.core.generate_idea("quantum computing", "science")
        assert idea.concept == "quantum computing"
        assert idea.category == "science"
        assert "science" in idea.tags
        assert len(self.core.ideas) == 1

    def test_generate_idea_default_category(self):
        idea = self.core.generate_idea("test concept")
        assert idea.category == "general"

    def test_reason(self):
        chain = self.core.reason("What is AI?")
        assert chain.question == "What is AI?"
        assert chain.reasoning_type == ReasoningType.DEDUCTIVE
        assert "Step 1" in chain.reasoning_steps[0]
        assert len(self.core.reasoning_chains) == 1

    def test_reason_custom_type(self):
        chain = self.core.reason("Why rain?", ReasoningType.CAUSAL)
        assert chain.reasoning_type == ReasoningType.CAUSAL

    def test_get_recent_thoughts(self):
        for i in range(5):
            self.core.think(f"prompt {i}")
        recent = self.core.get_recent_thoughts(3)
        assert len(recent) == 3
        assert recent[0].input_prompt == "prompt 2"

    def test_get_recent_thoughts_limit_exceeds(self):
        for i in range(3):
            self.core.think(f"prompt {i}")
        recent = self.core.get_recent_thoughts(10)
        assert len(recent) == 3

    def test_get_statistics(self):
        self.core.think("a", ThinkingMode.ANALYTICAL)
        self.core.think("b", ThinkingMode.CREATIVE)
        self.core.generate_idea("x")
        self.core.reason("y")
        stats = self.core.get_statistics()
        assert stats["total_thoughts"] == 2
        assert stats["total_ideas"] == 1
        assert stats["total_reasoning_chains"] == 1
        assert "analytical" in stats["modes_used"]
        assert "creative" in stats["modes_used"]

    def test_get_statistics_empty(self):
        stats = self.core.get_statistics()
        assert stats["total_thoughts"] == 0
        assert stats["modes_used"] == []
