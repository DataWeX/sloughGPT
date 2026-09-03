"""Tests for domains.cognitive.core — CognitiveCore, ThinkingMode, dataclasses."""

from __future__ import annotations

import time
import pytest

from domains.cognitive.core import (
    ThinkingMode,
    ReasoningType,
    ThoughtProcess,
    CreativeIdea,
    ReasoningChain,
    CognitiveCore,
)


# ── Enums ─────────────────────────────────────────────────────────────────────

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


# ── Dataclasses ───────────────────────────────────────────────────────────────

class TestDataclasses:
    def test_thought_process_fields(self):
        tp = ThoughtProcess(
            id="t1",
            mode=ThinkingMode.ANALYTICAL,
            reasoning_type=ReasoningType.DEDUCTIVE,
            input_prompt="test",
            thought_content="thought",
            confidence=0.9,
            creativity_score=0.7,
            logical_score=0.8,
            timestamp=1000.0,
            processing_time=0.1,
        )
        assert tp.id == "t1"
        assert tp.mode == ThinkingMode.ANALYTICAL
        assert tp.confidence == 0.9

    def test_creative_idea_fields(self):
        ci = CreativeIdea(
            id="i1",
            concept="test",
            description="desc",
            novelty_score=0.8,
            feasibility_score=0.7,
            creativity_score=0.9,
            category="general",
            tags=["tag1"],
            timestamp=1000.0,
        )
        assert ci.concept == "test"
        assert ci.tags == ["tag1"]

    def test_reasoning_chain_fields(self):
        rc = ReasoningChain(
            id="r1",
            question="why?",
            reasoning_steps=["step1"],
            conclusion="because",
            confidence=0.85,
            reasoning_type=ReasoningType.INDUCTIVE,
            evidence=["ev1"],
            timestamp=1000.0,
        )
        assert rc.question == "why?"
        assert rc.conclusion == "because"


# ── CognitiveCore ─────────────────────────────────────────────────────────────

class TestCognitiveCore:
    def test_think_analytical(self):
        core = CognitiveCore()
        tp = core.think("What is 2+2?", ThinkingMode.ANALYTICAL)
        assert tp.mode == ThinkingMode.ANALYTICAL
        assert "Analysis" in tp.thought_content
        assert tp.confidence > 0
        assert len(core.thought_history) == 1

    def test_think_creative(self):
        core = CognitiveCore()
        tp = core.think("Design a UI", ThinkingMode.CREATIVE)
        assert "Creative" in tp.thought_content

    def test_think_critical(self):
        core = CognitiveCore()
        tp = core.think("Review code", ThinkingMode.CRITICAL)
        assert "Critical" in tp.thought_content

    def test_think_strategic(self):
        core = CognitiveCore()
        tp = core.think("Plan launch", ThinkingMode.STRATEGIC)
        assert "Strategic" in tp.thought_content

    def test_think_reflective(self):
        core = CognitiveCore()
        tp = core.think("What did we learn?", ThinkingMode.REFLECTIVE)
        assert "Reflection" in tp.thought_content

    def test_generate_idea(self):
        core = CognitiveCore()
        idea = core.generate_idea("quantum computing", category="tech")
        assert idea.concept == "quantum computing"
        assert idea.category == "tech"
        assert idea.novelty_score > 0
        assert len(core.ideas) == 1

    def test_reason(self):
        core = CognitiveCore()
        chain = core.reason("Why is sky blue?", ReasoningType.DEDUCTIVE)
        assert chain.question == "Why is sky blue?"
        assert chain.reasoning_type == ReasoningType.DEDUCTIVE
        assert len(chain.reasoning_steps) > 0
        assert len(core.reasoning_chains) == 1

    def test_get_recent_thoughts(self):
        core = CognitiveCore()
        for i in range(5):
            core.think(f"thought {i}")
        recent = core.get_recent_thoughts(limit=3)
        assert len(recent) == 3
        assert recent[0].input_prompt == "thought 2"

    def test_get_statistics(self):
        core = CognitiveCore()
        core.think("test", ThinkingMode.ANALYTICAL)
        core.generate_idea("concept")
        core.reason("question")
        stats = core.get_statistics()
        assert stats["total_thoughts"] == 1
        assert stats["total_ideas"] == 1
        assert stats["total_reasoning_chains"] == 1
        assert "analytical" in stats["modes_used"]

    def test_thought_ids_increment(self):
        core = CognitiveCore()
        t1 = core.think("first")
        t2 = core.think("second")
        assert t1.id != t2.id

    def test_empty_statistics(self):
        core = CognitiveCore()
        stats = core.get_statistics()
        assert stats["total_thoughts"] == 0
        assert stats["modes_used"] == []
