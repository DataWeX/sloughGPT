"""Meaningful tests for CognitiveCore — think, generate_idea, reason, statistics."""

from domains.cognitive.core import CognitiveCore, ThinkingMode, ReasoningType


class TestCognitiveCoreThink:
    def test_think_analytical(self):
        cc = CognitiveCore()
        t = cc.think("What is AI?", mode=ThinkingMode.ANALYTICAL)
        assert "Analysis" in t.thought_content
        assert t.mode == ThinkingMode.ANALYTICAL
        assert t.confidence == 0.85

    def test_think_creative(self):
        cc = CognitiveCore()
        t = cc.think("What is AI?", mode=ThinkingMode.CREATIVE)
        assert "Creative" in t.thought_content

    def test_think_critical(self):
        cc = CognitiveCore()
        t = cc.think("What is AI?", mode=ThinkingMode.CRITICAL)
        assert "Critical" in t.thought_content

    def test_think_strategic(self):
        cc = CognitiveCore()
        t = cc.think("What is AI?", mode=ThinkingMode.STRATEGIC)
        assert "Strategic" in t.thought_content

    def test_think_reflective(self):
        cc = CognitiveCore()
        t = cc.think("What is AI?", mode=ThinkingMode.REFLECTIVE)
        assert "Reflection" in t.thought_content

    def test_think_stores_history(self):
        cc = CognitiveCore()
        cc.think("Q1")
        cc.think("Q2")
        assert len(cc.thought_history) == 2
        assert cc.thought_history[0].input_prompt == "Q1"

    def test_think_timestamps_increase(self):
        cc = CognitiveCore()
        t1 = cc.think("Q1")
        t2 = cc.think("Q2")
        assert t2.timestamp >= t1.timestamp

    def test_think_ids_increment(self):
        cc = CognitiveCore()
        t1 = cc.think("Q1")
        t2 = cc.think("Q2")
        assert t1.id == "thought_0"
        assert t2.id == "thought_1"


class TestCognitiveCoreIdea:
    def test_generate_idea(self):
        cc = CognitiveCore()
        idea = cc.generate_idea("quantum computing", category="tech")
        assert idea.concept == "quantum computing"
        assert idea.category == "tech"
        assert idea.novelty_score == 0.75

    def test_idea_stored(self):
        cc = CognitiveCore()
        cc.generate_idea("A")
        cc.generate_idea("B")
        assert len(cc.ideas) == 2

    def test_idea_ids_increment(self):
        cc = CognitiveCore()
        i1 = cc.generate_idea("A")
        i2 = cc.generate_idea("B")
        assert i1.id == "idea_0"
        assert i2.id == "idea_1"


class TestCognitiveCoreReason:
    def test_reason_deductive(self):
        cc = CognitiveCore()
        chain = cc.reason("Why is sky blue?", ReasoningType.DEDUCTIVE)
        assert chain.question == "Why is sky blue?"
        assert chain.reasoning_type == ReasoningType.DEDUCTIVE
        assert len(chain.reasoning_steps) == 2
        assert len(chain.evidence) == 2

    def test_reason_stored(self):
        cc = CognitiveCore()
        cc.reason("Q1")
        cc.reason("Q2")
        assert len(cc.reasoning_chains) == 2


class TestCognitiveCoreStats:
    def test_statistics(self):
        cc = CognitiveCore()
        cc.think("Q1", ThinkingMode.ANALYTICAL)
        cc.think("Q2", ThinkingMode.CREATIVE)
        cc.generate_idea("I1")
        cc.reason("R1")
        stats = cc.get_statistics()
        assert stats["total_thoughts"] == 2
        assert stats["total_ideas"] == 1
        assert stats["total_reasoning_chains"] == 1
        assert "analytical" in stats["modes_used"]
        assert "creative" in stats["modes_used"]

    def test_statistics_empty(self):
        cc = CognitiveCore()
        stats = cc.get_statistics()
        assert stats["total_thoughts"] == 0
        assert stats["modes_used"] == []

    def test_get_recent_thoughts(self):
        cc = CognitiveCore()
        for i in range(15):
            cc.think(f"Q{i}")
        recent = cc.get_recent_thoughts(limit=5)
        assert len(recent) == 5
        assert recent[0].input_prompt == "Q10"
