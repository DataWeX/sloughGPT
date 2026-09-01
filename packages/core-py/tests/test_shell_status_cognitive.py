from domains.shell.cmds.status import _fmt_uptime
from domains.cognitive.core import (
    CognitiveCore, ThinkingMode, ReasoningType,
    ThoughtProcess, CreativeIdea, ReasoningChain,
)


class TestFmtUptime:
    def test_seconds(self):
        assert _fmt_uptime(30) == "30s"

    def test_zero(self):
        assert _fmt_uptime(0) == "0s"

    def test_minutes(self):
        assert _fmt_uptime(125) == "2m 5s"

    def test_exact_minute(self):
        assert _fmt_uptime(60) == "1m 0s"

    def test_59_seconds(self):
        assert _fmt_uptime(59) == "59s"

    def test_hours(self):
        assert _fmt_uptime(3661) == "1h 01m"

    def test_hours_minutes(self):
        assert _fmt_uptime(5400) == "1h 30m"

    def test_large_uptime(self):
        assert _fmt_uptime(86400) == "24h 00m"


class TestThinkingMode:
    def test_values(self):
        assert ThinkingMode.ANALYTICAL.value == "analytical"
        assert ThinkingMode.CREATIVE.value == "creative"
        assert ThinkingMode.CRITICAL.value == "critical"
        assert ThinkingMode.STRATEGIC.value == "strategic"
        assert ThinkingMode.REFLECTIVE.value == "reflective"

    def test_count(self):
        assert len(ThinkingMode) == 5


class TestReasoningType:
    def test_values(self):
        assert ReasoningType.DEDUCTIVE.value == "deductive"
        assert ReasoningType.INDUCTIVE.value == "inductive"
        assert ReasoningType.ABDUCTIVE.value == "abductive"
        assert ReasoningType.CAUSAL.value == "causal"
        assert ReasoningType.ANALOGICAL.value == "analogical"

    def test_count(self):
        assert len(ReasoningType) == 5


class TestThoughtProcess:
    def test_create(self):
        tp = ThoughtProcess(
            id="t1", mode=ThinkingMode.ANALYTICAL,
            reasoning_type=ReasoningType.DEDUCTIVE,
            input_prompt="test", thought_content="thinking",
            confidence=0.9, creativity_score=0.5, logical_score=0.8,
            timestamp=1.0, processing_time=0.1,
        )
        assert tp.id == "t1"
        assert tp.mode == ThinkingMode.ANALYTICAL
        assert tp.confidence == 0.9


class TestCreativeIdea:
    def test_create(self):
        ci = CreativeIdea(
            id="i1", concept="test", description="desc",
            novelty_score=0.8, feasibility_score=0.7, creativity_score=0.9,
            category="science", tags=["tag1"], timestamp=1.0,
        )
        assert ci.concept == "test"
        assert ci.novelty_score == 0.8
        assert ci.tags == ["tag1"]


class TestReasoningChain:
    def test_create(self):
        rc = ReasoningChain(
            id="r1", question="why?",
            reasoning_steps=["step1", "step2"],
            conclusion="because", confidence=0.85,
            reasoning_type=ReasoningType.CAUSAL,
            evidence=["ev1"], timestamp=1.0,
        )
        assert rc.question == "why?"
        assert len(rc.reasoning_steps) == 2
        assert rc.conclusion == "because"


class TestCognitiveCore:
    def test_init(self):
        core = CognitiveCore(db_path=":memory:")
        assert core.thought_history == []
        assert core.ideas == []
        assert core.reasoning_chains == []

    def test_think(self):
        core = CognitiveCore()
        thought = core.think("What is 2+2?")
        assert thought.mode == ThinkingMode.ANALYTICAL
        assert thought.thought_content  # non-empty
        assert thought.id.startswith("thought_")
        assert len(core.thought_history) == 1

    def test_think_creative_mode(self):
        core = CognitiveCore()
        thought = core.think("Be creative", mode=ThinkingMode.CREATIVE)
        assert thought.mode == ThinkingMode.CREATIVE

    def test_generate_idea(self):
        core = CognitiveCore()
        idea = core.generate_idea("solar energy", category="science")
        assert idea.concept == "solar energy"
        assert idea.category == "science"
        assert idea.novelty_score >= 0
        assert len(core.ideas) == 1

    def test_reason(self):
        core = CognitiveCore()
        chain = core.reason("Why is the sky blue?")
        assert chain.question == "Why is the sky blue?"
        assert chain.conclusion  # non-empty
        assert len(core.reasoning_chains) == 1

    def test_multiple_thoughts(self):
        core = CognitiveCore()
        core.think("q1")
        core.think("q2")
        core.think("q3")
        assert len(core.thought_history) == 3
        assert core.thought_history[0].id == "thought_0"
        assert core.thought_history[2].id == "thought_2"
