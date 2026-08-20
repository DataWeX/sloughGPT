"""Tests for domains.cognitive.reasoning.advanced — all 7 reasoning engines + factory.

Covers: ChainOfThought, TreeOfThoughts, SelfConsistency, ConstitutionalAI,
CausalReasoning, SyllogismReasoning, ReActReasoning, advanced_reasoning factory.
Pure logic tests with default LLM stubs, no external dependencies.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_core_dir = str(Path(__file__).resolve().parents[2])
if _core_dir not in sys.path:
    sys.path.insert(0, _core_dir)

from domains.cognitive.reasoning.advanced import (
    ReasoningMode,
    ThoughtStep,
    ReasoningResult,
    ChainOfThought,
    TreeOfThoughts,
    SelfConsistency,
    ConstitutionalAI,
    CausalReasoning,
    SyllogismReasoning,
    ReActReasoning,
    advanced_reasoning,
)


# ── Data classes ──────────────────────────────────────────────────────

class TestThoughtStep:
    def test_creation(self):
        s = ThoughtStep(step_id=0, thought="hello", reasoning_type="test", confidence=0.9)
        assert s.step_id == 0
        assert s.thought == "hello"
        assert s.confidence == 0.9
        assert s.parent_id is None
        assert s.children_ids == []
        assert s.value == 0.0
        assert s.is_final is False

    def test_with_parent(self):
        s = ThoughtStep(step_id=1, thought="b", reasoning_type="branch",
                        confidence=0.8, parent_id=0, value=0.6, is_final=True)
        assert s.parent_id == 0
        assert s.is_final is True


class TestReasoningResult:
    def test_creation(self):
        r = ReasoningResult(
            conclusion="42", confidence=0.9, mode=ReasoningMode.CHAIN_OF_THOUGHT,
            steps=[], metadata={}, execution_time_ms=10.0,
        )
        assert r.conclusion == "42"
        assert r.mode == ReasoningMode.CHAIN_OF_THOUGHT


# ── Chain of Thought ─────────────────────────────────────────────────

class TestChainOfThought:
    @pytest.mark.asyncio
    async def test_basic_reasoning(self):
        cot = ChainOfThought()
        result = await cot.reason("What is 2+2?")
        assert isinstance(result, ReasoningResult)
        assert result.mode == ReasoningMode.CHAIN_OF_THOUGHT
        assert len(result.steps) > 0
        assert result.execution_time_ms >= 0

    @pytest.mark.asyncio
    async def test_max_steps(self):
        cot = ChainOfThought()
        result = await cot.reason("Hard problem", max_steps=2)
        assert len(result.steps) <= 2

    @pytest.mark.asyncio
    async def test_custom_llm(self):
        calls = []
        async def mock_llm(prompt):
            calls.append(prompt)
            return "Therefore the answer is 42. Thus we conclude."
        cot = ChainOfThought(llm_call=mock_llm)
        result = await cot.reason("Test", confidence_threshold=0.99)
        assert len(calls) > 0
        assert "Therefore" in result.steps[-1].thought

    def test_evaluate_confidence_high(self):
        cot = ChainOfThought()
        assert cot._evaluate_confidence("Therefore it is done. Thus solved.") > 0.6

    def test_evaluate_confidence_low(self):
        cot = ChainOfThought()
        assert cot._evaluate_confidence("maybe") == 0.5

    def test_extract_subproblem(self):
        cot = ChainOfThought()
        assert cot._extract_subproblem("remaining: solve X") == "solve X"
        assert cot._extract_subproblem("next: do Y") == "do Y"
        assert cot._extract_subproblem("random text") is None

    def test_extract_conclusion(self):
        cot = ChainOfThought()
        assert "42" in cot._extract_conclusion("Therefore the answer is 42.")
        assert "hello" in cot._extract_conclusion("answer: hello world")
        assert cot._extract_conclusion("no match here") == "no match here"


# ── Tree of Thoughts ─────────────────────────────────────────────────

class TestTreeOfThoughts:
    @pytest.mark.asyncio
    async def test_basic_reasoning(self):
        tot = TreeOfThoughts()
        result = await tot.reason("Explore space", max_depth=2)
        assert result.mode == ReasoningMode.TREE_OF_THOUGHTS
        assert len(result.steps) > 0

    @pytest.mark.asyncio
    async def test_beam_width(self):
        tot = TreeOfThoughts(beam_width=2)
        result = await tot.reason("Test", max_depth=1)
        assert result.metadata["total_nodes"] >= 1

    def test_evaluate_node(self):
        tot = TreeOfThoughts()
        assert tot._evaluate_node("Therefore done") > 0.5
        assert tot._evaluate_node("maybe") == 0.5

    def test_is_solution(self):
        tot = TreeOfThoughts()
        assert tot._is_solution("The answer: 42")
        assert tot._is_solution("Therefore X")
        assert not tot._is_solution("maybe not")

    def test_prune_nodes(self):
        tot = TreeOfThoughts(beam_width=2)
        tot.nodes = {
            0: ThoughtStep(0, "a", "test", 0.5, value=0.9),
            1: ThoughtStep(1, "b", "test", 0.5, value=0.2),
            2: ThoughtStep(2, "c", "test", 0.5, value=0.7),
        }
        pruned = tot._prune_nodes([0, 1, 2], threshold=0.3)
        assert len(pruned) <= 2
        assert 1 not in pruned

    def test_get_path(self):
        tot = TreeOfThoughts()
        tot.nodes = {
            0: ThoughtStep(0, "root", "root", 1.0, parent_id=None),
            1: ThoughtStep(1, "a", "branch", 0.8, parent_id=0),
            2: ThoughtStep(2, "b", "branch", 0.7, parent_id=1),
        }
        path = tot._get_path(2)
        assert path == [0, 1, 2]


# ── Self-Consistency ─────────────────────────────────────────────────

class TestSelfConsistency:
    @pytest.mark.asyncio
    async def test_basic_reasoning(self):
        sc = SelfConsistency(num_paths=3)
        result = await sc.reason("Consensus problem")
        assert result.mode == ReasoningMode.SELF_CONSISTENCY
        assert result.confidence > 0
        assert len(result.steps) > 0

    def test_extract_conclusion(self):
        sc = SelfConsistency()
        assert sc._extract_conclusion("Answer: 42") == "42"
        assert len(sc._extract_conclusion("short")) > 0

    def test_majority_vote(self):
        sc = SelfConsistency()
        assert sc._majority_vote(["A", "A", "B"]) == "A"
        assert sc._majority_vote(["X"]) == "X"


# ── Constitutional AI ────────────────────────────────────────────────

class TestConstitutionalAI:
    @pytest.mark.asyncio
    async def test_basic_reasoning(self):
        ai = ConstitutionalAI()
        result = await ai.reason("Ethical dilemma")
        assert result.mode == ReasoningMode.CONSTITUTIONAL
        assert len(result.steps) == 3
        assert result.confidence == 0.9

    @pytest.mark.asyncio
    async def test_custom_principles(self):
        ai = ConstitutionalAI()
        result = await ai.reason("Test", custom_principles=["Be kind", "Be honest"])
        assert ai.principles == ["Be kind", "Be honest"]
        assert result.metadata["principles_used"] == 2

    def test_default_principles(self):
        ai = ConstitutionalAI()
        assert len(ai.principles) == 5


# ── Causal Reasoning ─────────────────────────────────────────────────

class TestCausalReasoning:
    @pytest.mark.asyncio
    async def test_basic_reasoning(self):
        cr = CausalReasoning()
        result = await cr.reason("Rain because clouds. Therefore ground is wet.")
        assert result.mode == ReasoningMode.CAUSAL
        assert "Causes" in result.conclusion

    def test_identify_causes(self):
        cr = CausalReasoning()
        causes = cr._identify_causes("It happened because of X. Due to Y.")
        assert len(causes) >= 1

    def test_identify_effects(self):
        cr = CausalReasoning()
        effects = cr._identify_effects("Therefore the result happened. Thus Z.")
        assert len(effects) >= 1

    def test_identify_relationships(self):
        cr = CausalReasoning()
        rels = cr._identify_relationships("Because X. Therefore Y.")
        assert len(rels) >= 1

    def test_build_conclusion_empty(self):
        cr = CausalReasoning()
        assert "Unable" in cr._build_causal_conclusion([], [], [])


# ── Syllogism ────────────────────────────────────────────────────────

class TestSyllogismReasoning:
    @pytest.mark.asyncio
    async def test_basic_reasoning(self):
        sr = SyllogismReasoning()
        result = await sr.reason("All men are mortal. Socrates is a man.")
        assert result.mode == ReasoningMode.SYLLOGISM
        assert result.metadata["valid"] is True

    def test_parse_premises(self):
        sr = SyllogismReasoning()
        p = sr._parse_premises("All A are B. X is A. Therefore X is B.")
        assert len(p) >= 2

    def test_parse_premises_default(self):
        sr = SyllogismReasoning()
        p = sr._parse_premises("short")
        assert "Socrates" in p[1]

    def test_identify_figure(self):
        sr = SyllogismReasoning()
        assert sr._identify_figure(["a", "b"]) == 1

    def test_identify_mood(self):
        sr = SyllogismReasoning()
        assert sr._identify_mood(["a", "b", "c"]) == "AAA"
        assert sr._identify_mood(["a"]) == "AA"

    def test_apply_rules_valid(self):
        sr = SyllogismReasoning()
        valid, _ = sr._apply_syllogistic_rules(1, "AAA")
        assert valid is True

    def test_apply_rules_invalid(self):
        sr = SyllogismReasoning()
        valid, _ = sr._apply_syllogistic_rules(1, "XYZ")
        assert valid is False

    def test_derive_conclusion(self):
        sr = SyllogismReasoning()
        c = sr._derive_conclusion(["All A are B", "X is A"])
        assert "Therefore" in c

    def test_derive_conclusion_insufficient(self):
        sr = SyllogismReasoning()
        assert "Insufficient" in sr._derive_conclusion(["one premise only"])


# ── ReAct ────────────────────────────────────────────────────────────

class TestReActReasoning:
    @pytest.mark.asyncio
    async def test_basic_reasoning(self):
        ra = ReActReasoning()
        result = await ra.reason("Find answer", max_steps=3)
        assert result.mode == ReasoningMode.REACT
        assert result.metadata["steps"] > 0

    @pytest.mark.asyncio
    async def test_with_tools(self):
        ra = ReActReasoning(tool_registry={"search": lambda q: "found"})
        result = await ra.reason("Search for X", max_steps=3)
        assert result.metadata["actions"] >= 1

    def test_is_solved(self):
        ra = ReActReasoning()
        assert ra._is_solved("The answer: 42")
        assert ra._is_solved("Final answer found")
        assert not ra._is_solved("thinking about it")

    @pytest.mark.asyncio
    async def test_act(self):
        ra = ReActReasoning(tool_registry={"calc": lambda x: "42"})
        tool, result = await ra._act("compute something")
        assert tool == "calc"
        assert "calc" in result


# ── Factory ──────────────────────────────────────────────────────────

class TestAdvancedReasoningFactory:
    @pytest.mark.asyncio
    async def test_chain_of_thought(self):
        r = await advanced_reasoning("Q", mode=ReasoningMode.CHAIN_OF_THOUGHT)
        assert r.mode == ReasoningMode.CHAIN_OF_THOUGHT

    @pytest.mark.asyncio
    async def test_tree_of_thoughts(self):
        r = await advanced_reasoning("Q", mode=ReasoningMode.TREE_OF_THOUGHTS, max_depth=1)
        assert r.mode == ReasoningMode.TREE_OF_THOUGHTS

    @pytest.mark.asyncio
    async def test_self_consistency(self):
        r = await advanced_reasoning("Q", mode=ReasoningMode.SELF_CONSISTENCY, num_paths=2)
        assert r.mode == ReasoningMode.SELF_CONSISTENCY

    @pytest.mark.asyncio
    async def test_constitutional(self):
        r = await advanced_reasoning("Q", mode=ReasoningMode.CONSTITUTIONAL)
        assert r.mode == ReasoningMode.CONSTITUTIONAL

    @pytest.mark.asyncio
    async def test_causal(self):
        r = await advanced_reasoning("Because X. Therefore Y.", mode=ReasoningMode.CAUSAL)
        assert r.mode == ReasoningMode.CAUSAL

    @pytest.mark.asyncio
    async def test_syllogism(self):
        r = await advanced_reasoning("All A are B. X is A.", mode=ReasoningMode.SYLLOGISM)
        assert r.mode == ReasoningMode.SYLLOGISM

    @pytest.mark.asyncio
    async def test_react(self):
        r = await advanced_reasoning("Q", mode=ReasoningMode.REACT)
        assert r.mode == ReasoningMode.REACT

    @pytest.mark.asyncio
    async def test_unknown_mode_defaults(self):
        r = await advanced_reasoning("Q", mode="unknown_mode")
        assert r.mode == ReasoningMode.CHAIN_OF_THOUGHT
