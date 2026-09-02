"""Tests for domains.cognitive.reasoning.advanced — reasoning strategies."""

from __future__ import annotations

import asyncio
import pytest
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


# ── Dataclasses ───────────────────────────────────────────────────────────────

class TestDataclasses:
    def test_thought_step_defaults(self):
        s = ThoughtStep(step_id=0, thought="test", reasoning_type="decomp", confidence=0.8)
        assert s.parent_id is None
        assert s.children_ids == []
        assert s.value == 0.0
        assert s.is_final is False

    def test_reasoning_result_fields(self):
        r = ReasoningResult(
            conclusion="yes",
            confidence=0.9,
            mode=ReasoningMode.CHAIN_OF_THOUGHT,
            steps=[],
            metadata={"k": "v"},
            execution_time_ms=10.0,
        )
        assert r.conclusion == "yes"
        assert r.mode == ReasoningMode.CHAIN_OF_THOUGHT

    def test_reasoning_mode_values(self):
        assert ReasoningMode.CHAIN_OF_THOUGHT.value == "chain_of_thought"
        assert ReasoningMode.TREE_OF_THOUGHTS.value == "tree_of_thoughts"
        assert ReasoningMode.REACT.value == "react"


# ── ChainOfThought ────────────────────────────────────────────────────────────

class TestChainOfThought:
    @pytest.mark.asyncio
    async def test_basic_reasoning(self):
        async def mock_llm(prompt):
            return "Therefore the answer is 42."
        cot = ChainOfThought(llm_call=mock_llm)
        result = await cot.reason("What is 6*7?", max_steps=3)
        assert result.mode == ReasoningMode.CHAIN_OF_THOUGHT
        assert len(result.steps) > 0
        assert result.execution_time_ms >= 0

    @pytest.mark.asyncio
    async def test_confidence_threshold_stops_early(self):
        call_count = 0
        async def mock_llm(prompt):
            nonlocal call_count
            call_count += 1
            return "Thus we conclude the answer."
        cot = ChainOfThought(llm_call=mock_llm)
        result = await cot.reason("problem", max_steps=10, confidence_threshold=0.5)
        assert call_count <= 3

    @pytest.mark.asyncio
    async def test_max_steps_limit(self):
        async def mock_llm(prompt):
            return "Thinking about this step."
        cot = ChainOfThought(llm_call=mock_llm)
        result = await cot.reason("problem", max_steps=2)
        assert len(result.steps) <= 2

    def test_evaluate_confidence_high(self):
        cot = ChainOfThought()
        assert cot._evaluate_confidence("Therefore we conclude. Thus.") >= 0.7

    def test_evaluate_confidence_low(self):
        cot = ChainOfThought()
        assert cot._evaluate_confidence("maybe") == 0.5

    def test_extract_subproblem(self):
        cot = ChainOfThought()
        assert cot._extract_subproblem("Remaining: solve for x") == "solve for x"
        assert cot._extract_subproblem("Next: find y") == "find y"
        assert cot._extract_subproblem("Now we need to compute z") == "compute z"
        assert cot._extract_subproblem("no pattern here") is None

    def test_extract_conclusion(self):
        cot = ChainOfThought()
        assert cot._extract_conclusion("Therefore the answer is 42.") == "the answer is 42"
        assert cot._extract_conclusion("Thus we find x=5.") == "we find x=5"
        assert cot._extract_conclusion("Answer: 7") == "7"
        assert cot._extract_conclusion("Conclusion: yes.") == "yes"
        assert cot._extract_conclusion("no patterns") == "no patterns"


# ── TreeOfThoughts ────────────────────────────────────────────────────────────

class TestTreeOfThoughts:
    @pytest.mark.asyncio
    async def test_basic_reasoning(self):
        async def mock_llm(prompt):
            return "Thought: this is a solution. Answer: 42."
        tot = TreeOfThoughts(llm_call=mock_llm, beam_width=2)
        result = await tot.reason("problem", max_depth=2)
        assert result.mode == ReasoningMode.TREE_OF_THOUGHTS
        assert len(result.steps) > 0

    def test_prune_nodes(self):
        tot = TreeOfThoughts()
        tot.nodes = {
            0: ThoughtStep(0, "root", "test", 0.5),
            1: ThoughtStep(1, "child1", "test", 0.8, value=0.8),
            2: ThoughtStep(2, "child2", "test", 0.2, value=0.2),
        }
        pruned = tot._prune_nodes([1, 2], threshold=0.3)
        assert 1 in pruned
        assert 2 not in pruned

    def test_get_path(self):
        tot = TreeOfThoughts()
        tot.nodes = {
            0: ThoughtStep(0, "root", "test", 0.5, parent_id=None),
            1: ThoughtStep(1, "child", "test", 0.5, parent_id=0),
            2: ThoughtStep(2, "grandchild", "test", 0.5, parent_id=1),
        }
        path = tot._get_path(2)
        assert path == [0, 1, 2]

    def test_evaluate_node(self):
        tot = TreeOfThoughts()
        assert tot._evaluate_node("solution found") > 0.5
        assert tot._evaluate_node("maybe") == 0.5

    def test_is_solution(self):
        tot = TreeOfThoughts()
        assert tot._is_solution("solution: 42") is True
        assert tot._is_solution("answer: 5") is True
        assert tot._is_solution("therefore x=1") is True
        assert tot._is_solution("conclusion: yes") is True
        assert tot._is_solution("no match") is False


# ── SelfConsistency ───────────────────────────────────────────────────────────

class TestSelfConsistency:
    @pytest.mark.asyncio
    async def test_basic_reasoning(self):
        async def mock_llm(prompt):
            return "Therefore the answer is 42."
        sc = SelfConsistency(llm_call=mock_llm, num_paths=3)
        result = await sc.reason("What is 6*7?")
        assert result.mode == ReasoningMode.SELF_CONSISTENCY
        assert len(result.steps) >= 3  # Multiple steps per path

    def test_majority_vote(self):
        sc = SelfConsistency()
        conclusions = ["42", "42", "42", "43", "41"]
        assert sc._majority_vote(conclusions) == "42"

    def test_majority_vote_empty(self):
        sc = SelfConsistency()
        assert sc._majority_vote([]) == ""

    def test_extract_conclusion(self):
        sc = SelfConsistency()
        assert sc._extract_conclusion("answer: 42") == "42"
        assert sc._extract_conclusion("Answer: 7") == "7"
        assert sc._extract_conclusion("no match here") == "no match here"


# ── ConstitutionalAI ──────────────────────────────────────────────────────────

class TestConstitutionalAI:
    @pytest.mark.asyncio
    async def test_basic_reasoning(self):
        async def mock_llm(prompt):
            if "critique" in prompt.lower():
                return "The response could be improved."
            return "The answer is correct and helpful."
        cai = ConstitutionalAI(llm_call=mock_llm)
        result = await cai.reason("Is this ethical?")
        assert result.mode == ReasoningMode.CONSTITUTIONAL
        assert len(result.steps) > 0


# ── CausalReasoning ───────────────────────────────────────────────────────────

class TestCausalReasoning:
    @pytest.mark.asyncio
    async def test_basic_reasoning(self):
        cr = CausalReasoning()
        result = await cr.reason("Rain causes wet ground. Wet ground causes slippery roads.")
        assert result.mode == ReasoningMode.CAUSAL
        assert len(result.steps) > 0

    def test_identify_causes(self):
        cr = CausalReasoning()
        causes = cr._identify_causes("Flooding because rain. Damage due to wind.")
        assert "rain" in [c.lower() for c in causes]
        assert "wind" in [c.lower() for c in causes]

    def test_identify_effects(self):
        cr = CausalReasoning()
        effects = cr._identify_effects("Rain fell. Therefore flooding occurred. Thus damage happened.")
        assert "flooding occurred" in [e.lower() for e in effects]
        assert "damage happened" in [e.lower() for e in effects]

    def test_identify_relationships(self):
        cr = CausalReasoning()
        rels = cr._identify_relationships("Rain causes flooding.")
        assert len(rels) > 0


# ── SyllogismReasoning ────────────────────────────────────────────────────────

class TestSyllogismReasoning:
    @pytest.mark.asyncio
    async def test_basic_reasoning(self):
        sr = SyllogismReasoning()
        result = await sr.reason("All men are mortal. Socrates is a man.")
        assert result.mode == ReasoningMode.SYLLOGISM
        assert len(result.steps) > 0

    def test_parse_premises(self):
        sr = SyllogismReasoning()
        premises = sr._parse_premises("All men are mortal. Socrates is a man.")
        assert len(premises) == 2
        assert "men" in premises[0].lower()

    def test_identify_figure(self):
        sr = SyllogismReasoning()
        # Figure 1: M-P, S-M
        premises = ["All men are mortal", "Socrates is a man"]
        fig = sr._identify_figure(premises)
        assert fig in [1, 2, 3, 4]


# ── ReActReasoning ────────────────────────────────────────────────────────────

class TestReActReasoning:
    @pytest.mark.asyncio
    async def test_basic_reasoning(self):
        async def mock_tool(name, args):
            return f"Result from {name}"
        tools = {"search": mock_tool}
        react = ReActReasoning(tool_registry=tools)
        result = await react.reason("Find information about cats.")
        assert result.mode == ReasoningMode.REACT
        assert len(result.steps) > 0

    def test_is_solved(self):
        react = ReActReasoning()
        assert react._is_solved("Final answer: 42") is True
        assert react._is_solved("Observation: something") is False


# ── advanced_reasoning ────────────────────────────────────────────────────────

class TestAdvancedReasoning:
    @pytest.mark.asyncio
    async def test_chain_of_thought(self):
        result = await advanced_reasoning("test", mode=ReasoningMode.CHAIN_OF_THOUGHT)
        assert result.mode == ReasoningMode.CHAIN_OF_THOUGHT

    @pytest.mark.asyncio
    async def test_tree_of_thoughts(self):
        result = await advanced_reasoning("test", mode=ReasoningMode.TREE_OF_THOUGHTS)
        assert result.mode == ReasoningMode.TREE_OF_THOUGHTS

    @pytest.mark.asyncio
    async def test_self_consistency(self):
        result = await advanced_reasoning("test", mode=ReasoningMode.SELF_CONSISTENCY)
        assert result.mode == ReasoningMode.SELF_CONSISTENCY

    @pytest.mark.asyncio
    async def test_constitutional(self):
        result = await advanced_reasoning("test", mode=ReasoningMode.CONSTITUTIONAL)
        assert result.mode == ReasoningMode.CONSTITUTIONAL

    @pytest.mark.asyncio
    async def test_causal(self):
        result = await advanced_reasoning("test", mode=ReasoningMode.CAUSAL)
        assert result.mode == ReasoningMode.CAUSAL

    @pytest.mark.asyncio
    async def test_syllogism(self):
        result = await advanced_reasoning("test", mode=ReasoningMode.SYLLOGISM)
        assert result.mode == ReasoningMode.SYLLOGISM

    @pytest.mark.asyncio
    async def test_react(self):
        result = await advanced_reasoning("test", mode=ReasoningMode.REACT)
        assert result.mode == ReasoningMode.REACT
