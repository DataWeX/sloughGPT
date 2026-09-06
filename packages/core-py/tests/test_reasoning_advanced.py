"""Tests for cognitive/reasoning/advanced.py — 7 reasoning strategies."""

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


# ── Data classes ──────────────────────────────────────────────────────────────


class TestReasoningMode:
    def test_all_modes_exist(self):
        modes = [m.value for m in ReasoningMode]
        assert "chain_of_thought" in modes
        assert "tree_of_thoughts" in modes
        assert "self_consistency" in modes
        assert "constitutional" in modes
        assert "react" in modes
        assert "causal" in modes
        assert "syllogism" in modes
        assert "counterfactual" in modes

    def test_mode_count(self):
        assert len(ReasoningMode) == 8


class TestThoughtStep:
    def test_defaults(self):
        step = ThoughtStep(step_id=0, thought="test", reasoning_type="decomposition", confidence=0.8)
        assert step.parent_id is None
        assert step.children_ids == []
        assert step.value == 0.0
        assert step.is_final is False

    def test_with_parent(self):
        step = ThoughtStep(step_id=1, thought="child", reasoning_type="branch", confidence=0.7, parent_id=0)
        assert step.parent_id == 0


class TestReasoningResult:
    def test_fields(self):
        result = ReasoningResult(
            conclusion="answer",
            confidence=0.9,
            mode=ReasoningMode.CHAIN_OF_THOUGHT,
            steps=[],
            metadata={},
            execution_time_ms=1.5,
        )
        assert result.conclusion == "answer"
        assert result.confidence == 0.9
        assert result.execution_time_ms == 1.5


# ── Chain of Thought ──────────────────────────────────────────────────────────


class TestChainOfThought:
    @pytest.mark.asyncio
    async def test_basic_reasoning(self):
        cot = ChainOfThought()
        result = await cot.reason("What is 2+2?")
        assert isinstance(result, ReasoningResult)
        assert result.mode == ReasoningMode.CHAIN_OF_THOUGHT
        assert len(result.steps) > 0
        assert result.confidence > 0

    @pytest.mark.asyncio
    async def test_custom_llm(self):
        call_count = 0

        async def mock_llm(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                return "Therefore, the answer is 42. Thus we conclude."
            return "Step 1: We analyze the problem. Remaining: find the answer."

        cot = ChainOfThought(llm_call=mock_llm)
        result = await cot.reason("What is the meaning of life?", confidence_threshold=0.9)
        assert result.mode == ReasoningMode.CHAIN_OF_THOUGHT
        assert call_count >= 2

    @pytest.mark.asyncio
    async def test_max_steps_limit(self):
        async def never_confident(prompt: str) -> str:
            return "Analyzing further..."

        cot = ChainOfThought(llm_call=never_confident)
        result = await cot.reason("Hard problem", max_steps=3)
        assert len(result.steps) <= 3

    def test_evaluate_confidence_high(self):
        cot = ChainOfThought()
        assert cot._evaluate_confidence("Therefore, thus, hence, so.") >= 0.9

    def test_evaluate_confidence_low(self):
        cot = ChainOfThought()
        assert cot._evaluate_confidence("maybe") < 0.7

    def test_extract_subproblem_remaining(self):
        cot = ChainOfThought()
        result = cot._extract_subproblem("We found X. Remaining: solve Y")
        assert result == "solve Y"

    def test_extract_subproblem_none(self):
        cot = ChainOfThought()
        assert cot._extract_subproblem("Just a statement.") is None

    def test_extract_conclusion_therefore(self):
        cot = ChainOfThought()
        result = cot._extract_conclusion("We analyzed everything. Therefore the answer is 42.")
        assert "42" in result

    def test_extract_conclusion_fallback(self):
        cot = ChainOfThought()
        result = cot._extract_conclusion("No conclusion markers here")
        assert result == "No conclusion markers here"


# ── Tree of Thoughts ──────────────────────────────────────────────────────────


class TestTreeOfThoughts:
    @pytest.mark.asyncio
    async def test_basic_reasoning(self):
        tot = TreeOfThoughts(beam_width=2)
        result = await tot.reason("Explore this problem", max_depth=2)
        assert isinstance(result, ReasoningResult)
        assert result.mode == ReasoningMode.TREE_OF_THOUGHTS
        assert len(result.steps) > 0

    @pytest.mark.asyncio
    async def test_custom_llm(self):
        async def mock_llm(prompt: str) -> str:
            return "Solution: the answer is found through analysis."

        tot = TreeOfThoughts(llm_call=mock_llm, beam_width=2)
        result = await tot.reason("Problem", max_depth=2)
        assert result.mode == ReasoningMode.TREE_OF_THOUGHTS

    def test_evaluate_node_solution(self):
        tot = TreeOfThoughts()
        assert tot._evaluate_node("The answer is 42.") > 0.5
        assert tot._evaluate_node("Just thinking") == 0.5

    def test_is_solution(self):
        tot = TreeOfThoughts()
        assert tot._is_solution("Answer: 42") is True
        assert tot._is_solution("Solution: found it") is True
        assert tot._is_solution("The answer is 42") is False  # no colon
        assert tot._is_solution("Just thinking") is False

    def test_prune_nodes(self):
        tot = TreeOfThoughts(beam_width=2)
        tot.nodes = {
            0: ThoughtStep(0, "a", "branch", 0.8, value=0.9),
            1: ThoughtStep(1, "b", "branch", 0.7, value=0.2),
            2: ThoughtStep(2, "c", "branch", 0.6, value=0.8),
        }
        result = tot._prune_nodes([0, 1, 2], threshold=0.3)
        assert 1 not in result  # value 0.2 < threshold
        assert len(result) <= 2

    def test_get_path(self):
        tot = TreeOfThoughts()
        tot.nodes = {
            0: ThoughtStep(0, "root", "root", 1.0, parent_id=None),
            1: ThoughtStep(1, "child", "branch", 0.8, parent_id=0),
            2: ThoughtStep(2, "grandchild", "branch", 0.7, parent_id=1),
        }
        path = tot._get_path(2)
        assert path == [0, 1, 2]


# ── Self-Consistency ──────────────────────────────────────────────────────────


class TestSelfConsistency:
    @pytest.mark.asyncio
    async def test_basic_reasoning(self):
        sc = SelfConsistency(num_paths=3)
        result = await sc.reason("What is 2+2?")
        assert isinstance(result, ReasoningResult)
        assert result.mode == ReasoningMode.SELF_CONSISTENCY
        assert len(result.steps) > 0

    @pytest.mark.asyncio
    async def test_custom_llm(self):
        call_count = 0

        async def mock_llm(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            return "Answer: 42"

        sc = SelfConsistency(llm_call=mock_llm, num_paths=2)
        result = await sc.reason("Problem")
        assert call_count >= 2

    def test_majority_vote(self):
        sc = SelfConsistency()
        assert sc._majority_vote(["A", "A", "B"]) == "A"
        assert sc._majority_vote(["X"]) == "X"
        assert sc._majority_vote([]) == ""

    def test_extract_conclusion_with_answer(self):
        sc = SelfConsistency()
        result = sc._extract_conclusion("Some reasoning. Answer: 42 is correct.")
        assert "42" in result

    def test_extract_conclusion_fallback(self):
        sc = SelfConsistency()
        result = sc._extract_conclusion("Just some text at the end of the reasoning chain")
        assert len(result) <= 100


# ── Constitutional AI ────────────────────────────────────────────────────────


class TestConstitutionalAI:
    @pytest.mark.asyncio
    async def test_basic_reasoning(self):
        ai = ConstitutionalAI()
        result = await ai.reason("Should I lie?")
        assert isinstance(result, ReasoningResult)
        assert result.mode == ReasoningMode.CONSTITUTIONAL
        assert len(result.steps) == 3  # initial, critique, revision

    @pytest.mark.asyncio
    async def test_custom_principles(self):
        ai = ConstitutionalAI()
        result = await ai.reason("Problem", custom_principles=["Be truthful", "Be kind"])
        assert result.metadata["principles_used"] == 2

    @pytest.mark.asyncio
    async def test_custom_llm(self):
        calls = []

        async def mock_llm(prompt: str) -> str:
            calls.append(prompt[:30])
            return "Revised response based on principles."

        ai = ConstitutionalAI(llm_call=mock_llm)
        await ai.reason("Test")
        assert len(calls) == 3  # initial, critique, revise


# ── Causal Reasoning ──────────────────────────────────────────────────────────


class TestCausalReasoning:
    @pytest.mark.asyncio
    async def test_basic_reasoning(self):
        cr = CausalReasoning()
        result = await cr.reason("Rain caused flooding because the ground was saturated.")
        assert isinstance(result, ReasoningResult)
        assert result.mode == ReasoningMode.CAUSAL
        assert len(result.steps) > 0

    def test_identify_causes_because(self):
        cr = CausalReasoning()
        causes = cr._identify_causes("The event happened because of rain.")
        assert len(causes) >= 1
        assert "rain" in causes[0].lower()

    def test_identify_causes_due_to(self):
        cr = CausalReasoning()
        causes = cr._identify_causes("The failure was due to low battery.")
        assert len(causes) >= 1

    def test_identify_causes_no_match(self):
        cr = CausalReasoning()
        causes = cr._identify_causes("Just a random sentence.")
        assert causes == ["Unknown cause"]

    def test_identify_effects_therefore(self):
        cr = CausalReasoning()
        effects = cr._identify_effects("It rained, therefore the ground is wet.")
        assert len(effects) >= 1

    def test_identify_effects_no_match(self):
        cr = CausalReasoning()
        effects = cr._identify_effects("No causal language here.")
        assert effects == ["Unknown effect"]

    def test_build_causal_conclusion(self):
        cr = CausalReasoning()
        result = cr._build_causal_conclusion(
            ["rain"], ["flooding"], [("rain", "flooding", 0.8)]
        )
        assert "rain" in result
        assert "flooding" in result

    def test_build_causal_conclusion_empty(self):
        cr = CausalReasoning()
        result = cr._build_causal_conclusion([], [], [])
        assert "Unable" in result


# ── Syllogism Reasoning ───────────────────────────────────────────────────────


class TestSyllogismReasoning:
    @pytest.mark.asyncio
    async def test_basic_reasoning(self):
        sr = SyllogismReasoning()
        result = await sr.reason("All humans are mortal. Socrates is human.")
        assert isinstance(result, ReasoningResult)
        assert result.mode == ReasoningMode.SYLLOGISM
        assert result.metadata["valid"] is True  # AAA mood is valid

    def test_parse_premises(self):
        sr = SyllogismReasoning()
        premises = sr._parse_premises("All men are mortal. Socrates is a man.")
        assert len(premises) >= 2

    def test_parse_premises_fallback(self):
        sr = SyllogismReasoning()
        premises = sr._parse_premises("short")
        assert premises == ["All humans are mortal.", "Socrates is human."]

    def test_identify_figure(self):
        sr = SyllogismReasoning()
        assert sr._identify_figure(["A", "B"]) == 1

    def test_identify_mood(self):
        sr = SyllogismReasoning()
        assert sr._identify_mood(["A", "B", "C"]) == "AAA"
        assert sr._identify_mood(["A"]) == "AA"

    def test_apply_syllogistic_rules_valid(self):
        sr = SyllogismReasoning()
        valid, msg = sr._apply_syllogistic_rules(1, "AAA")
        assert valid is True
        assert "valid" in msg.lower()

    def test_apply_syllogistic_rules_invalid(self):
        sr = SyllogismReasoning()
        valid, msg = sr._apply_syllogistic_rules(1, "OOO")
        assert valid is False

    def test_derive_conclusion(self):
        sr = SyllogismReasoning()
        result = sr._derive_conclusion(["All men are mortal", "Socrates is a man"])
        assert "Socrates" in result

    def test_derive_conclusion_insufficient(self):
        sr = SyllogismReasoning()
        result = sr._derive_conclusion(["Only one premise"])
        assert "Insufficient" in result


# ── ReAct Reasoning ───────────────────────────────────────────────────────────


class TestReActReasoning:
    @pytest.mark.asyncio
    async def test_basic_reasoning(self):
        rr = ReActReasoning()
        result = await rr.reason("Solve this problem")
        assert isinstance(result, ReasoningResult)
        assert result.mode == ReasoningMode.REACT
        assert len(result.steps) > 0

    @pytest.mark.asyncio
    async def test_with_tools(self):
        rr = ReActReasoning(tool_registry={"search": lambda q: "result"})
        result = await rr.reason("Find information", max_steps=3)
        assert result.metadata["actions"] >= 1

    @pytest.mark.asyncio
    async def test_solved_detection(self):
        rr = ReActReasoning()
        result = await rr.reason("Answer: the answer is 42")
        assert any(s.is_final for s in result.steps)

    def test_is_solved(self):
        rr = ReActReasoning()
        assert rr._is_solved("Answer: 42") is True
        assert rr._is_solved("Solution: found it") is True
        assert rr._is_solved("Just thinking") is False

    @pytest.mark.asyncio
    async def test_action_limit(self):
        rr = ReActReasoning(tool_registry={"search": lambda q: "result"})
        result = await rr.reason("Problem", max_steps=10)
        assert result.metadata["actions"] <= 2


# ── Factory function ──────────────────────────────────────────────────────────


class TestAdvancedReasoning:
    @pytest.mark.asyncio
    async def test_chain_of_thought(self):
        result = await advanced_reasoning("Problem", mode=ReasoningMode.CHAIN_OF_THOUGHT)
        assert result.mode == ReasoningMode.CHAIN_OF_THOUGHT

    @pytest.mark.asyncio
    async def test_tree_of_thoughts(self):
        result = await advanced_reasoning("Problem", mode=ReasoningMode.TREE_OF_THOUGHTS, max_depth=2)
        assert result.mode == ReasoningMode.TREE_OF_THOUGHTS

    @pytest.mark.asyncio
    async def test_self_consistency(self):
        result = await advanced_reasoning("Problem", mode=ReasoningMode.SELF_CONSISTENCY, num_paths=2)
        assert result.mode == ReasoningMode.SELF_CONSISTENCY

    @pytest.mark.asyncio
    async def test_constitutional(self):
        result = await advanced_reasoning("Problem", mode=ReasoningMode.CONSTITUTIONAL)
        assert result.mode == ReasoningMode.CONSTITUTIONAL

    @pytest.mark.asyncio
    async def test_causal(self):
        result = await advanced_reasoning("Rain caused flooding.", mode=ReasoningMode.CAUSAL)
        assert result.mode == ReasoningMode.CAUSAL

    @pytest.mark.asyncio
    async def test_syllogism(self):
        result = await advanced_reasoning("All men are mortal. Socrates is a man.", mode=ReasoningMode.SYLLOGISM)
        assert result.mode == ReasoningMode.SYLLOGISM

    @pytest.mark.asyncio
    async def test_react(self):
        result = await advanced_reasoning("Problem", mode=ReasoningMode.REACT)
        assert result.mode == ReasoningMode.REACT

    @pytest.mark.asyncio
    async def test_unknown_mode_defaults_to_cot(self):
        result = await advanced_reasoning("Problem", mode="unknown_mode")
        assert result.mode == ReasoningMode.CHAIN_OF_THOUGHT
