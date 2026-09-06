"""Tests for cognitive/reasoning — advanced reasoning strategies."""

from __future__ import annotations

import asyncio
import pytest

from domains.cognitive.reasoning import (
    ReasoningEngine,
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
    DeepReasoning,
    FormalLogicEngine,
    WorkingMemory,
    Term,
    Predicate,
    WellFormedFormula,
    LogicalOperator,
)


# ── ReasoningMode ─────────────────────────────────────────────────────────────


class TestReasoningMode:
    def test_has_all_modes(self):
        modes = [
            "CHAIN_OF_THOUGHT",
            "TREE_OF_THOUGHTS",
            "SELF_CONSISTENCY",
            "CONSTITUTIONAL",
            "REACT",
            "CAUSAL",
            "COUNTERFACTUAL",
            "SYLLOGISM",
        ]
        for mode in modes:
            assert hasattr(ReasoningMode, mode)

    def test_mode_values(self):
        assert ReasoningMode.CHAIN_OF_THOUGHT.value == "chain_of_thought"
        assert ReasoningMode.TREE_OF_THOUGHTS.value == "tree_of_thoughts"
        assert ReasoningMode.SELF_CONSISTENCY.value == "self_consistency"

    def test_mode_count(self):
        assert len(ReasoningMode) == 8


# ── ThoughtStep ───────────────────────────────────────────────────────────────


class TestThoughtStep:
    def test_creation(self):
        step = ThoughtStep(
            step_id=1,
            thought="Test thought",
            reasoning_type="decomposition",
            confidence=0.8,
        )
        assert step.step_id == 1
        assert step.thought == "Test thought"
        assert step.reasoning_type == "decomposition"
        assert step.confidence == 0.8

    def test_defaults(self):
        step = ThoughtStep(step_id=0, thought="x", reasoning_type="y", confidence=0.5)
        assert step.parent_id is None
        assert step.children_ids == []
        assert step.value == 0.0
        assert step.is_final is False

    def test_with_parent(self):
        step = ThoughtStep(
            step_id=1, thought="x", reasoning_type="y", confidence=0.5, parent_id=0
        )
        assert step.parent_id == 0

    def test_with_children(self):
        step = ThoughtStep(
            step_id=0, thought="x", reasoning_type="y", confidence=0.5, children_ids=[1, 2]
        )
        assert step.children_ids == [1, 2]


# ── ReasoningResult ───────────────────────────────────────────────────────────


class TestReasoningResult:
    def test_creation(self):
        result = ReasoningResult(
            conclusion="Answer",
            confidence=0.9,
            mode=ReasoningMode.CHAIN_OF_THOUGHT,
            steps=[],
            metadata={},
            execution_time_ms=100.0,
        )
        assert result.conclusion == "Answer"
        assert result.confidence == 0.9
        assert result.mode == ReasoningMode.CHAIN_OF_THOUGHT
        assert result.execution_time_ms == 100.0

    def test_with_steps(self):
        steps = [
            ThoughtStep(0, "step1", "type1", 0.5),
            ThoughtStep(1, "step2", "type2", 0.7),
        ]
        result = ReasoningResult(
            conclusion="Done", confidence=0.8, mode=ReasoningMode.TREE_OF_THOUGHTS,
            steps=steps, metadata={"depth": 2}, execution_time_ms=200.0,
        )
        assert len(result.steps) == 2
        assert result.metadata["depth"] == 2


# ── ChainOfThought ────────────────────────────────────────────────────────────


class TestChainOfThought:
    @pytest.fixture
    def cot(self):
        return ChainOfThought()

    @pytest.mark.asyncio
    async def test_reason_returns_result(self, cot):
        result = await cot.reason("What is 2+2?")
        assert isinstance(result, ReasoningResult)
        assert result.mode == ReasoningMode.CHAIN_OF_THOUGHT

    @pytest.mark.asyncio
    async def test_reason_has_steps(self, cot):
        result = await cot.reason("Solve this problem")
        assert len(result.steps) > 0

    @pytest.mark.asyncio
    async def test_reason_confidence(self, cot):
        result = await cot.reason("Simple problem")
        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_reason_with_custom_llm(self):
        async def mock_llm(prompt):
            return "Therefore, the answer is 42."

        cot = ChainOfThought(llm_call=mock_llm)
        result = await cot.reason("What is 6*7?")
        assert "42" in result.conclusion

    @pytest.mark.asyncio
    async def test_reason_max_steps(self, cot):
        result = await cot.reason("Hard problem", max_steps=3)
        assert len(result.steps) <= 3

    @pytest.mark.asyncio
    async def test_evaluate_confidence(self, cot):
        assert cot._evaluate_confidence("Therefore, x is true") > 0.5
        assert cot._evaluate_confidence("maybe") == 0.5

    @pytest.mark.asyncio
    async def test_extract_conclusion(self, cot):
        assert "answer is 5" in cot._extract_conclusion("Therefore, answer is 5.")
        assert cot._extract_conclusion("Solution: done") == "done"

    @pytest.mark.asyncio
    async def test_extract_subproblem(self, cot):
        assert cot._extract_subproblem("Next: solve part B") == "solve part B"
        assert cot._extract_subproblem("no pattern") is None


# ── TreeOfThoughts ────────────────────────────────────────────────────────────


class TestTreeOfThoughts:
    @pytest.fixture
    def tot(self):
        return TreeOfThoughts(beam_width=2)

    @pytest.mark.asyncio
    async def test_reason_returns_result(self, tot):
        result = await tot.reason("Explore this")
        assert isinstance(result, ReasoningResult)
        assert result.mode == ReasoningMode.TREE_OF_THOUGHTS

    @pytest.mark.asyncio
    async def test_reason_has_nodes(self, tot):
        result = await tot.reason("Problem")
        assert len(tot.nodes) > 0

    @pytest.mark.asyncio
    async def test_reason_with_depth(self, tot):
        result = await tot.reason("Deep problem", max_depth=2)
        assert result.metadata["depth"] == 2

    @pytest.mark.asyncio
    async def test_evaluate_node(self, tot):
        assert tot._evaluate_node("Therefore, conclusion") > 0.5
        assert tot._evaluate_node("maybe") == 0.5

    @pytest.mark.asyncio
    async def test_is_solution(self, tot):
        assert tot._is_solution("Answer: 42") is True
        assert tot._is_solution("no answer here") is False

    @pytest.mark.asyncio
    async def test_prune_nodes(self, tot):
        tot.nodes = {
            0: ThoughtStep(0, "a", "t", 0.5, value=0.9),
            1: ThoughtStep(1, "b", "t", 0.5, value=0.2),
            2: ThoughtStep(2, "c", "t", 0.5, value=0.7),
        }
        pruned = tot._prune_nodes([0, 1, 2], threshold=0.3)
        assert 1 not in pruned

    @pytest.mark.asyncio
    async def test_get_path(self, tot):
        tot.nodes = {
            0: ThoughtStep(0, "root", "t", 0.5, parent_id=None),
            1: ThoughtStep(1, "child", "t", 0.5, parent_id=0),
        }
        path = tot._get_path(1)
        assert path == [0, 1]


# ── SelfConsistency ───────────────────────────────────────────────────────────


class TestSelfConsistency:
    @pytest.fixture
    def sc(self):
        return SelfConsistency(num_paths=3)

    @pytest.mark.asyncio
    async def test_reason_returns_result(self, sc):
        result = await sc.reason("Consistent problem")
        assert isinstance(result, ReasoningResult)
        assert result.mode == ReasoningMode.SELF_CONSISTENCY

    @pytest.mark.asyncio
    async def test_reason_metadata(self, sc):
        result = await sc.reason("Problem")
        assert result.metadata["num_paths"] == 3

    @pytest.mark.asyncio
    async def test_majority_vote(self, sc):
        assert sc._majority_vote(["a", "a", "b"]) == "a"
        assert sc._majority_vote([]) == ""

    @pytest.mark.asyncio
    async def test_extract_conclusion(self, sc):
        assert sc._extract_conclusion("Answer: 42") == "42"
        assert len(sc._extract_conclusion("Some long text without answer keyword")) <= 100


# ── ConstitutionalAI ──────────────────────────────────────────────────────────


class TestConstitutionalAI:
    @pytest.fixture
    def ai(self):
        return ConstitutionalAI()

    @pytest.mark.asyncio
    async def test_reason_returns_result(self, ai):
        result = await ai.reason("Helpful response")
        assert isinstance(result, ReasoningResult)
        assert result.mode == ReasoningMode.CONSTITUTIONAL

    @pytest.mark.asyncio
    async def test_reason_has_principles(self, ai):
        result = await ai.reason("Problem")
        assert result.metadata["principles_used"] == len(ConstitutionalAI.PRINCIPLES)

    @pytest.mark.asyncio
    async def test_custom_principles(self, ai):
        custom = ["Be creative"]
        result = ai._generate_initial  # just checking method exists
        # Custom principles test via reason
        result = await ai.reason("Problem", custom_principles=custom)
        assert result.metadata["principles_used"] == 1

    @pytest.mark.asyncio
    async def test_review_history(self, ai):
        assert ai.review_history == []


# ── ReasoningEngine ───────────────────────────────────────────────────────────


class TestReasoningEngine:
    @pytest.fixture
    def engine(self):
        return ReasoningEngine()

    @pytest.mark.asyncio
    async def test_init(self, engine):
        assert engine.mode == ReasoningMode.CHAIN_OF_THOUGHT
        assert engine.reasoning_history == []

    @pytest.mark.asyncio
    async def test_reason(self, engine):
        result = await engine.reason("What is AI?", {})
        assert isinstance(result, str)
        assert len(engine.reasoning_history) == 1

    @pytest.mark.asyncio
    async def test_set_mode(self, engine):
        await engine.set_mode(ReasoningMode.TREE_OF_THOUGHTS)
        assert engine.mode == ReasoningMode.TREE_OF_THOUGHTS

    @pytest.mark.asyncio
    async def test_get_history(self, engine):
        await engine.reason("Problem 1", {})
        await engine.reason("Problem 2", {})
        history = await engine.get_history()
        assert len(history) == 2

    @pytest.mark.asyncio
    async def test_assert_and_query(self, engine):
        engine.assert_fact("likes", "alice", "pizza")
        assert engine.query("likes", "alice", "pizza") is True
        assert engine.query("likes", "bob", "pizza") is False


# ── FormalLogicEngine ─────────────────────────────────────────────────────────


class TestFormalLogicEngine:
    def test_init(self):
        engine = FormalLogicEngine()
        assert engine is not None

    def test_assert_predicate(self):
        engine = FormalLogicEngine()
        engine.assert_predicate("human", "socrates")
        assert engine.knowledge_base is not None

    def test_prove_syllogism(self):
        engine = FormalLogicEngine()
        result = engine.prove_syllogism(
            ("All", "are", "mortal"),  # (quantifier, copula, predicate)
            ("Socrates", "is", "human"),
            ("Socrates", "is", "mortal"),
        )
        assert isinstance(result, dict)
        assert "valid" in result


# ── WorkingMemory ─────────────────────────────────────────────────────────────


class TestWorkingMemory:
    def test_init(self):
        wm = WorkingMemory()
        assert wm is not None

    def test_add_and_get_recent(self):
        wm = WorkingMemory()
        wm.add("item1")
        wm.add("item2")
        recent = wm.get_recent(5)
        assert "item1" in recent
        assert "item2" in recent

    def test_capacity_eviction(self):
        wm = WorkingMemory(capacity=2)
        wm.add("a")
        wm.add("b")
        wm.add("c")  # Should evict 'a'
        assert len(wm.items) == 2
        assert "a" not in wm.items

    def test_clear(self):
        wm = WorkingMemory()
        wm.add("item")
        wm.clear()
        assert len(wm.items) == 0


# ── DeepReasoning ─────────────────────────────────────────────────────────────


class TestDeepReasoning:
    @pytest.mark.asyncio
    async def test_reason(self):
        dr = DeepReasoning()
        result = await dr.reason("Deep problem", max_depth=2)
        assert isinstance(result, ReasoningResult)
