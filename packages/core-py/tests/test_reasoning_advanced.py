"""Tests for cognitive/reasoning/advanced.py — dataclasses + pure helpers."""

import pytest
from domains.cognitive.reasoning.advanced import (
    ReasoningMode,
    ThoughtStep,
    ReasoningResult,
    ChainOfThought,
    TreeOfThoughts,
    SelfConsistency,
    CausalReasoning,
    SyllogismReasoning,
)


class TestReasoningMode:
    def test_all_modes_have_values(self):
        for mode in ReasoningMode:
            assert isinstance(mode.value, str)
            assert len(mode.value) > 0

    def test_modes_are_unique(self):
        values = [m.value for m in ReasoningMode]
        assert len(values) == len(set(values))

    def test_chain_of_thought_exists(self):
        assert ReasoningMode.CHAIN_OF_THOUGHT.value == "chain_of_thought"

    def test_tree_of_thoughts_exists(self):
        assert ReasoningMode.TREE_OF_THOUGHTS.value == "tree_of_thoughts"

    def test_self_consistency_exists(self):
        assert ReasoningMode.SELF_CONSISTENCY.value == "self_consistency"


class TestThoughtStep:
    def test_creation(self):
        step = ThoughtStep(step_id=0, thought="test", reasoning_type="decomp", confidence=0.8)
        assert step.step_id == 0
        assert step.thought == "test"
        assert step.confidence == 0.8
        assert step.parent_id is None
        assert step.children_ids == []
        assert step.value == 0.0
        assert step.is_final is False

    def test_with_parent(self):
        step = ThoughtStep(step_id=1, thought="x", reasoning_type="r", confidence=0.5, parent_id=0)
        assert step.parent_id == 0

    def test_with_children(self):
        step = ThoughtStep(step_id=0, thought="x", reasoning_type="r", confidence=0.5, children_ids=[1, 2])
        assert step.children_ids == [1, 2]

    def test_final_step(self):
        step = ThoughtStep(step_id=5, thought="done", reasoning_type="conclusion", confidence=0.95, is_final=True)
        assert step.is_final is True

    def test_value(self):
        step = ThoughtStep(step_id=0, thought="v", reasoning_type="r", confidence=0.5, value=3.14)
        assert step.value == 3.14


class TestReasoningResult:
    def test_creation(self):
        result = ReasoningResult(
            conclusion="42",
            confidence=0.9,
            mode=ReasoningMode.CHAIN_OF_THOUGHT,
            steps=[],
            metadata={"key": "val"},
            execution_time_ms=100.0,
        )
        assert result.conclusion == "42"
        assert result.confidence == 0.9
        assert result.mode == ReasoningMode.CHAIN_OF_THOUGHT
        assert result.execution_time_ms == 100.0

    def test_with_steps(self):
        step = ThoughtStep(step_id=0, thought="s", reasoning_type="r", confidence=0.7)
        result = ReasoningResult(
            conclusion="c", confidence=0.8, mode=ReasoningMode.REACT,
            steps=[step], metadata={}, execution_time_ms=50.0,
        )
        assert len(result.steps) == 1


class TestChainOfThought:
    def test_init_default(self):
        cot = ChainOfThought()
        assert cot.llm_call is not None
        assert cot.steps == []

    def test_init_custom_llm(self):
        async def my_llm(prompt):
            return "answer"
        cot = ChainOfThought(llm_call=my_llm)
        assert cot.llm_call is my_llm

    def test_evaluate_confidence_high(self):
        cot = ChainOfThought()
        conf = cot._evaluate_confidence("Therefore, the answer is 42.")
        assert conf >= 0.7

    def test_evaluate_confidence_low(self):
        cot = ChainOfThought()
        conf = cot._evaluate_confidence("maybe")
        assert conf <= 0.6

    def test_extract_subproblem(self):
        cot = ChainOfThought()
        result = cot._extract_subproblem("We need to remaining: solve the equation x+1=0")
        assert result is not None
        assert "solve" in result.lower() or "x" in result

    def test_extract_subproblem_none(self):
        cot = ChainOfThought()
        result = cot._extract_subproblem("just a random sentence")
        assert result is None

    def test_extract_conclusion_therefore(self):
        cot = ChainOfThought()
        result = cot._extract_conclusion("Therefore, the answer is 42.")
        assert "42" in result

    def test_extract_conclusion_answer(self):
        cot = ChainOfThought()
        result = cot._extract_conclusion("Answer: blue")
        assert "blue" in result

    def test_extract_conclusion_fallback(self):
        cot = ChainOfThought()
        result = cot._extract_conclusion("random text with no pattern")
        assert result == "random text with no pattern"

    def test_extract_subproblem_next(self):
        cot = ChainOfThought()
        result = cot._extract_subproblem("next: we should check the data")
        assert result is not None

    def test_extract_subproblem_now(self):
        cot = ChainOfThought()
        result = cot._extract_subproblem("now we need to verify the result")
        assert result is not None


class TestTreeOfThoughts:
    def test_init_default(self):
        tot = TreeOfThoughts()
        assert tot.beam_width == 3

    def test_init_custom(self):
        tot = TreeOfThoughts(beam_width=5)
        assert tot.beam_width == 5

    def test_evaluate_node(self):
        tot = TreeOfThoughts()
        score = tot._evaluate_node("therefore the solution is correct")
        assert isinstance(score, float)

    def test_is_solution_answer(self):
        tot = TreeOfThoughts()
        assert tot._is_solution("the answer: 42") is True

    def test_is_solution_therefore(self):
        tot = TreeOfThoughts()
        assert tot._is_solution("therefore x equals 5") is True

    def test_is_not_solution(self):
        tot = TreeOfThoughts()
        assert tot._is_solution("I'm still thinking") is False


class TestSelfConsistency:
    def test_init_default(self):
        sc = SelfConsistency()
        assert sc.num_paths == 5

    def test_init_custom(self):
        sc = SelfConsistency(num_paths=10)
        assert sc.num_paths == 10

    def test_majority_vote(self):
        sc = SelfConsistency()
        result = sc._majority_vote(["blue", "red", "blue", "blue", "green"])
        assert result == "blue"

    def test_majority_vote_single(self):
        sc = SelfConsistency()
        result = sc._majority_vote(["only"])
        assert result == "only"

    def test_extract_conclusion(self):
        sc = SelfConsistency()
        result = sc._extract_conclusion("therefore the result is 5")
        assert "5" in result


class TestCausalReasoning:
    def test_init(self):
        cr = CausalReasoning()
        assert hasattr(cr, '_identify_causes')

    def test_identify_causes(self):
        cr = CausalReasoning()
        causes = cr._identify_causes("Because it rained, the ground is wet")
        assert isinstance(causes, list)

    def test_identify_effects(self):
        cr = CausalReasoning()
        effects = cr._identify_effects("It rained, so the ground is wet")
        assert isinstance(effects, list)

    def test_identify_relationships(self):
        cr = CausalReasoning()
        rels = cr._identify_relationships("rain causes wet ground")
        assert isinstance(rels, list)


class TestSyllogismReasoning:
    def test_init(self):
        sr = SyllogismReasoning()
        assert hasattr(sr, '_parse_premises')

    def test_parse_premises(self):
        sr = SyllogismReasoning()
        premises = sr._parse_premises("All men are mortal.\nSocrates is a man.")
        assert len(premises) == 2

    def test_identify_figure(self):
        sr = SyllogismReasoning()
        premises = ["All men are mortal", "Socrates is a man"]
        fig = sr._identify_figure(premises)
        assert isinstance(fig, int)
        assert 1 <= fig <= 4
