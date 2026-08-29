"""Tests for domains.cognitive.reasoning.advanced — dataclasses + sync methods."""

import pytest
from domains.cognitive.reasoning.advanced import (
    ReasoningMode, ThoughtStep, ReasoningResult, ChainOfThought,
)


class TestReasoningMode:
    def test_all_members(self):
        assert len(ReasoningMode) == 8
    def test_values(self):
        assert ReasoningMode.CHAIN_OF_THOUGHT.value == "chain_of_thought"
        assert ReasoningMode.REACT.value == "react"


class TestThoughtStep:
    def test_fields(self):
        step = ThoughtStep(
            step_id=1,
            thought="First, we analyze...",
            reasoning_type="deductive",
            confidence=0.8,
        )
        assert step.step_id == 1
        assert step.parent_id is None
        assert step.children_ids == []

    def test_with_children(self):
        step = ThoughtStep(
            step_id=0, thought="root", reasoning_type="inductive",
            confidence=0.9, children_ids=[1, 2],
        )
        assert len(step.children_ids) == 2


class TestReasoningResult:
    def test_fields(self):
        result = ReasoningResult(
            conclusion="The answer is 42",
            confidence=0.85,
            mode=ReasoningMode.CHAIN_OF_THOUGHT,
            steps=[],
            metadata={"source": "test"},
            execution_time_ms=100.0,
        )
        assert result.conclusion == "The answer is 42"
        assert result.mode == ReasoningMode.CHAIN_OF_THOUGHT


class TestChainOfThought:
    def test_init(self):
        cot = ChainOfThought()
        assert cot.llm_call is not None

    def test_evaluate_confidence_high(self):
        cot = ChainOfThought()
        c = cot._evaluate_confidence("Therefore, we conclude X.")
        assert c >= 0.7

    def test_evaluate_confidence_low(self):
        cot = ChainOfThought()
        c = cot._evaluate_confidence("maybe")
        assert c <= 0.6

    def test_extract_subproblem_remaining(self):
        cot = ChainOfThought()
        result = cot._extract_subproblem("remaining: solve for X")
        assert result == "solve for X"

    def test_extract_subproblem_next(self):
        cot = ChainOfThought()
        result = cot._extract_subproblem("next: compute gradient")
        assert result == "compute gradient"

    def test_extract_subproblem_none(self):
        cot = ChainOfThought()
        result = cot._extract_subproblem("just some text")
        assert result is None

    def test_extract_conclusion_therefore(self):
        cot = ChainOfThought()
        result = cot._extract_conclusion("Therefore the answer is yes.")
        assert "yes" in result.lower()

    def test_extract_conclusion_answer(self):
        cot = ChainOfThought()
        result = cot._extract_conclusion("Answer: 42 is correct.")
        assert "42" in result

    def test_extract_conclusion_fallback(self):
        cot = ChainOfThought()
        result = cot._extract_conclusion("some random text without conclusion markers")
        assert "some random text" in result
