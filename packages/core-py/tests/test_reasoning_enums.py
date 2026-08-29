"""Tests for domains.cognitive.reasoning.deep — LogicalOperator; domains.cognitive.reasoning.advanced — ReasoningMode, ThoughtStep."""

from domains.cognitive.reasoning.deep import LogicalOperator
from domains.cognitive.reasoning.advanced import ReasoningMode, ThoughtStep


class TestLogicalOperator:
    def test_all_members(self):
        assert len(LogicalOperator) == 7
    def test_values(self):
        assert LogicalOperator.AND.value == "∧"
        assert LogicalOperator.OR.value == "∨"
        assert LogicalOperator.NOT.value == "¬"
        assert LogicalOperator.IMPLIES.value == "→"
        assert LogicalOperator.FORALL.value == "∀"
        assert LogicalOperator.EXISTS.value == "∃"


class TestReasoningMode:
    def test_all_members(self):
        assert len(ReasoningMode) == 8
    def test_values(self):
        assert ReasoningMode.CHAIN_OF_THOUGHT.value == "chain_of_thought"
        assert ReasoningMode.TREE_OF_THOUGHTS.value == "tree_of_thoughts"
        assert ReasoningMode.REACT.value == "react"


class TestThoughtStep:
    def test_fields(self):
        ts = ThoughtStep(step_id=1, thought="test", reasoning_type="deduction", confidence=0.9)
        assert ts.step_id == 1
        assert ts.thought == "test"
        assert ts.confidence == 0.9
        assert ts.parent_id is None
        assert ts.children_ids == []
