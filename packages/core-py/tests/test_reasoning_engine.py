"""Tests for ReasoningEngine."""

import pytest

from domains.cognitive.reasoning import (
    ReasoningEngine,
    ReasoningMode,
    ReasoningResult,
    DeepReasoning,
    FormalLogicEngine,
    WorkingMemory,
)


class TestReasoningEngine:
    def test_initial_state(self):
        re = ReasoningEngine()
        assert re.mode == ReasoningMode.CHAIN_OF_THOUGHT
        assert re.reasoning_history == []
        assert re.deep_reasoning is not None
        assert re.logic_engine is not None
        assert re.working_memory is not None

    async def test_reason_returns_string(self):
        re = ReasoningEngine()
        result = await re.reason("test premise", {"context": "val"})
        assert isinstance(result, str)
        assert len(result) > 0

    async def test_reason_appends_to_history(self):
        re = ReasoningEngine()
        await re.reason("first", {})
        await re.reason("second", {})
        assert len(re.reasoning_history) == 2

    async def test_reason_history_contains_results(self):
        re = ReasoningEngine()
        await re.reason("test", {})
        assert isinstance(re.reasoning_history[0], ReasoningResult)

    async def test_deep_reason(self):
        re = ReasoningEngine()
        result = await re.deep_reason("deep problem", max_depth=1)
        assert isinstance(result, ReasoningResult)
        assert len(result.conclusion) > 0

    async def test_deep_reason_has_steps(self):
        re = ReasoningEngine()
        result = await re.deep_reason("test", max_depth=2)
        assert isinstance(result, ReasoningResult)
        assert len(result.steps) > 0

    async def test_logical_proof_valid(self):
        re = ReasoningEngine()
        result = await re.logical_proof(
            ("All", "humans", "mortal"),
            ("All", "socrates", "human"),
            ("All", "socrates", "mortal"),
        )
        assert isinstance(result, dict)
        assert "valid" in result

    async def test_logical_proof_returns_figure(self):
        re = ReasoningEngine()
        result = await re.logical_proof(
            ("All", "humans", "mortal"),
            ("All", "socrates", "human"),
            ("All", "socrates", "mortal"),
        )
        assert "figure" in result
        assert isinstance(result["figure"], int)

    async def test_set_mode(self):
        re = ReasoningEngine()
        await re.set_mode(ReasoningMode.TREE_OF_THOUGHTS)
        assert re.mode == ReasoningMode.TREE_OF_THOUGHTS

    async def test_set_mode_tree_of_thoughts(self):
        re = ReasoningEngine()
        await re.set_mode(ReasoningMode.TREE_OF_THOUGHTS)
        assert re.mode == ReasoningMode.TREE_OF_THOUGHTS

    async def test_get_history_empty(self):
        re = ReasoningEngine()
        history = await re.get_history()
        assert history == []

    async def test_get_history_after_reason(self):
        re = ReasoningEngine()
        await re.reason("test", {})
        history = await re.get_history()
        assert len(history) == 1

    def test_assert_fact_and_query(self):
        re = ReasoningEngine()
        re.assert_fact("mortal", "socrates")
        result = re.query("mortal", "socrates")
        assert result is True

    def test_query_false(self):
        re = ReasoningEngine()
        re.assert_fact("mortal", "socrates")
        result = re.query("mortal", "plato")
        assert result is False

    def test_set_mode_immediate(self):
        re = ReasoningEngine()
        re.mode = ReasoningMode.CHAIN_OF_THOUGHT
        assert re.mode == ReasoningMode.CHAIN_OF_THOUGHT
