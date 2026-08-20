"""Tests for domains.cognitive.reasoning.__init__ — ReasoningEngine."""

import asyncio
import pytest
from domains.cognitive.reasoning import (
    ReasoningEngine, ReasoningMode, WorkingMemory,
)


class TestReasoningEngine:
    def test_init(self):
        engine = ReasoningEngine()
        assert engine.mode == ReasoningMode.CHAIN_OF_THOUGHT
        assert engine.reasoning_history == []

    def test_assert_and_query(self):
        engine = ReasoningEngine()
        engine.assert_fact("human", "socrates")
        assert engine.query("human", "socrates") is True

    def test_query_not_asserted(self):
        engine = ReasoningEngine()
        assert engine.query("human", "socrates") is False

    def test_set_mode(self):
        engine = ReasoningEngine()
        asyncio.run(engine.set_mode(ReasoningMode.TREE_OF_THOUGHTS))
        assert engine.mode == ReasoningMode.TREE_OF_THOUGHTS

    def test_get_history_empty(self):
        engine = ReasoningEngine()
        history = asyncio.run(engine.get_history())
        assert history == []

    def test_reason(self):
        engine = ReasoningEngine()
        result = asyncio.run(engine.reason("What is 2+2?", {}))
        assert isinstance(result, str)
        assert len(engine.reasoning_history) == 1


class TestWorkingMemory:
    def test_init(self):
        wm = WorkingMemory()
        assert wm is not None
