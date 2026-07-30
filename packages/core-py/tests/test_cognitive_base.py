"""Tests for CognitiveDomain."""

from __future__ import annotations

import asyncio
import pytest
from domains import Memory, Thought, ThoughtType
from domains.cognitive.base import CognitiveDomain, CognitiveException


class TestCognitiveDomain:
    async def test_initial_state(self):
        cd = CognitiveDomain()
        assert cd.domain_name == "cognitive"
        assert cd.cognitive_state == "idle"
        assert cd.active_thoughts == []
        assert cd.memory_store == {}
        assert cd.memory_manager is None
        assert cd.reasoning_engine is None
        assert cd.metacognitive_monitor is None

    async def test_get_cognitive_state_before_init(self):
        cd = CognitiveDomain()
        state = await cd.get_cognitive_state()
        assert state["state"] == "idle"
        assert state["active_thoughts_count"] == 0
        assert state["components_status"]["reasoning_engine"] == "not_initialized"

    async def test_initialize_sets_up_components(self):
        cd = CognitiveDomain()
        await cd._on_initialize()
        assert cd.reasoning_engine is not None
        assert cd.metacognitive_monitor is not None
        assert cd.cognitive_processor is not None
        assert len(cd._background_tasks) == 3
        assert cd.is_initialized is True
        await cd._on_shutdown()

    async def test_shutdown_clears_tasks(self):
        cd = CognitiveDomain()
        await cd._on_initialize()
        await cd._on_shutdown()
        assert cd.is_initialized is False
        assert len(cd._background_tasks) == 0

    async def test_get_cognitive_state_after_init(self):
        cd = CognitiveDomain()
        await cd._on_initialize()
        state = await cd.get_cognitive_state()
        assert state["components_status"]["reasoning_engine"] == "initialized"
        await cd._on_shutdown()

    async def test_process_thought(self):
        cd = CognitiveDomain()
        await cd._on_initialize()
        result = await cd.process_thought("solve x+2=5", "analytical")
        assert "original_thought" in result
        assert "processed_thought" in result
        assert "confidence" in result
        assert len(cd.active_thoughts) == 1
        await cd._on_shutdown()

    async def test_process_thought_without_processor(self):
        cd = CognitiveDomain()
        with pytest.raises(CognitiveException, match="Cognitive processor not initialized"):
            await cd.process_thought("test")

    async def test_reason(self):
        cd = CognitiveDomain()
        await cd._on_initialize()
        result = await cd.reason("test premise", {"context": "val"})
        assert isinstance(result, str)
        await cd._on_shutdown()

    async def test_reason_without_engine(self):
        cd = CognitiveDomain()
        with pytest.raises(CognitiveException, match="Reasoning engine not initialized"):
            await cd.reason("test", {})

    async def test_store_memory_without_manager(self):
        cd = CognitiveDomain()
        with pytest.raises(CognitiveException, match="Memory manager not initialized"):
            await cd.store_memory("content")

    async def test_retrieve_memory_without_manager(self):
        cd = CognitiveDomain()
        with pytest.raises(CognitiveException, match="Memory manager not initialized"):
            await cd.retrieve_memory("some_id")

    async def test_shutdown_component_nonexistent(self):
        cd = CognitiveDomain()
        await cd._shutdown_component("nonexistent")

    async def test_shutdown_component_none(self):
        cd = CognitiveDomain()
        cd.memory_manager = None
        await cd._shutdown_component("memory_manager")

    async def test_background_tasks_are_tracked(self):
        cd = CognitiveDomain()
        await cd._on_initialize()
        assert len(cd._background_tasks) == 3
        assert all(
            t._coro.__name__ in ("_memory_consolidation_loop", "_metacognitive_monitoring_loop", "_reasoning_optimization_loop")
            for t in cd._background_tasks
        )
        await cd._on_shutdown()

    async def test_double_shutdown_safe(self):
        cd = CognitiveDomain()
        await cd._on_initialize()
        await cd._on_shutdown()
        await cd._on_shutdown()

    async def test_get_cognitive_state_memory_store(self):
        cd = CognitiveDomain()
        cd.memory_store = {"k1": "v1"}
        state = await cd.get_cognitive_state()
        assert state["memory_count"] == 1


class TestCognitiveException:
    def test_is_domain_exception(self):
        from domains import DomainException
        assert issubclass(CognitiveException, DomainException)

    def test_can_be_raised(self):
        with pytest.raises(CognitiveException):
            raise CognitiveException("test error")

    def test_message_preserved(self):
        try:
            raise CognitiveException("custom message")
        except CognitiveException as e:
            assert "custom message" in str(e)
