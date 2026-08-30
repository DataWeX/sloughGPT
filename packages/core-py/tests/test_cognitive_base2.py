"""Tests for domains.cognitive.base — CognitiveDomain and CognitiveException."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from domains import DomainException, Thought, ThoughtType, Memory
from domains.cognitive.base import CognitiveDomain, CognitiveException


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------

class TestCognitiveException:
    def test_is_subclass_of_domain_exception(self):
        assert issubclass(CognitiveException, DomainException)

    def test_is_exception(self):
        assert issubclass(CognitiveException, Exception)

    def test_message_round_trip(self):
        exc = CognitiveException("boom")
        assert str(exc) == "boom"


# ---------------------------------------------------------------------------
# CognitiveDomain — construction defaults
# ---------------------------------------------------------------------------

class TestCognitiveDomainDefaults:
    def setup_method(self):
        self.domain = CognitiveDomain()

    def test_domain_name(self):
        assert self.domain.domain_name == "cognitive"

    def test_has_domain_name(self):
        assert hasattr(self.domain, "domain_name")
        assert self.domain.domain_name == "cognitive"

    def test_has_initialize_shutdown_interface(self):
        assert hasattr(self.domain, "_on_initialize")
        assert hasattr(self.domain, "_on_shutdown")

    def test_components_empty(self):
        assert self.domain.components == {}

    def test_is_initialized_false(self):
        assert self.domain.is_initialized is False

    def test_cognitive_state_idle(self):
        assert self.domain.cognitive_state == "idle"

    def test_active_thoughts_empty(self):
        assert self.domain.active_thoughts == []

    def test_memory_store_empty(self):
        assert self.domain.memory_store == {}

    def test_core_components_none(self):
        assert self.domain.memory_manager is None
        assert self.domain.reasoning_engine is None
        assert self.domain.metacognitive_monitor is None
        assert self.domain.cognitive_processor is None

    def test_background_tasks_empty(self):
        assert self.domain._background_tasks == []

    def test_logger_name(self):
        assert self.domain.logger.name == "slo.cognitive"


# ---------------------------------------------------------------------------
# get_cognitive_state
# ---------------------------------------------------------------------------

class TestGetCognitiveState:
    def setup_method(self):
        self.domain = CognitiveDomain()

    @pytest.mark.asyncio
    async def test_returns_dict(self):
        state = await self.domain.get_cognitive_state()
        assert isinstance(state, dict)

    @pytest.mark.asyncio
    async def test_keys(self):
        state = await self.domain.get_cognitive_state()
        assert set(state.keys()) == {"state", "active_thoughts_count", "memory_count", "components_status"}

    @pytest.mark.asyncio
    async def test_idle_state(self):
        state = await self.domain.get_cognitive_state()
        assert state["state"] == "idle"

    @pytest.mark.asyncio
    async def test_active_thoughts_count_zero(self):
        state = await self.domain.get_cognitive_state()
        assert state["active_thoughts_count"] == 0

    @pytest.mark.asyncio
    async def test_memory_count_zero(self):
        state = await self.domain.get_cognitive_state()
        assert state["memory_count"] == 0

    @pytest.mark.asyncio
    async def test_all_components_not_initialized(self):
        state = await self.domain.get_cognitive_state()
        for status in state["components_status"].values():
            assert status == "not_initialized"

    @pytest.mark.asyncio
    async def test_active_thoughts_reflects_list(self):
        self.domain.active_thoughts = [1, 2, 3]
        state = await self.domain.get_cognitive_state()
        assert state["active_thoughts_count"] == 3

    @pytest.mark.asyncio
    async def test_memory_store_reflects_dict(self):
        self.domain.memory_store = {"a": 1, "b": 2}
        state = await self.domain.get_cognitive_state()
        assert state["memory_count"] == 2

    @pytest.mark.asyncio
    async def test_components_status_with_mock_objects(self):
        self.domain.memory_manager = MagicMock()
        self.domain.reasoning_engine = None
        state = await self.domain.get_cognitive_state()
        assert state["components_status"]["memory_manager"] == "initialized"
        assert state["components_status"]["reasoning_engine"] == "not_initialized"


# ---------------------------------------------------------------------------
# process_thought — processor not initialized
# ---------------------------------------------------------------------------

class TestProcessThoughtNoProcessor:
    @pytest.mark.asyncio
    async def test_raises_cognitive_exception(self):
        domain = CognitiveDomain()
        with pytest.raises(CognitiveException, match="Cognitive processor not initialized"):
            await domain.process_thought("hello")


# ---------------------------------------------------------------------------
# store_memory — memory manager not initialized
# ---------------------------------------------------------------------------

class TestStoreMemoryNoManager:
    @pytest.mark.asyncio
    async def test_raises_cognitive_exception(self):
        domain = CognitiveDomain()
        with pytest.raises(CognitiveException, match="Memory manager not initialized"):
            await domain.store_memory("some content")


# ---------------------------------------------------------------------------
# retrieve_memory — memory manager not initialized
# ---------------------------------------------------------------------------

class TestRetrieveMemoryNoManager:
    @pytest.mark.asyncio
    async def test_raises_cognitive_exception(self):
        domain = CognitiveDomain()
        with pytest.raises(CognitiveException, match="Memory manager not initialized"):
            await domain.retrieve_memory("mem_123")


# ---------------------------------------------------------------------------
# reason — reasoning engine not initialized
# ---------------------------------------------------------------------------

class TestReasonNoEngine:
    @pytest.mark.asyncio
    async def test_raises_cognitive_exception(self):
        domain = CognitiveDomain()
        with pytest.raises(CognitiveException, match="Reasoning engine not initialized"):
            await domain.reason("premise", {})


# ---------------------------------------------------------------------------
# _shutdown_component
# ---------------------------------------------------------------------------

class TestShutdownComponent:
    @pytest.mark.asyncio
    async def test_calls_shutdown_if_present(self):
        domain = CognitiveDomain()
        comp = AsyncMock()
        domain.cognitive_processor = comp
        await domain._shutdown_component("cognitive_processor")
        comp.shutdown.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_shutdown_method_is_noop(self):
        domain = CognitiveDomain()
        comp = MagicMock(spec=[])  # no shutdown attr
        domain.cognitive_processor = comp
        await domain._shutdown_component("cognitive_processor")

    @pytest.mark.asyncio
    async def test_shutdown_exception_is_swallowed(self):
        domain = CognitiveDomain()
        comp = AsyncMock()
        comp.shutdown.side_effect = RuntimeError("fail")
        domain.cognitive_processor = comp
        # should not raise
        await domain._shutdown_component("cognitive_processor")

    @pytest.mark.asyncio
    async def test_missing_attribute_is_handled(self):
        domain = CognitiveDomain()
        # nonexistent attribute
        await domain._shutdown_component("nonexistent_component")


# ---------------------------------------------------------------------------
# _stop_cognitive_processes
# ---------------------------------------------------------------------------

class TestStopCognitiveProcesses:
    @pytest.mark.asyncio
    async def test_cancels_pending_tasks(self):
        domain = CognitiveDomain()

        async def slow():
            try:
                await asyncio.sleep(100)
            except asyncio.CancelledError:
                return

        task = asyncio.create_task(slow())
        domain._background_tasks = [task]
        await domain._stop_cognitive_processes()
        assert task.cancelled()
        assert domain._background_tasks == []

    @pytest.mark.asyncio
    async def test_clears_empty_list(self):
        domain = CognitiveDomain()
        domain._background_tasks = []
        await domain._stop_cognitive_processes()
        assert domain._background_tasks == []

    @pytest.mark.asyncio
    async def test_handles_already_done_tasks(self):
        domain = CognitiveDomain()

        async def done_soon():
            return 42

        task = asyncio.create_task(done_soon())
        await task  # let it finish
        domain._background_tasks = [task]
        await domain._stop_cognitive_processes()
        assert domain._background_tasks == []


# ---------------------------------------------------------------------------
# _on_shutdown — component order
# ---------------------------------------------------------------------------

class TestOnShutdown:
    @pytest.mark.asyncio
    async def test_shutdown_calls_components_in_order(self):
        domain = CognitiveDomain()
        order = []
        for name in ("cognitive_processor", "metacognitive_monitor", "reasoning_engine", "memory_manager"):
            comp = AsyncMock()
            comp.shutdown.side_effect = lambda n=name: order.append(n)
            setattr(domain, name, comp)

        await domain._on_shutdown()
        assert order == ["cognitive_processor", "metacognitive_monitor", "reasoning_engine", "memory_manager"]
        assert domain.is_initialized is False

    @pytest.mark.asyncio
    async def test_shutdown_handles_none_components(self):
        domain = CognitiveDomain()
        # all components are None — should not raise
        await domain._on_shutdown()
        assert domain.is_initialized is False
