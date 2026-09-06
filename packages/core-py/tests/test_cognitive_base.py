"""Tests for cognitive.base — CognitiveDomain, CognitiveException."""

from __future__ import annotations

import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from domains.cognitive.base import CognitiveDomain, CognitiveException
from domains import DomainException


# ── CognitiveException ────────────────────────────────────────────────────


class TestCognitiveException:

    def test_is_domain_exception(self):
        assert issubclass(CognitiveException, DomainException)

    def test_message(self):
        exc = CognitiveException("test error")
        assert str(exc) == "test error"


# ── CognitiveDomain ───────────────────────────────────────────────────────


class TestCognitiveDomain:

    def test_init(self):
        domain = CognitiveDomain()
        assert domain.domain_name == "cognitive"
        assert domain.cognitive_state == "idle"
        assert domain.active_thoughts == []
        assert domain.is_initialized is False

    def test_init_components_none(self):
        domain = CognitiveDomain()
        assert domain.memory_manager is None
        assert domain.reasoning_engine is None
        assert domain.metacognitive_monitor is None
        assert domain.cognitive_processor is None

    @pytest.mark.asyncio
    async def test_get_cognitive_state(self):
        domain = CognitiveDomain()
        state = await domain.get_cognitive_state()
        assert state["state"] == "idle"
        assert state["active_thoughts_count"] == 0
        assert "memory_manager" in state["components_status"]
        assert "reasoning_engine" in state["components_status"]

    @pytest.mark.asyncio
    async def test_process_thought_no_processor(self):
        domain = CognitiveDomain()
        with pytest.raises(CognitiveException, match="not initialized"):
            await domain.process_thought("test thought")

    @pytest.mark.asyncio
    async def test_store_memory_no_manager(self):
        domain = CognitiveDomain()
        with pytest.raises(CognitiveException, match="not initialized"):
            await domain.store_memory("test content")

    @pytest.mark.asyncio
    async def test_retrieve_memory_no_manager(self):
        domain = CognitiveDomain()
        with pytest.raises(CognitiveException, match="not initialized"):
            await domain.retrieve_memory("some-id")

    @pytest.mark.asyncio
    async def test_reason_no_engine(self):
        domain = CognitiveDomain()
        with pytest.raises(CognitiveException, match="not initialized"):
            await domain.reason("premise", {})

    @pytest.mark.asyncio
    async def test_stop_cognitive_processes(self):
        domain = CognitiveDomain()
        task = asyncio.create_task(asyncio.sleep(999))
        domain._background_tasks = [task]
        await domain._stop_cognitive_processes()
        assert task.cancelled()
        assert domain._background_tasks == []

    @pytest.mark.asyncio
    async def test_shutdown_component_with_shutdown(self):
        domain = CognitiveDomain()
        mock_comp = AsyncMock()
        domain.memory_manager = mock_comp
        await domain._shutdown_component("memory_manager")
        mock_comp.shutdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_component_without_shutdown(self):
        domain = CognitiveDomain()
        domain.memory_manager = MagicMock(spec=[])
        await domain._shutdown_component("memory_manager")

    @pytest.mark.asyncio
    async def test_shutdown_component_error(self):
        domain = CognitiveDomain()
        mock_comp = AsyncMock()
        mock_comp.shutdown.side_effect = RuntimeError("boom")
        domain.memory_manager = mock_comp
        await domain._shutdown_component("memory_manager")
