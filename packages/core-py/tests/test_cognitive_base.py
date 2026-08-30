"""Comprehensive tests for CognitiveDomain base class.

Covers: initialization, shutdown, state management, background tasks,
process_thought, reason, store/retrieve memory, cognitive state,
and CognitiveException hierarchy.
"""

from __future__ import annotations

import asyncio
import pytest
from domains import Memory, Thought, ThoughtType, DomainException
from domains.cognitive.base import CognitiveDomain, CognitiveException


class TestCognitiveDomainInitialState:
    async def test_initial_state(self):
        cd = CognitiveDomain()
        assert cd.domain_name == "cognitive"
        assert cd.cognitive_state == "idle"
        assert cd.active_thoughts == []
        assert cd.memory_store == {}
        assert cd.memory_manager is None
        assert cd.reasoning_engine is None
        assert cd.metacognitive_monitor is None

    async def test_initial_components_empty(self):
        cd = CognitiveDomain()
        assert cd.components == {}

    async def test_initial_background_tasks_empty(self):
        cd = CognitiveDomain()
        assert cd._background_tasks == []

    async def test_initial_is_not_initialized(self):
        cd = CognitiveDomain()
        assert cd.is_initialized is False

    async def test_cognitive_processor_initial_none(self):
        cd = CognitiveDomain()
        assert cd.cognitive_processor is None


class TestCognitiveDomainGetState:
    async def test_get_cognitive_state_before_init(self):
        cd = CognitiveDomain()
        state = await cd.get_cognitive_state()
        assert state["state"] == "idle"
        assert state["active_thoughts_count"] == 0
        assert state["components_status"]["reasoning_engine"] == "not_initialized"

    async def test_get_cognitive_state_after_init(self):
        cd = CognitiveDomain()
        await cd._on_initialize()
        state = await cd.get_cognitive_state()
        assert state["components_status"]["reasoning_engine"] == "initialized"
        await cd._on_shutdown()

    async def test_get_cognitive_state_memory_store(self):
        cd = CognitiveDomain()
        cd.memory_store = {"k1": "v1"}
        state = await cd.get_cognitive_state()
        assert state["memory_count"] == 1

    async def test_get_cognitive_state_empty_memory_store(self):
        cd = CognitiveDomain()
        state = await cd.get_cognitive_state()
        assert state["memory_count"] == 0

    async def test_get_cognitive_state_all_components(self):
        cd = CognitiveDomain()
        state = await cd.get_cognitive_state()
        for name in ["memory_manager", "reasoning_engine",
                      "metacognitive_monitor", "cognitive_processor"]:
            assert name in state["components_status"]

    async def test_get_cognitive_state_memory_manager_not_initialized(self):
        cd = CognitiveDomain()
        state = await cd.get_cognitive_state()
        assert state["components_status"]["memory_manager"] == "not_initialized"


class TestCognitiveDomainInitialize:
    async def test_initialize_sets_up_components(self):
        cd = CognitiveDomain()
        await cd._on_initialize()
        assert cd.reasoning_engine is not None
        assert cd.metacognitive_monitor is not None
        assert cd.cognitive_processor is not None
        assert len(cd._background_tasks) == 3
        assert cd.is_initialized is True
        await cd._on_shutdown()

    async def test_initialize_sets_components_dict(self):
        cd = CognitiveDomain()
        await cd._on_initialize()
        assert "reasoning_engine" in cd.components
        assert "metacognitive_monitor" in cd.components
        assert "cognitive_processor" in cd.components
        await cd._on_shutdown()

    async def test_initialize_background_task_names(self):
        cd = CognitiveDomain()
        await cd._on_initialize()
        task_names = {
            t._coro.__name__ for t in cd._background_tasks
        }
        assert "_memory_consolidation_loop" in task_names
        assert "_metacognitive_monitoring_loop" in task_names
        assert "_reasoning_optimization_loop" in task_names
        await cd._on_shutdown()

    async def test_shutdown_clears_tasks(self):
        cd = CognitiveDomain()
        await cd._on_initialize()
        await cd._on_shutdown()
        assert cd.is_initialized is False
        assert len(cd._background_tasks) == 0


class TestCognitiveDomainShutdown:
    async def test_shutdown_clears_tasks(self):
        cd = CognitiveDomain()
        await cd._on_initialize()
        await cd._on_shutdown()
        assert cd.is_initialized is False
        assert len(cd._background_tasks) == 0

    async def test_double_shutdown_safe(self):
        cd = CognitiveDomain()
        await cd._on_initialize()
        await cd._on_shutdown()
        await cd._on_shutdown()

    async def test_shutdown_without_init(self):
        cd = CognitiveDomain()
        # No components to shut down
        await cd._on_shutdown()


class TestShutdownComponent:
    async def test_shutdown_component_nonexistent(self):
        cd = CognitiveDomain()
        await cd._shutdown_component("nonexistent")

    async def test_shutdown_component_none(self):
        cd = CognitiveDomain()
        cd.memory_manager = None
        await cd._shutdown_component("memory_manager")

    async def test_shutdown_component_with_shutdown_method(self):
        cd = CognitiveDomain()
        mock_component = type("MockComponent", (), {"shutdown": lambda s: asyncio.sleep(0)})()
        cd.memory_manager = mock_component
        await cd._shutdown_component("memory_manager")

    async def test_shutdown_component_raises_exception(self):
        cd = CognitiveDomain()
        bad = type("Bad", (), {"shutdown": lambda s: 1 / 0})()
        cd.memory_manager = bad
        # Should not raise
        await cd._shutdown_component("memory_manager")


class TestProcessThought:
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

    async def test_process_thought_multiple(self):
        cd = CognitiveDomain()
        await cd._on_initialize()
        await cd.process_thought("first", "analytical")
        await cd.process_thought("second", "creative")
        assert len(cd.active_thoughts) == 2
        await cd._on_shutdown()

    async def test_process_thought_returns_original(self):
        cd = CognitiveDomain()
        await cd._on_initialize()
        result = await cd.process_thought("my question", "analytical")
        assert result["original_thought"] == "my question"
        await cd._on_shutdown()

    async def test_process_thought_has_reasoning_path(self):
        cd = CognitiveDomain()
        await cd._on_initialize()
        result = await cd.process_thought("test", "analytical")
        assert "reasoning_path" in result
        await cd._on_shutdown()


class TestReason:
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

    async def test_reason_returns_string(self):
        cd = CognitiveDomain()
        await cd._on_initialize()
        result = await cd.reason("any premise", {})
        assert isinstance(result, str)
        assert len(result) > 0
        await cd._on_shutdown()


class TestMemoryOperations:
    async def test_store_memory_without_manager(self):
        cd = CognitiveDomain()
        with pytest.raises(CognitiveException, match="Memory manager not initialized"):
            await cd.store_memory("content")

    async def test_retrieve_memory_without_manager(self):
        cd = CognitiveDomain()
        with pytest.raises(CognitiveException, match="Memory manager not initialized"):
            await cd.retrieve_memory("some_id")


class TestBackgroundTasks:
    async def test_background_tasks_are_tracked(self):
        cd = CognitiveDomain()
        await cd._on_initialize()
        assert len(cd._background_tasks) == 3
        assert all(
            t._coro.__name__ in ("_memory_consolidation_loop",
                                  "_metacognitive_monitoring_loop",
                                  "_reasoning_optimization_loop")
            for t in cd._background_tasks
        )
        await cd._on_shutdown()

    async def test_background_tasks_are_asyncio_tasks(self):
        cd = CognitiveDomain()
        await cd._on_initialize()
        for t in cd._background_tasks:
            assert isinstance(t, asyncio.Task)
        await cd._on_shutdown()


class TestCognitiveException:
    def test_is_domain_exception(self):
        assert issubclass(CognitiveException, DomainException)

    def test_can_be_raised(self):
        with pytest.raises(CognitiveException):
            raise CognitiveException("test error")

    def test_message_preserved(self):
        try:
            raise CognitiveException("custom message")
        except CognitiveException as e:
            assert "custom message" in str(e)

    def test_is_exception(self):
        assert issubclass(CognitiveException, Exception)

    def test_catch_as_domain_exception(self):
        with pytest.raises(DomainException):
            raise CognitiveException("cognitive error")


class TestMemoryClass:
    def test_memory_creation(self):
        m = Memory(key="k1", value="v1")
        assert m.key == "k1"
        assert m.value == "v1"
        assert m.content == "v1"

    def test_memory_defaults(self):
        m = Memory(key="k", value="v")
        assert m.memory_type == "episodic"
        assert m.importance == 0.5
        assert m.retrieval_count == 0

    def test_memory_importance(self):
        m = Memory(key="k", value="v", memory_type="semantic", importance=0.9)
        assert m.memory_type == "semantic"
        assert m.importance == 0.9


class TestThoughtClass:
    def test_thought_creation(self):
        t = Thought(thought_id="t1", content="test thought")
        assert t.thought_id == "t1"
        assert t.content == "test thought"
        assert t.thought_type == "reasoning"
        assert t.confidence == 0.5

    def test_thought_custom_type(self):
        t = Thought(thought_id="t1", content="test",
                     thought_type="creativity", confidence=0.9)
        assert t.thought_type == "creativity"
        assert t.confidence == 0.9

    def test_thought_metadata(self):
        t = Thought(thought_id="t1", content="test",
                     metadata={"key": "val"})
        assert t.metadata == {"key": "val"}


class TestThoughtType:
    def test_thought_types_exist(self):
        assert ThoughtType.PERCEPTION == "perception"
        assert ThoughtType.REASONING == "reasoning"
        assert ThoughtType.CREATIVITY == "creativity"
        assert ThoughtType.REFLECTION == "reflection"
        assert ThoughtType.DECISION == "decision"


class TestBaseDomain:
    def test_base_domain_name(self):
        cd = CognitiveDomain()
        assert cd.domain_name == "cognitive"
