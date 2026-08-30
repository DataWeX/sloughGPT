"""Tests for CognitiveProcessor — pure logic, no external mocks."""

import asyncio
import time

import pytest

from domains.__init__ import BaseComponent, ComponentException, Memory, Thought, ThoughtType
from domains.cognitive.processor import CognitiveProcessor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _thought(
    content: str = "test thought",
    thought_type: str = "reasoning",
    confidence: float = 0.5,
    metadata: dict | None = None,
    thought_id: str = "t1",
) -> Thought:
    return Thought(
        thought_id=thought_id,
        content=content,
        thought_type=thought_type,
        confidence=confidence,
        metadata=metadata or {},
    )


def _make_memory(content: str = "mem", importance: float = 0.5, memory_type: str = "episodic") -> Memory:
    m = Memory(key="k", value=content, memory_type=memory_type, importance=importance)
    m.content = content
    return m


class StubMemoryManager:
    """In-memory stub — no network, no real DB."""

    def __init__(self) -> None:
        self.stored: list[dict] = []
        self.memories: list[Memory] = []
        self._initialized = False
        self._shutdown_called = False

    async def initialize(self) -> None:
        self._initialized = True

    async def shutdown(self) -> None:
        self._shutdown_called = True

    async def search_memories(self, query: str, limit: int = 5) -> list[Memory]:
        return self.memories[:limit]

    async def store_memory(self, data: dict, memory_type: str = "episodic", importance: float = 0.5) -> str:
        self.stored.append({"data": data, "memory_type": memory_type, "importance": importance})
        return f"mem_{len(self.stored)}"

    async def get_memory_statistics(self) -> dict:
        return {"total": len(self.stored)}


class StubReasoningEngine:
    def __init__(self, result: str = "reasoned") -> None:
        self.result = result
        self._initialized = False
        self._shutdown_called = False
        self.last_context: dict | None = None

    async def initialize(self) -> None:
        self._initialized = True

    async def shutdown(self) -> None:
        self._shutdown_called = True

    async def reason(self, content: str, context: dict) -> str:
        self.last_context = context
        return self.result

    async def get_reasoning_path(self) -> list[str]:
        return ["step1", "step2"]


class StubMetacognitiveMonitor:
    def __init__(self, confidence: float = 0.7, state: str = "stable") -> None:
        self.confidence = confidence
        self.state = state
        self._initialized = False
        self._shutdown_called = False
        self._monitoring_level: str | None = None
        self.monitored_thoughts: list[list[Thought]] = []

    async def initialize(self) -> None:
        self._initialized = True

    async def shutdown(self) -> None:
        self._shutdown_called = True

    async def assess_confidence(self, thought: Thought) -> float:
        return self.confidence

    async def monitor_thought_process(self, thoughts: list[Thought]) -> None:
        self.monitored_thoughts.append(thoughts)

    async def get_cognitive_state_snapshot(self) -> dict:
        return {"cognitive_load": self.state}

    async def set_monitoring_level(self, level: str) -> None:
        self._monitoring_level = level

    async def get_metacognitive_report(self, window: str) -> dict:
        return {"window": window, "status": "ok"}


# ---------------------------------------------------------------------------
# Constructor & initialization
# ---------------------------------------------------------------------------

class TestCognitiveProcessorInit:
    def test_creates_with_defaults(self) -> None:
        proc = CognitiveProcessor()
        assert proc.component_name == "cognitive_processor"
        assert isinstance(proc, BaseComponent)
        assert proc.current_thoughts == []
        assert proc.is_processing is False
        assert proc.is_initialized is False
        assert proc.processing_stats["total_thoughts_processed"] == 0

    def test_creates_with_components(self) -> None:
        mem = StubMemoryManager()
        reason = StubReasoningEngine()
        mono = StubMetacognitiveMonitor()
        proc = CognitiveProcessor(memory_manager=mem, reasoning_engine=reason, metacognitive_monitor=mono)
        assert proc.memory_manager is mem
        assert proc.reasoning_engine is reason
        assert proc.metacognitive_monitor is mono

    @pytest.mark.asyncio
    async def test_initialize_sets_flag_and_delegates(self) -> None:
        mem = StubMemoryManager()
        reason = StubReasoningEngine()
        mono = StubMetacognitiveMonitor()
        proc = CognitiveProcessor(memory_manager=mem, reasoning_engine=reason, metacognitive_monitor=mono)

        await proc.initialize()
        assert proc.is_initialized is True
        assert mem._initialized is True
        assert reason._initialized is True
        assert mono._initialized is True

        await proc.shutdown()

    @pytest.mark.asyncio
    async def test_initialize_no_components(self) -> None:
        proc = CognitiveProcessor()
        await proc.initialize()
        assert proc.is_initialized is True
        await proc.shutdown()

    @pytest.mark.asyncio
    async def test_initialize_component_without_initialize_method(self) -> None:
        class Bare:
            pass

        proc = CognitiveProcessor(memory_manager=Bare())
        await proc.initialize()
        assert proc.is_initialized is True
        await proc.shutdown()


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------

class TestCognitiveProcessorShutdown:
    @pytest.mark.asyncio
    async def test_shutdown_clears_flag_and_cancels_task(self) -> None:
        proc = CognitiveProcessor()
        await proc.initialize()
        task = proc.processing_task
        assert task is not None

        await proc.shutdown()
        assert proc.is_initialized is False
        assert task.cancelled()

    @pytest.mark.asyncio
    async def test_shutdown_delegates_to_components(self) -> None:
        mem = StubMemoryManager()
        reason = StubReasoningEngine()
        mono = StubMetacognitiveMonitor()
        proc = CognitiveProcessor(memory_manager=mem, reasoning_engine=reason, metacognitive_monitor=mono)
        await proc.initialize()

        await proc.shutdown()
        assert mem._shutdown_called is True
        assert reason._shutdown_called is True
        assert mono._shutdown_called is True

    @pytest.mark.asyncio
    async def test_shutdown_component_without_shutdown_method(self) -> None:
        class Bare:
            pass

        proc = CognitiveProcessor(memory_manager=Bare())
        await proc.initialize()
        await proc.shutdown()
        assert proc.is_initialized is False


# ---------------------------------------------------------------------------
# process_thought — pipeline stages
# ---------------------------------------------------------------------------

class TestProcessThought:
    @pytest.mark.asyncio
    async def test_stores_thought_in_current_list(self) -> None:
        proc = CognitiveProcessor()
        await proc.initialize()
        t = _thought("hello")
        await proc.process_thought(t)
        assert t in proc.current_thoughts
        await proc.shutdown()

    @pytest.mark.asyncio
    async def test_with_memory_manager_retrieves_and_stores(self) -> None:
        mem = StubMemoryManager()
        mem.memories = [_make_memory("related")]
        proc = CognitiveProcessor(memory_manager=mem)
        await proc.initialize()

        t = _thought("hello")
        result = await proc.process_thought(t)

        assert "memory_context" in t.metadata
        assert t.metadata["memory_context"]["retrieval_count"] == 1
        assert len(mem.stored) == 1
        assert mem.stored[0]["data"]["thought_content"] == "hello"
        assert t.metadata["memory_id"] == "mem_1"
        await proc.shutdown()

    @pytest.mark.asyncio
    async def test_with_reasoning_engine_applies_reasoning(self) -> None:
        reason = StubReasoningEngine(result="deduced")
        proc = CognitiveProcessor(reasoning_engine=reason)
        await proc.initialize()

        t = _thought("think")
        await proc.process_thought(t)

        assert t.metadata["reasoning_result"]["reasoning_output"] == "deduced"
        assert t.metadata["reasoning_result"]["reasoning_path"] == ["step1", "step2"]
        assert "timestamp" in t.metadata["reasoning_result"]
        await proc.shutdown()

    @pytest.mark.asyncio
    async def test_with_metacognitive_monitor_sets_confidence(self) -> None:
        mono = StubMetacognitiveMonitor(confidence=0.92)
        proc = CognitiveProcessor(metacognitive_monitor=mono)
        await proc.initialize()

        t = _thought(confidence=0.1)
        await proc.process_thought(t)

        assert t.confidence == 0.92
        assert len(mono.monitored_thoughts) == 1
        await proc.shutdown()

    @pytest.mark.asyncio
    async def test_full_pipeline_all_components(self) -> None:
        mem = StubMemoryManager()
        mem.memories = [_make_memory("m1")]
        reason = StubReasoningEngine("r")
        mono = StubMetacognitiveMonitor(confidence=0.8)
        proc = CognitiveProcessor(memory_manager=mem, reasoning_engine=reason, metacognitive_monitor=mono)
        await proc.initialize()

        t = _thought("full")
        result = await proc.process_thought(t)

        assert result is t
        assert t.confidence == 0.8
        assert t.metadata["reasoning_result"]["reasoning_output"] == "r"
        assert t.metadata["memory_id"] == "mem_1"
        assert len(mono.monitored_thoughts) == 1
        await proc.shutdown()

    @pytest.mark.asyncio
    async def test_updates_stats_on_success(self) -> None:
        proc = CognitiveProcessor()
        await proc.initialize()
        t = _thought("stats")
        await proc.process_thought(t)

        assert proc.processing_stats["total_thoughts_processed"] == 1
        assert proc.processing_stats["successful_processes"] == 1
        assert proc.processing_stats["failed_processes"] == 0
        assert proc.processing_stats["average_processing_time"] >= 0
        await proc.shutdown()


# ---------------------------------------------------------------------------
# _calculate_thought_importance
# ---------------------------------------------------------------------------

class TestThoughtImportance:
    @pytest.mark.asyncio
    async def test_base_importance_is_confidence(self) -> None:
        proc = CognitiveProcessor()
        t = _thought(confidence=0.6)
        score = await proc._calculate_thought_importance(t)
        assert score == pytest.approx(0.6)

    @pytest.mark.asyncio
    async def test_analytical_boost(self) -> None:
        proc = CognitiveProcessor()
        t = _thought(confidence=0.5, thought_type="analytical")
        score = await proc._calculate_thought_importance(t)
        assert score == pytest.approx(0.6)

    @pytest.mark.asyncio
    async def test_creative_boost(self) -> None:
        proc = CognitiveProcessor()
        t = _thought(confidence=0.5, thought_type="creative")
        score = await proc._calculate_thought_importance(t)
        assert score == pytest.approx(0.7)

    @pytest.mark.asyncio
    async def test_metacognitive_boost(self) -> None:
        proc = CognitiveProcessor()
        t = _thought(confidence=0.5, thought_type="metacognitive")
        score = await proc._calculate_thought_importance(t)
        assert score == pytest.approx(0.8)

    @pytest.mark.asyncio
    async def test_intuitive_boost(self) -> None:
        proc = CognitiveProcessor()
        t = _thought(confidence=0.5, thought_type="intuitive")
        score = await proc._calculate_thought_importance(t)
        assert score == pytest.approx(0.55)

    @pytest.mark.asyncio
    async def test_unknown_type_no_boost(self) -> None:
        proc = CognitiveProcessor()
        t = _thought(confidence=0.4, thought_type="unknown")
        score = await proc._calculate_thought_importance(t)
        assert score == pytest.approx(0.4)

    @pytest.mark.asyncio
    async def test_reasoning_result_boost(self) -> None:
        proc = CognitiveProcessor()
        t = _thought(confidence=0.5, metadata={"reasoning_result": {"x": 1}})
        score = await proc._calculate_thought_importance(t)
        assert score == pytest.approx(0.6)

    @pytest.mark.asyncio
    async def test_all_boosts_combined(self) -> None:
        proc = CognitiveProcessor()
        t = _thought(confidence=0.5, thought_type="metacognitive", metadata={"reasoning_result": True})
        score = await proc._calculate_thought_importance(t)
        # 0.5 + 0.3 (metacognitive) + 0.1 (reasoning) = 0.9
        assert score == pytest.approx(0.9)

    @pytest.mark.asyncio
    async def test_clamped_to_max_1(self) -> None:
        proc = CognitiveProcessor()
        t = _thought(confidence=0.9, thought_type="metacognitive", metadata={"reasoning_result": True})
        score = await proc._calculate_thought_importance(t)
        # 0.9 + 0.3 + 0.1 = 1.3 → clamped to 1.0
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_clamped_to_min_0(self) -> None:
        proc = CognitiveProcessor()
        t = _thought(confidence=0.0)
        score = await proc._calculate_thought_importance(t)
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_no_reasoning_result_no_boost(self) -> None:
        proc = CognitiveProcessor()
        t = _thought(confidence=0.5, metadata={})
        score = await proc._calculate_thought_importance(t)
        assert score == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# _cleanup_old_thoughts
# ---------------------------------------------------------------------------

class TestCleanupOldThoughts:
    @pytest.mark.asyncio
    async def test_no_cleanup_under_limit(self) -> None:
        proc = CognitiveProcessor()
        proc.current_thoughts = [_thought(thought_id=str(i)) for i in range(50)]
        await proc._cleanup_old_thoughts()
        assert len(proc.current_thoughts) == 50

    @pytest.mark.asyncio
    async def test_trims_to_100(self) -> None:
        proc = CognitiveProcessor()
        proc.current_thoughts = [_thought(thought_id=str(i)) for i in range(150)]
        await proc._cleanup_old_thoughts()
        assert len(proc.current_thoughts) == 100
        assert proc.current_thoughts[0].thought_id == "50"

    @pytest.mark.asyncio
    async def test_exact_100_not_trimmed(self) -> None:
        proc = CognitiveProcessor()
        proc.current_thoughts = [_thought(thought_id=str(i)) for i in range(100)]
        await proc._cleanup_old_thoughts()
        assert len(proc.current_thoughts) == 100


# ---------------------------------------------------------------------------
# _update_processing_stats
# ---------------------------------------------------------------------------

class TestUpdateProcessingStats:
    def test_first_stat(self) -> None:
        proc = CognitiveProcessor()
        proc._update_processing_stats(0.5, True)
        s = proc.processing_stats
        assert s["total_thoughts_processed"] == 1
        assert s["successful_processes"] == 1
        assert s["failed_processes"] == 0
        assert s["average_processing_time"] == pytest.approx(0.5)

    def test_failed_stat(self) -> None:
        proc = CognitiveProcessor()
        proc._update_processing_stats(0.1, False)
        assert proc.processing_stats["failed_processes"] == 1
        assert proc.processing_stats["successful_processes"] == 0

    def test_average_computed_correctly(self) -> None:
        proc = CognitiveProcessor()
        proc._update_processing_stats(1.0, True)
        proc._update_processing_stats(3.0, True)
        # step 1: (0.0 * 0 + 1.0) / 1 = 1.0
        # step 2: (1.0 * 1 + 3.0) / 2 = 2.0
        assert proc.processing_stats["average_processing_time"] == pytest.approx(2.0)
        assert proc.processing_stats["total_thoughts_processed"] == 2


# ---------------------------------------------------------------------------
# get_cognitive_state
# ---------------------------------------------------------------------------

class TestGetCognitiveState:
    @pytest.mark.asyncio
    async def test_no_monitor_returns_unavailable(self) -> None:
        proc = CognitiveProcessor()
        state = await proc.get_cognitive_state()
        assert state == "monitoring_unavailable"

    @pytest.mark.asyncio
    async def test_monitor_returns_state(self) -> None:
        mono = StubMetacognitiveMonitor(state="high_load")
        proc = CognitiveProcessor(metacognitive_monitor=mono)
        state = await proc.get_cognitive_state()
        assert state == "high_load"

    @pytest.mark.asyncio
    async def test_monitor_returns_none_as_unknown(self) -> None:
        mono = StubMetacognitiveMonitor(state=None)
        proc = CognitiveProcessor(metacognitive_monitor=mono)
        state = await proc.get_cognitive_state()
        assert state == "unknown"

    @pytest.mark.asyncio
    async def test_monitor_exception_returns_error(self) -> None:
        class BadMonitor:
            async def get_cognitive_state_snapshot(self) -> dict:
                raise RuntimeError("boom")

        proc = CognitiveProcessor(metacognitive_monitor=BadMonitor())
        state = await proc.get_cognitive_state()
        assert state == "error"


# ---------------------------------------------------------------------------
# set_cognitive_state
# ---------------------------------------------------------------------------

class TestSetCognitiveState:
    @pytest.mark.asyncio
    async def test_focused_sets_strategic(self) -> None:
        mono = StubMetacognitiveMonitor()
        proc = CognitiveProcessor(metacognitive_monitor=mono)
        await proc.set_cognitive_state("focused")
        assert mono._monitoring_level == "strategic"

    @pytest.mark.asyncio
    async def test_creative_sets_reflective(self) -> None:
        mono = StubMetacognitiveMonitor()
        proc = CognitiveProcessor(metacognitive_monitor=mono)
        await proc.set_cognitive_state("creative")
        assert mono._monitoring_level == "reflective"

    @pytest.mark.asyncio
    async def test_analytical_sets_adaptive(self) -> None:
        mono = StubMetacognitiveMonitor()
        proc = CognitiveProcessor(metacognitive_monitor=mono)
        await proc.set_cognitive_state("analytical")
        assert mono._monitoring_level == "adaptive"

    @pytest.mark.asyncio
    async def test_unknown_state_no_monitoring_change(self) -> None:
        mono = StubMetacognitiveMonitor()
        proc = CognitiveProcessor(metacognitive_monitor=mono)
        await proc.set_cognitive_state("random")
        assert mono._monitoring_level is None

    @pytest.mark.asyncio
    async def test_no_monitor_does_not_raise(self) -> None:
        proc = CognitiveProcessor()
        await proc.set_cognitive_state("focused")

    @pytest.mark.asyncio
    async def test_exception_raises_component_exception(self) -> None:
        class BadMonitor:
            async def set_monitoring_level(self, level: str) -> None:
                raise RuntimeError("fail")

        proc = CognitiveProcessor(metacognitive_monitor=BadMonitor())
        with pytest.raises(ComponentException, match="Cognitive state setting failed"):
            await proc.set_cognitive_state("focused")


# ---------------------------------------------------------------------------
# _retrieve_relevant_memories
# ---------------------------------------------------------------------------

class TestRetrieveMemories:
    @pytest.mark.asyncio
    async def test_no_memory_manager_returns_empty(self) -> None:
        proc = CognitiveProcessor()
        result = await proc._retrieve_relevant_memories(_thought("x"))
        assert result == {}

    @pytest.mark.asyncio
    async def test_populates_thought_metadata(self) -> None:
        mem = StubMemoryManager()
        mem.memories = [_make_memory("m1", importance=0.8)]
        proc = CognitiveProcessor(memory_manager=mem)
        t = _thought("q")
        ctx = await proc._retrieve_relevant_memories(t)

        assert "memory_context" in t.metadata
        assert ctx["retrieval_count"] == 1
        assert ctx["relevant_memories"][0]["content"] == "m1"
        assert ctx["relevant_memories"][0]["importance"] == 0.8

    @pytest.mark.asyncio
    async def test_exception_returns_empty(self) -> None:
        class BadMem:
            async def search_memories(self, query: str, limit: int = 5) -> list:
                raise RuntimeError("fail")

        proc = CognitiveProcessor(memory_manager=BadMem())
        result = await proc._retrieve_relevant_memories(_thought("q"))
        assert result == {}


# ---------------------------------------------------------------------------
# _apply_reasoning
# ---------------------------------------------------------------------------

class TestApplyReasoning:
    @pytest.mark.asyncio
    async def test_no_reasoning_engine_returns_empty(self) -> None:
        proc = CognitiveProcessor()
        result = await proc._apply_reasoning(_thought("x"), {})
        assert result == {}

    @pytest.mark.asyncio
    async def test_passes_thought_type_string(self) -> None:
        reason = StubReasoningEngine()
        proc = CognitiveProcessor(reasoning_engine=reason)
        t = _thought("q", thought_type="analytical")
        await proc._apply_reasoning(t, {"some": "ctx"})

        assert reason.last_context["thought_type"] == "analytical"
        assert reason.last_context["memory_context"] == {"some": "ctx"}

    @pytest.mark.asyncio
    async def test_passes_enum_value_if_object(self) -> None:
        class FakeEnum:
            value = "creative"

        reason = StubReasoningEngine()
        proc = CognitiveProcessor(reasoning_engine=reason)
        t = _thought("q")
        t.thought_type = FakeEnum()
        await proc._apply_reasoning(t, {})

        assert reason.last_context["thought_type"] == "creative"

    @pytest.mark.asyncio
    async def test_exception_returns_empty(self) -> None:
        class BadReason:
            async def reason(self, content: str, context: dict) -> str:
                raise RuntimeError("fail")
            async def get_reasoning_path(self) -> list[str]:
                return []

        proc = CognitiveProcessor(reasoning_engine=BadReason())
        result = await proc._apply_reasoning(_thought("x"), {})
        assert result == {}


# ---------------------------------------------------------------------------
# _store_thought_memory
# ---------------------------------------------------------------------------

class TestStoreThoughtMemory:
    @pytest.mark.asyncio
    async def test_no_memory_manager_does_nothing(self) -> None:
        proc = CognitiveProcessor()
        t = _thought("x")
        await proc._store_thought_memory(t)
        assert "memory_id" not in t.metadata

    @pytest.mark.asyncio
    async def test_stores_with_correct_fields(self) -> None:
        mem = StubMemoryManager()
        proc = CognitiveProcessor(memory_manager=mem)
        t = _thought("content", thought_type="creative", confidence=0.9, metadata={"reasoning_result": "r1"})
        await proc._store_thought_memory(t)

        assert len(mem.stored) == 1
        stored = mem.stored[0]
        assert stored["data"]["thought_content"] == "content"
        assert stored["data"]["thought_type"] == "creative"
        assert stored["data"]["confidence"] == 0.9
        assert stored["memory_type"] == "episodic"
        assert t.metadata["memory_id"] == "mem_1"

    @pytest.mark.asyncio
    async def test_exception_does_not_propagate(self) -> None:
        class BadMem:
            async def store_memory(self, data: dict, memory_type: str = "episodic", importance: float = 0.5) -> str:
                raise RuntimeError("store fail")
            async def search_memories(self, query: str, limit: int = 5) -> list:
                return []

        proc = CognitiveProcessor(memory_manager=BadMem())
        t = _thought("x")
        await proc._store_thought_memory(t)  # should not raise
        assert "memory_id" not in t.metadata


# ---------------------------------------------------------------------------
# queue_thought_for_processing
# ---------------------------------------------------------------------------

class TestQueueThought:
    @pytest.mark.asyncio
    async def test_adds_to_queue(self) -> None:
        proc = CognitiveProcessor()
        t = _thought("queued")
        await proc.queue_thought_for_processing(t)
        assert proc.processing_queue.qsize() == 1


# ---------------------------------------------------------------------------
# get_processing_statistics
# ---------------------------------------------------------------------------

class TestGetProcessingStatistics:
    @pytest.mark.asyncio
    async def test_empty_stats(self) -> None:
        proc = CognitiveProcessor()
        stats = await proc.get_processing_statistics()
        assert stats["total_thoughts_processed"] == 0
        assert stats["success_rate"] == 0.0
        assert stats["queue_size"] == 0
        assert stats["current_thoughts_count"] == 0

    @pytest.mark.asyncio
    async def test_success_rate_after_processing(self) -> None:
        proc = CognitiveProcessor()
        await proc.initialize()

        await proc.process_thought(_thought("a"))
        await proc.process_thought(_thought("b"))

        stats = await proc.get_processing_statistics()
        assert stats["success_rate"] == 1.0
        assert stats["current_thoughts_count"] == 2
        await proc.shutdown()


# ---------------------------------------------------------------------------
# batch_process_thoughts
# ---------------------------------------------------------------------------

class TestBatchProcess:
    @pytest.mark.asyncio
    async def test_processes_all(self) -> None:
        proc = CognitiveProcessor()
        await proc.initialize()

        thoughts = [_thought(f"t{i}") for i in range(3)]
        results = await proc.batch_process_thoughts(thoughts)

        assert len(results) == 3
        for r in results:
            assert r in proc.current_thoughts
        await proc.shutdown()

    @pytest.mark.asyncio
    async def test_continues_on_failure(self) -> None:
        class FailOnceReason:
            def __init__(self) -> None:
                self.call_count = 0

            async def reason(self, content: str, context: dict) -> str:
                self.call_count += 1
                if self.call_count == 1:
                    raise RuntimeError("first fails")
                return "ok"

            async def get_reasoning_path(self) -> list[str]:
                return []

        reason = FailOnceReason()
        proc = CognitiveProcessor(reasoning_engine=reason)
        await proc.initialize()

        t1 = _thought("fail", thought_id="f1")
        t2 = _thought("ok", thought_id="f2")
        results = await proc.batch_process_thoughts([t1, t2])

        assert len(results) == 2
        # first one should be original (unprocessed) because it failed
        assert results[0] is t1
        # second one should be processed
        assert results[1] is t2
        await proc.shutdown()


# ---------------------------------------------------------------------------
# trigger_cognitive_assessment
# ---------------------------------------------------------------------------

class TestTriggerCognitiveAssessment:
    @pytest.mark.asyncio
    async def test_basic_assessment(self) -> None:
        proc = CognitiveProcessor()
        await proc.initialize()

        assessment = await proc.trigger_cognitive_assessment()
        assert "timestamp" in assessment
        assert assessment["current_thoughts"] == 0
        assert "processing_stats" in assessment
        await proc.shutdown()

    @pytest.mark.asyncio
    async def test_with_metacognitive_report(self) -> None:
        mono = StubMetacognitiveMonitor()
        proc = CognitiveProcessor(metacognitive_monitor=mono)
        await proc.initialize()

        assessment = await proc.trigger_cognitive_assessment()
        assert "metacognitive_report" in assessment
        assert assessment["metacognitive_report"]["window"] == "1h"
        await proc.shutdown()

    @pytest.mark.asyncio
    async def test_with_memory_statistics(self) -> None:
        mem = StubMemoryManager()
        proc = CognitiveProcessor(memory_manager=mem)
        await proc.initialize()

        assessment = await proc.trigger_cognitive_assessment()
        assert "memory_statistics" in assessment
        assert assessment["memory_statistics"]["total"] == 0
        await proc.shutdown()

    @pytest.mark.asyncio
    async def test_metacognitive_exception_handled(self) -> None:
        class BadMono:
            async def get_metacognitive_report(self, window: str) -> dict:
                raise RuntimeError("fail")

        proc = CognitiveProcessor(metacognitive_monitor=BadMono())
        await proc.initialize()

        assessment = await proc.trigger_cognitive_assessment()
        assert "metacognitive_report" not in assessment
        assert "processing_stats" in assessment
        await proc.shutdown()

    @pytest.mark.asyncio
    async def test_memory_exception_handled(self) -> None:
        class BadMem:
            async def get_memory_statistics(self) -> dict:
                raise RuntimeError("fail")

        proc = CognitiveProcessor(memory_manager=BadMem())
        await proc.initialize()

        assessment = await proc.trigger_cognitive_assessment()
        assert "memory_statistics" not in assessment
        await proc.shutdown()


# ---------------------------------------------------------------------------
# _processing_loop integration
# ---------------------------------------------------------------------------

class TestProcessingLoop:
    @pytest.mark.asyncio
    async def test_processes_queued_thoughts(self) -> None:
        proc = CognitiveProcessor()
        await proc.initialize()

        t = _thought("queued")
        await proc.queue_thought_for_processing(t)

        # give the loop a moment to pick up the item
        await asyncio.sleep(0.3)

        assert t in proc.current_thoughts
        assert proc.processing_stats["total_thoughts_processed"] == 1
        await proc.shutdown()

    @pytest.mark.asyncio
    async def test_loop_stops_on_shutdown(self) -> None:
        proc = CognitiveProcessor()
        await proc.initialize()
        task = proc.processing_task
        assert task is not None

        await proc.shutdown()
        assert task.cancelled()
