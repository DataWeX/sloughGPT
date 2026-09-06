"""Tests for cognitive.processor — CognitiveProcessor."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock, AsyncMock

import pytest

from domains import BaseComponent, ComponentException, Thought, ThoughtType
from domains.cognitive.processor import CognitiveProcessor


# ── Helpers ─────────────────────────────────────────────────────────────────


def _make_thought(content="test thought", thought_type="reasoning", confidence=0.5):
    return Thought(
        thought_id="t1",
        content=content,
        thought_type=thought_type,
        confidence=confidence,
    )


class MockMemoryManager:
    def __init__(self):
        self.stored = []
        self.memories = []

    async def initialize(self):
        pass

    async def shutdown(self):
        pass

    async def search_memories(self, query, limit=5):
        return self.memories

    async def store_memory(self, data, memory_type="episodic", importance=0.5):
        self.stored.append({"data": data, "type": memory_type, "importance": importance})
        return "mem_001"

    async def get_memory_statistics(self):
        return {"total": len(self.stored)}


class MockReasoningEngine:
    def __init__(self):
        self.result = {"output": "reasoned"}

    async def initialize(self):
        pass

    async def shutdown(self):
        pass

    async def reason(self, content, context):
        return self.result

    async def get_reasoning_path(self):
        return ["step1", "step2"]


class MockMetacognitiveMonitor:
    def __init__(self):
        self.confidence = 0.7

    async def initialize(self):
        pass

    async def shutdown(self):
        pass

    async def assess_confidence(self, thought):
        return self.confidence

    async def monitor_thought_process(self, thoughts):
        pass

    async def get_cognitive_state_snapshot(self):
        return {"cognitive_load": "moderate"}

    async def set_monitoring_level(self, level):
        pass

    async def get_metacognitive_report(self, period):
        return {"period": period, "insights": []}


# ── CognitiveProcessor ──────────────────────────────────────────────────────


class TestCognitiveProcessor:

    def test_init(self):
        proc = CognitiveProcessor()
        assert proc.component_name == "cognitive_processor"
        assert proc.is_initialized is False
        assert proc.current_thoughts == []
        assert proc.processing_stats["total_thoughts_processed"] == 0

    def test_init_with_components(self):
        mm = MockMemoryManager()
        re = MockReasoningEngine()
        mc = MockMetacognitiveMonitor()
        proc = CognitiveProcessor(mm, re, mc)
        assert proc.memory_manager is mm
        assert proc.reasoning_engine is re
        assert proc.metacognitive_monitor is mc

    @pytest.mark.asyncio
    async def test_initialize(self):
        proc = CognitiveProcessor()
        await proc.initialize()
        assert proc.is_initialized is True
        assert proc.processing_task is not None
        proc.processing_task.cancel()

    @pytest.mark.asyncio
    async def test_initialize_with_components(self):
        mm = MockMemoryManager()
        re = MockReasoningEngine()
        mc = MockMetacognitiveMonitor()
        proc = CognitiveProcessor(mm, re, mc)
        await proc.initialize()
        assert proc.is_initialized is True
        proc.processing_task.cancel()

    @pytest.mark.asyncio
    async def test_shutdown(self):
        proc = CognitiveProcessor()
        await proc.initialize()
        await proc.shutdown()
        assert proc.is_initialized is False

    @pytest.mark.asyncio
    async def test_shutdown_without_init(self):
        proc = CognitiveProcessor()
        await proc.shutdown()
        assert proc.is_initialized is False

    @pytest.mark.asyncio
    async def test_process_thought_no_components(self):
        proc = CognitiveProcessor()
        await proc.initialize()
        thought = _make_thought()
        result = await proc.process_thought(thought)
        assert result is thought
        assert len(proc.current_thoughts) == 1
        assert proc.processing_stats["total_thoughts_processed"] == 1
        assert proc.processing_stats["successful_processes"] == 1
        proc.processing_task.cancel()

    @pytest.mark.asyncio
    async def test_process_thought_with_components(self):
        mm = MockMemoryManager()
        re = MockReasoningEngine()
        mc = MockMetacognitiveMonitor()
        proc = CognitiveProcessor(mm, re, mc)
        await proc.initialize()
        thought = _make_thought()
        result = await proc.process_thought(thought)
        assert result is thought
        assert thought.metadata.get("memory_id") == "mem_001"
        assert "reasoning_result" in thought.metadata
        assert thought.confidence == 0.7
        assert len(mm.stored) == 1
        proc.processing_task.cancel()

    @pytest.mark.asyncio
    async def test_process_thought_importance_calculation(self):
        mm = MockMemoryManager()
        proc = CognitiveProcessor(memory_manager=mm)
        await proc.initialize()
        thought = _make_thought(thought_type="metacognitive", confidence=0.6)
        thought.metadata["reasoning_result"] = {"output": "x"}
        await proc.process_thought(thought)
        # 0.6 + 0.3 (metacognitive) + 0.1 (reasoning) = 1.0
        assert abs(mm.stored[0]["importance"] - 1.0) < 0.001
        proc.processing_task.cancel()

    @pytest.mark.asyncio
    async def test_process_thought_importance_clamped(self):
        mm = MockMemoryManager()
        proc = CognitiveProcessor(memory_manager=mm)
        await proc.initialize()
        thought = _make_thought(confidence=0.9)
        thought.metadata["reasoning_result"] = True
        await proc.process_thought(thought)
        assert mm.stored[0]["importance"] <= 1.0
        proc.processing_task.cancel()

    @pytest.mark.asyncio
    async def test_process_thought_failure(self):
        proc = CognitiveProcessor()
        await proc.initialize()
        async def fail_store(thought):
            raise RuntimeError("boom")
        proc._store_thought_memory = fail_store
        thought = _make_thought()
        with pytest.raises(Exception, match="Thought processing failed"):
            await proc.process_thought(thought)
        assert proc.processing_stats["failed_processes"] == 1
        proc.processing_task.cancel()

    @pytest.mark.asyncio
    async def test_get_cognitive_state_no_monitor(self):
        proc = CognitiveProcessor()
        await proc.initialize()
        state = await proc.get_cognitive_state()
        assert state == "monitoring_unavailable"
        proc.processing_task.cancel()

    @pytest.mark.asyncio
    async def test_get_cognitive_state_with_monitor(self):
        mc = MockMetacognitiveMonitor()
        proc = CognitiveProcessor(metacognitive_monitor=mc)
        await proc.initialize()
        state = await proc.get_cognitive_state()
        assert state == "moderate"
        proc.processing_task.cancel()

    @pytest.mark.asyncio
    async def test_set_cognitive_state_focused(self):
        mc = MockMetacognitiveMonitor()
        proc = CognitiveProcessor(metacognitive_monitor=mc)
        await proc.initialize()
        await proc.set_cognitive_state("focused")
        # No exception = success
        proc.processing_task.cancel()

    @pytest.mark.asyncio
    async def test_set_cognitive_state_creative(self):
        mc = MockMetacognitiveMonitor()
        proc = CognitiveProcessor(metacognitive_monitor=mc)
        await proc.initialize()
        await proc.set_cognitive_state("creative")
        proc.processing_task.cancel()

    @pytest.mark.asyncio
    async def test_set_cognitive_state_analytical(self):
        mc = MockMetacognitiveMonitor()
        proc = CognitiveProcessor(metacognitive_monitor=mc)
        await proc.initialize()
        await proc.set_cognitive_state("analytical")
        proc.processing_task.cancel()

    @pytest.mark.asyncio
    async def test_queue_thought(self):
        proc = CognitiveProcessor()
        await proc.initialize()
        thought = _make_thought()
        await proc.queue_thought_for_processing(thought)
        assert proc.processing_queue.qsize() == 1
        proc.processing_task.cancel()

    @pytest.mark.asyncio
    async def test_batch_process(self):
        proc = CognitiveProcessor()
        await proc.initialize()
        thoughts = [_make_thought(f"t{i}") for i in range(3)]
        results = await proc.batch_process_thoughts(thoughts)
        assert len(results) == 3
        assert proc.processing_stats["total_thoughts_processed"] == 3
        proc.processing_task.cancel()

    @pytest.mark.asyncio
    async def test_batch_process_with_failure(self):
        proc = CognitiveProcessor()
        await proc.initialize()
        call_count = 0
        original_process = proc.process_thought

        async def fail_on_second(t):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise ComponentException("fail")
            return await original_process(t)

        proc.process_thought = fail_on_second
        thoughts = [_make_thought(f"t{i}") for i in range(3)]
        results = await proc.batch_process_thoughts(thoughts)
        assert len(results) == 3
        proc.processing_task.cancel()

    @pytest.mark.asyncio
    async def test_get_processing_statistics(self):
        proc = CognitiveProcessor()
        await proc.initialize()
        stats = await proc.get_processing_statistics()
        assert "total_thoughts_processed" in stats
        assert "success_rate" in stats
        assert "queue_size" in stats
        assert stats["success_rate"] == 0.0
        proc.processing_task.cancel()

    @pytest.mark.asyncio
    async def test_trigger_cognitive_assessment(self):
        mm = MockMemoryManager()
        mc = MockMetacognitiveMonitor()
        proc = CognitiveProcessor(memory_manager=mm, metacognitive_monitor=mc)
        await proc.initialize()
        assessment = await proc.trigger_cognitive_assessment()
        assert "timestamp" in assessment
        assert "processing_stats" in assessment
        assert "metacognitive_report" in assessment
        assert "memory_statistics" in assessment
        proc.processing_task.cancel()

    @pytest.mark.asyncio
    async def test_cleanup_old_thoughts(self):
        proc = CognitiveProcessor()
        await proc.initialize()
        proc.current_thoughts = [_make_thought(f"t{i}") for i in range(150)]
        await proc._cleanup_old_thoughts()
        assert len(proc.current_thoughts) == 100
        proc.processing_task.cancel()

    def test_update_processing_stats(self):
        proc = CognitiveProcessor()
        proc._update_processing_stats(0.5, True)
        assert proc.processing_stats["total_thoughts_processed"] == 1
        assert proc.processing_stats["successful_processes"] == 1
        assert proc.processing_stats["average_processing_time"] == 0.5

        proc._update_processing_stats(0.3, False)
        assert proc.processing_stats["total_thoughts_processed"] == 2
        assert proc.processing_stats["failed_processes"] == 1
        avg = proc.processing_stats["average_processing_time"]
        assert 0.3 < avg < 0.5
