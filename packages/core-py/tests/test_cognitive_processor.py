"""Tests for CognitiveProcessor."""

from __future__ import annotations

import asyncio
import pytest
from domains import Thought
from domains.cognitive.processor import CognitiveProcessor


class DummyMemoryManager:
    def __init__(self):
        self.initialized = False
        self.memories = []

    async def initialize(self):
        self.initialized = True

    async def shutdown(self):
        self.initialized = False

    async def search_memories(self, content: str, limit: int = 5):
        return [m for m in self.memories]

    async def store_memory(self, data: dict, memory_type: str = "episodic", importance: float = 0.5):
        self.memories.append(data)
        return f"mem_{len(self.memories)}"

    async def get_memory_statistics(self):
        return {"total_memories": len(self.memories)}


class DummyReasoningEngine:
    def __init__(self):
        self.initialized = False

    async def initialize(self):
        self.initialized = True

    async def shutdown(self):
        self.initialized = False

    async def reason(self, content: str, context: dict):
        return f"reasoned: {content[:20]}"

    async def get_reasoning_path(self):
        return ["step1", "step2"]


class DummyMetacognitiveMonitor:
    def __init__(self):
        self.initialized = False

    async def initialize(self):
        self.initialized = True

    async def shutdown(self):
        self.initialized = False

    async def assess_confidence(self, thought):
        return 0.85

    async def monitor_thought_process(self, thoughts):
        return {"assessments": [], "overall_efficiency": 0.8}

    async def get_cognitive_state_snapshot(self):
        return {"cognitive_load": 0.4}

    async def set_monitoring_level(self, level: str):
        pass

    async def get_metacognitive_report(self, time_range: str):
        return {"assessments_count": 1, "average_efficiency": 0.8}


class TestCognitiveProcessor:
    def make_thought(self, content="test thought", confidence=0.8):
        return Thought(thought_id="t1", content=content, thought_type="reasoning", confidence=confidence)

    async def test_initial_state(self):
        cp = CognitiveProcessor()
        assert cp.is_initialized is False
        assert cp.is_processing is False
        assert cp.processing_stats["total_thoughts_processed"] == 0

    async def test_initialize_empty(self):
        cp = CognitiveProcessor()
        await cp.initialize()
        assert cp.is_initialized is True
        await cp.shutdown()

    async def test_initialize_with_components(self):
        mm = DummyMemoryManager()
        cp = CognitiveProcessor(memory_manager=mm)
        await cp.initialize()
        assert mm.initialized is True
        await cp.shutdown()

    async def test_shutdown_cleans_up(self):
        cp = CognitiveProcessor()
        await cp.initialize()
        await cp.shutdown()
        assert cp.is_initialized is False
        assert cp.is_processing is False

    async def test_process_thought_basic(self):
        cp = CognitiveProcessor()
        await cp.initialize()
        thought = self.make_thought()
        result = await cp.process_thought(thought)
        assert result.thought_id == "t1"
        assert cp.processing_stats["total_thoughts_processed"] == 1
        await cp.shutdown()

    async def test_process_thought_with_monitor(self):
        mm = DummyMetacognitiveMonitor()
        cp = CognitiveProcessor(metacognitive_monitor=mm)
        await cp.initialize()
        thought = self.make_thought(confidence=0.5)
        result = await cp.process_thought(thought)
        assert result.confidence == 0.85
        await cp.shutdown()

    async def test_process_thought_with_memory(self):
        mm = DummyMemoryManager()
        cp = CognitiveProcessor(memory_manager=mm)
        await cp.initialize()
        thought = self.make_thought()
        result = await cp.process_thought(thought)
        assert "memory_context" in thought.metadata
        await cp.shutdown()

    async def test_process_thought_with_reasoning(self):
        re = DummyReasoningEngine()
        cp = CognitiveProcessor(reasoning_engine=re)
        await cp.initialize()
        thought = self.make_thought()
        result = await cp.process_thought(thought)
        assert "reasoning_result" in thought.metadata
        await cp.shutdown()

    async def test_process_thought_appends_to_current(self):
        cp = CognitiveProcessor()
        await cp.initialize()
        t1 = self.make_thought(content="first", confidence=0.7)
        t2 = self.make_thought(content="second", confidence=0.8)
        await cp.process_thought(t1)
        await cp.process_thought(t2)
        assert len(cp.current_thoughts) == 2
        await cp.shutdown()

    async def test_get_cognitive_state_no_monitor(self):
        cp = CognitiveProcessor()
        state = await cp.get_cognitive_state()
        assert state == "monitoring_unavailable"

    async def test_get_cognitive_state_with_monitor(self):
        mm = DummyMetacognitiveMonitor()
        cp = CognitiveProcessor(metacognitive_monitor=mm)
        state = await cp.get_cognitive_state()
        assert state == "0.4"

    async def test_set_cognitive_state_focused(self):
        mm = DummyMetacognitiveMonitor()
        cp = CognitiveProcessor(metacognitive_monitor=mm)
        await cp.set_cognitive_state("focused")
        assert cp.is_initialized is False

    async def test_set_cognitive_state_creative(self):
        mm = DummyMetacognitiveMonitor()
        cp = CognitiveProcessor(metacognitive_monitor=mm)
        await cp.set_cognitive_state("creative")

    async def test_set_cognitive_state_analytical(self):
        mm = DummyMetacognitiveMonitor()
        cp = CognitiveProcessor(metacognitive_monitor=mm)
        await cp.set_cognitive_state("analytical")

    async def test_get_processing_statistics(self):
        cp = CognitiveProcessor()
        stats = await cp.get_processing_statistics()
        assert stats["total_thoughts_processed"] == 0
        assert stats["success_rate"] == 0.0
        assert "queue_size" in stats

    async def test_get_processing_statistics_after_process(self):
        cp = CognitiveProcessor()
        await cp.initialize()
        thought = self.make_thought()
        await cp.process_thought(thought)
        stats = await cp.get_processing_statistics()
        assert stats["total_thoughts_processed"] == 1
        assert stats["success_rate"] == 1.0
        await cp.shutdown()

    async def test_queue_and_process(self):
        cp = CognitiveProcessor()
        await cp.initialize()
        thought = self.make_thought()
        await cp.queue_thought_for_processing(thought)
        await asyncio.sleep(0.05)
        assert cp.processing_stats["total_thoughts_processed"] >= 1
        await cp.shutdown()

    async def test_batch_process_thoughts(self):
        cp = CognitiveProcessor()
        await cp.initialize()
        thoughts = [self.make_thought(content=f"thought_{i}", confidence=0.5 + i * 0.1) for i in range(3)]
        results = await cp.batch_process_thoughts(thoughts)
        assert len(results) == 3
        assert cp.processing_stats["total_thoughts_processed"] == 3
        await cp.shutdown()

    async def test_trigger_cognitive_assessment(self):
        cp = CognitiveProcessor()
        await cp.initialize()
        assessment = await cp.trigger_cognitive_assessment()
        assert "timestamp" in assessment
        assert "processing_stats" in assessment
        assert assessment["current_thoughts"] == 0
        await cp.shutdown()

    async def test_trigger_cognitive_assessment_with_monitor(self):
        mm = DummyMetacognitiveMonitor()
        cp = CognitiveProcessor(metacognitive_monitor=mm)
        assessment = await cp.trigger_cognitive_assessment()
        assert "metacognitive_report" in assessment

    async def test_trigger_cognitive_assessment_with_memory(self):
        mm = DummyMemoryManager()
        cp = CognitiveProcessor(memory_manager=mm)
        assessment = await cp.trigger_cognitive_assessment()
        assert "memory_statistics" in assessment
        assert assessment["memory_statistics"]["total_memories"] == 0

    async def test_cleanup_old_thoughts(self):
        cp = CognitiveProcessor()
        for i in range(105):
            cp.current_thoughts.append(self.make_thought(content=f"t{i}", confidence=0.5))
        await cp._cleanup_old_thoughts()
        assert len(cp.current_thoughts) <= 100

    async def test_update_processing_stats(self):
        cp = CognitiveProcessor()
        cp._update_processing_stats(0.5, True)
        assert cp.processing_stats["total_thoughts_processed"] == 1
        assert cp.processing_stats["successful_processes"] == 1
        cp._update_processing_stats(0.3, False)
        assert cp.processing_stats["total_thoughts_processed"] == 2
        assert cp.processing_stats["failed_processes"] == 1

    async def test_calculate_thought_importance(self):
        cp = CognitiveProcessor()
        thought = self.make_thought(confidence=0.7)
        importance = await cp._calculate_thought_importance(thought)
        assert 0.0 <= importance <= 1.0

    async def test_retrieve_relevant_memories_no_manager(self):
        cp = CognitiveProcessor()
        thought = self.make_thought()
        mems = await cp._retrieve_relevant_memories(thought)
        assert mems == {}

    async def test_retrieve_relevant_memories_with_manager(self):
        mm = DummyMemoryManager()
        cp = CognitiveProcessor(memory_manager=mm)
        thought = self.make_thought(content="test memory retrieval")
        mems = await cp._retrieve_relevant_memories(thought)
        assert "relevant_memories" in mems
        assert mems["retrieval_count"] == 0

    async def test_apply_reasoning_no_engine(self):
        cp = CognitiveProcessor()
        result = await cp._apply_reasoning(self.make_thought(), {})
        assert result == {}

    async def test_apply_reasoning_with_engine(self):
        re = DummyReasoningEngine()
        cp = CognitiveProcessor(reasoning_engine=re)
        result = await cp._apply_reasoning(self.make_thought(), {"ctx": "val"})
        assert "reasoning_output" in result
        assert "reasoning_path" in result

    async def test_store_thought_memory_no_manager(self):
        cp = CognitiveProcessor()
        await cp._store_thought_memory(self.make_thought())

    async def test_store_thought_memory_with_manager(self):
        mm = DummyMemoryManager()
        cp = CognitiveProcessor(memory_manager=mm)
        thought = self.make_thought()
        await cp._store_thought_memory(thought)
        assert "memory_id" in thought.metadata

    async def test_processing_loop_queues(self):
        cp = CognitiveProcessor()
        await cp.initialize()
        thought = self.make_thought()
        await cp.processing_queue.put(thought)
        await asyncio.sleep(0.05)
        assert cp.processing_stats["total_thoughts_processed"] >= 1
        await cp.shutdown()
