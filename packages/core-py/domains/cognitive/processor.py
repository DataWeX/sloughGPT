"""
Cognitive Processor Implementation

This module provides the main cognitive processor that coordinates
memory, reasoning, and metacognitive components.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from ..__init__ import (
    BaseComponent,
    ComponentException,
    ICognitiveProcessor,
    Thought,
)


class CognitiveProcessor(BaseComponent, ICognitiveProcessor):
    """Main cognitive processor that orchestrates cognitive components"""

    def __init__(
        self,
        memory_manager: Optional[Any] = None,
        reasoning_engine: Optional[Any] = None,
        metacognitive_monitor: Optional[Any] = None,
    ) -> None:
        super().__init__("cognitive_processor")
        self.logger = logging.getLogger(f"sloughgpt.{self.component_name}")

        # Core components
        self.memory_manager = memory_manager
        self.reasoning_engine = reasoning_engine
        self.metacognitive_monitor = metacognitive_monitor

        # Processing state
        self.current_thoughts: List[Thought] = []
        self.processing_queue: asyncio.Queue[Any] = asyncio.Queue()
        self.is_processing = False

        # Processing statistics
        self.processing_stats = {
            "total_thoughts_processed": 0,
            "average_processing_time": 0.0,
            "successful_processes": 0,
            "failed_processes": 0,
        }

        # Background processing
        self.processing_task: Optional[asyncio.Task[Any]] = None

        self.is_initialized = False

    async def initialize(self) -> None:
        """Initialize the cognitive processor"""
        try:
            self.logger.info("Initializing Cognitive Processor...")

            # Initialize components if not already initialized
            if self.memory_manager and hasattr(self.memory_manager, "initialize"):
                await self.memory_manager.initialize()

            if self.reasoning_engine and hasattr(self.reasoning_engine, "initialize"):
                await self.reasoning_engine.initialize()

            if self.metacognitive_monitor and hasattr(self.metacognitive_monitor, "initialize"):
                await self.metacognitive_monitor.initialize()

            # Start background processing
            self.processing_task = asyncio.create_task(self._processing_loop())

            self.is_initialized = True
            self.logger.info("Cognitive Processor initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize Cognitive Processor: {e}")
            raise ComponentException(f"Cognitive Processor initialization failed: {e}")

    async def shutdown(self) -> None:
        """Shutdown the cognitive processor"""
        try:
            self.logger.info("Shutting down Cognitive Processor...")

            # Stop processing
            self.is_processing = False

            # Cancel background task
            if self.processing_task:
                self.processing_task.cancel()
                try:
                    await self.processing_task
                except asyncio.CancelledError:
                    pass

            # Shutdown components
            if self.metacognitive_monitor and hasattr(self.metacognitive_monitor, "shutdown"):
                await self.metacognitive_monitor.shutdown()

            if self.reasoning_engine and hasattr(self.reasoning_engine, "shutdown"):
                await self.reasoning_engine.shutdown()

            if self.memory_manager and hasattr(self.memory_manager, "shutdown"):
                await self.memory_manager.shutdown()

            self.is_initialized = False
            self.logger.info("Cognitive Processor shutdown successfully")

        except Exception as e:
            self.logger.error(f"Failed to shutdown Cognitive Processor: {e}")
            raise ComponentException(f"Cognitive Processor shutdown failed: {e}")

    async def process_thought(self, thought: Thought) -> Thought:
        """Process a thought through the cognitive pipeline"""
        try:
            start_time = time.time()

            # Add to current thoughts
            self.current_thoughts.append(thought)

            # Stage 1: Memory Retrieval
            memory_context = await self._retrieve_relevant_memories(thought)

            # Stage 2: Metacognitive Assessment
            if self.metacognitive_monitor:
                assessed_confidence = await self.metacognitive_monitor.assess_confidence(thought)
                thought.confidence = assessed_confidence

            # Stage 3: Reasoning
            if self.reasoning_engine:
                reasoning_result = await self._apply_reasoning(thought, memory_context)
                thought.metadata["reasoning_result"] = reasoning_result

            # Stage 4: Memory Storage
            await self._store_thought_memory(thought)

            # Stage 5: Metacognitive Monitoring
            if self.metacognitive_monitor:
                await self.metacognitive_monitor.monitor_thought_process([thought])

            # Update statistics
            processing_time = time.time() - start_time
            self._update_processing_stats(processing_time, True)

            # Clean up old thoughts
            await self._cleanup_old_thoughts()

            self.logger.debug(
                f"Processed thought: {thought.content[:50]}... in {processing_time:.3f}s"
            )
            return thought

        except Exception as e:
            self.logger.error(f"Failed to process thought: {e}")
            self._update_processing_stats(0, False)
            raise ComponentException(f"Thought processing failed: {e}")

    async def get_cognitive_state(self) -> str:
        """Get current cognitive state"""
        try:
            if self.metacognitive_monitor:
                state_snapshot = await self.metacognitive_monitor.get_cognitive_state_snapshot()
                load = state_snapshot.get("cognitive_load", "unknown")
                return str(load) if load is not None else "unknown"
            return "monitoring_unavailable"

        except Exception as e:
            self.logger.error(f"Failed to get cognitive state: {e}")
            return "error"

    async def set_cognitive_state(self, state: str) -> None:
        """Set cognitive state"""
        try:
            # This would interface with the metacognitive monitor
            # to adjust cognitive parameters based on desired state
            self.logger.info(f"Cognitive state change requested: {state}")

            if self.metacognitive_monitor:
                # Adjust monitoring level based on state
                if state == "focused":
                    await self.metacognitive_monitor.set_monitoring_level("strategic")
                elif state == "creative":
                    await self.metacognitive_monitor.set_monitoring_level("reflective")
                elif state == "analytical":
                    await self.metacognitive_monitor.set_monitoring_level("adaptive")

        except Exception as e:
            self.logger.error(f"Failed to set cognitive state: {e}")
            raise ComponentException(f"Cognitive state setting failed: {e}")

    # Private methods

    async def _retrieve_relevant_memories(self, thought: Thought) -> Dict[str, Any]:
        """Retrieve memories relevant to the thought"""
        if not self.memory_manager:
            return {}

        try:
            # Search for relevant memories
            relevant_memories = await self.memory_manager.search_memories(thought.content, limit=5)

            # Format memory context
            memory_context = {
                "relevant_memories": [
                    {
                        "content": mem.content,
                        "importance": mem.importance,
                        "memory_type": mem.memory_type,
                    }
                    for mem in relevant_memories
                ],
                "retrieval_count": len(relevant_memories),
            }

            # Add to thought metadata
            thought.metadata["memory_context"] = memory_context

            return memory_context

        except Exception as e:
            self.logger.error(f"Memory retrieval failed: {e}")
            return {}

    async def _apply_reasoning(
        self, thought: Thought, memory_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply reasoning to the thought"""
        if not self.reasoning_engine:
            return {}

        try:
            # Prepare reasoning context
            reasoning_context = {
                "memory_context": memory_context,
                "thought_type": thought.thought_type.value,
                "confidence": thought.confidence,
            }

            # Apply reasoning
            reasoning_result = await self.reasoning_engine.reason(
                thought.content, reasoning_context
            )

            return {
                "reasoning_output": reasoning_result,
                "reasoning_path": await self.reasoning_engine.get_reasoning_path(),
                "timestamp": time.time(),
            }

        except Exception as e:
            self.logger.error(f"Reasoning application failed: {e}")
            return {}

    async def _store_thought_memory(self, thought: Thought) -> None:
        """Store the thought as a memory"""
        if not self.memory_manager:
            return

        try:
            # Determine memory importance
            importance = await self._calculate_thought_importance(thought)

            # Store as episodic memory
            memory_id = await self.memory_manager.store_memory(
                {
                    "thought_content": thought.content,
                    "thought_type": thought.thought_type.value,
                    "reasoning_result": thought.metadata.get("reasoning_result"),
                    "confidence": thought.confidence,
                },
                memory_type="episodic",
                importance=importance,
            )

            # Add memory ID to thought metadata
            thought.metadata["memory_id"] = memory_id

            self.logger.debug(f"Stored thought as memory: {memory_id}")

        except Exception as e:
            self.logger.error(f"Thought memory storage failed: {e}")

    async def _calculate_thought_importance(self, thought: Thought) -> float:
        """Calculate importance score for thought storage"""
        base_importance = thought.confidence

        # Boost for certain thought types
        type_boosts: Dict[str, float] = {
            "analytical": 0.1,
            "creative": 0.2,
            "metacognitive": 0.3,
            "intuitive": 0.05,
        }

        type_boost = type_boosts.get(thought.thought_type.value, 0.0)

        # Boost for reasoning results
        reasoning_boost = 0.0
        if thought.metadata.get("reasoning_result"):
            reasoning_boost = 0.1

        # Calculate final importance
        importance = float(base_importance) + float(type_boost) + float(reasoning_boost)
        return max(0.0, min(1.0, importance))

    async def _cleanup_old_thoughts(self) -> None:
        """Clean up old thoughts to prevent memory bloat"""
        max_thoughts = 100

        if len(self.current_thoughts) > max_thoughts:
            # Keep the most recent thoughts
            self.current_thoughts = self.current_thoughts[-max_thoughts:]

    def _update_processing_stats(self, processing_time: float, success: bool) -> None:
        """Update processing statistics"""
        self.processing_stats["total_thoughts_processed"] += 1

        if success:
            self.processing_stats["successful_processes"] += 1
        else:
            self.processing_stats["failed_processes"] += 1

        # Update average processing time
        total_processed = self.processing_stats["total_thoughts_processed"]
        current_avg = self.processing_stats["average_processing_time"]
        new_avg = (current_avg * (total_processed - 1) + processing_time) / total_processed
        self.processing_stats["average_processing_time"] = new_avg

    async def _processing_loop(self) -> None:
        """Background processing loop for queued thoughts"""
        self.is_processing = True

        while self.is_initialized:
            try:
                # Process queued thoughts
                thought = await asyncio.wait_for(self.processing_queue.get(), timeout=1.0)

                await self.process_thought(thought)
                self.processing_queue.task_done()

            except asyncio.TimeoutError:
                # No thoughts to process, continue
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Processing loop error: {e}")
                await asyncio.sleep(0.1)

    # Public API methods

    async def queue_thought_for_processing(self, thought: Thought) -> None:
        """Queue a thought for background processing"""
        await self.processing_queue.put(thought)

    async def get_processing_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        stats = self.processing_stats.copy()

        # Calculate success rate
        total = stats["total_thoughts_processed"]
        if total > 0:
            stats["success_rate"] = stats["successful_processes"] / total
        else:
            stats["success_rate"] = 0.0

        # Add current queue size
        stats["queue_size"] = self.processing_queue.qsize()
        stats["current_thoughts_count"] = len(self.current_thoughts)

        return stats

    async def batch_process_thoughts(self, thoughts: List[Thought]) -> List[Thought]:
        """Process multiple thoughts in batch"""
        processed_thoughts = []

        for thought in thoughts:
            try:
                processed_thought = await self.process_thought(thought)
                processed_thoughts.append(processed_thought)
            except Exception as e:
                self.logger.error(f"Failed to process thought in batch: {e}")
                # Add original thought if processing failed
                processed_thoughts.append(thought)

        return processed_thoughts

    async def trigger_cognitive_assessment(self) -> Dict[str, Any]:
        """Trigger comprehensive cognitive assessment"""
        assessment = {
            "timestamp": time.time(),
            "current_thoughts": len(self.current_thoughts),
            "processing_stats": await self.get_processing_statistics(),
        }

        # Add metacognitive assessment if available
        if self.metacognitive_monitor:
            try:
                metacognitive_report = await self.metacognitive_monitor.get_metacognitive_report(
                    "1h"
                )
                assessment["metacognitive_report"] = metacognitive_report
            except Exception as e:
                self.logger.error(f"Failed to get metacognitive report: {e}")

        # Add memory statistics if available
        if self.memory_manager:
            try:
                memory_stats = await self.memory_manager.get_memory_statistics()
                assessment["memory_statistics"] = memory_stats
            except Exception as e:
                self.logger.error(f"Failed to get memory statistics: {e}")

        return assessment
