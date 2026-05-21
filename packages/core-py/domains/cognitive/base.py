"""
Cognitive Domain Base Class

This module contains the base cognitive domain implementation.
"""

import asyncio
import logging
from typing import Any, Dict, Optional

from ..__init__ import (
    BaseDomain,
    DomainException,
    ICognitiveProcessor,
    IMemoryManager,
    IMetacognitiveMonitor,
    IReasoningEngine,
    Memory,
    Thought,
    ThoughtType,
)


class CognitiveDomain(BaseDomain):
    """Main cognitive architecture domain"""

    def __init__(self) -> None:
        super().__init__("cognitive")
        self.logger = logging.getLogger(f"sloughgpt.{self.domain_name}")

        # Core components
        self.memory_manager: Optional[IMemoryManager] = None
        self.reasoning_engine: Optional[IReasoningEngine] = None
        self.metacognitive_monitor: Optional[IMetacognitiveMonitor] = None
        self.cognitive_processor: Optional[ICognitiveProcessor] = None

        # Cognitive state
        self.cognitive_state = "idle"
        self.active_thoughts = []
        self.memory_store = {}

    async def _on_initialize(self) -> None:
        """Initialize cognitive domain components"""
        try:
            self.logger.info("Initializing Cognitive Domain...")

            # Initialize core components
            await self._initialize_memory_manager()
            await self._initialize_reasoning_engine()
            await self._initialize_metacognitive_monitor()
            await self._initialize_cognitive_processor()

            # Start cognitive processes
            await self._start_cognitive_processes()

            self.logger.info("Cognitive Domain initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize Cognitive Domain: {e}")
            raise CognitiveException(f"Initialization failed: {e}")

    async def _on_shutdown(self) -> None:
        """Shutdown cognitive domain components"""
        try:
            self.logger.info("Shutting down Cognitive Domain...")

            # Stop cognitive processes
            await self._stop_cognitive_processes()

            # Shutdown components
            if self.cognitive_processor:
                await self._shutdown_component("cognitive_processor")
            if self.metacognitive_monitor:
                await self._shutdown_component("metacognitive_monitor")
            if self.reasoning_engine:
                await self._shutdown_component("reasoning_engine")
            if self.memory_manager:
                await self._shutdown_component("memory_manager")

            self.logger.info("Cognitive Domain shutdown successfully")

        except Exception as e:
            self.logger.error(f"Failed to shutdown Cognitive Domain: {e}")
            raise CognitiveException(f"Shutdown failed: {e}")

    async def _initialize_memory_manager(self) -> None:
        """Initialize memory manager"""
        from .memory import MemoryManager

        self.memory_manager = MemoryManager()
        await self.memory_manager.initialize()
        self.components["memory_manager"] = self.memory_manager

    async def _initialize_reasoning_engine(self) -> None:
        """Initialize reasoning engine"""
        from .reasoning import ReasoningEngine

        self.reasoning_engine = ReasoningEngine()
        await self.reasoning_engine.initialize()
        self.components["reasoning_engine"] = self.reasoning_engine

    async def _initialize_metacognitive_monitor(self) -> None:
        """Initialize metacognitive monitor"""
        from .metacognition import MetacognitiveMonitor

        self.metacognitive_monitor = MetacognitiveMonitor()
        await self.metacognitive_monitor.initialize()
        self.components["metacognitive_monitor"] = self.metacognitive_monitor

    async def _initialize_cognitive_processor(self) -> None:
        """Initialize cognitive processor"""
        from .processor import CognitiveProcessor

        self.cognitive_processor = CognitiveProcessor(
            memory_manager=self.memory_manager,
            reasoning_engine=self.reasoning_engine,
            metacognitive_monitor=self.metacognitive_monitor,
        )
        await self.cognitive_processor.initialize()
        self.components["cognitive_processor"] = self.cognitive_processor

    async def _start_cognitive_processes(self) -> None:
        """Start background cognitive processes"""
        # Start memory consolidation
        asyncio.create_task(self._memory_consolidation_loop())

        # Start metacognitive monitoring
        asyncio.create_task(self._metacognitive_monitoring_loop())

        # Start reasoning optimization
        asyncio.create_task(self._reasoning_optimization_loop())

    async def _stop_cognitive_processes(self) -> None:
        """Stop background cognitive processes"""
        # Implementation for stopping background processes
        pass

    async def _memory_consolidation_loop(self) -> None:
        """Background memory consolidation loop"""
        while self.is_initialized:
            try:
                if self.memory_manager:
                    await self.memory_manager.consolidate_memories()
                await asyncio.sleep(60)  # Consolidate every minute
            except Exception as e:
                self.logger.error(f"Memory consolidation error: {e}")
                await asyncio.sleep(10)

    async def _metacognitive_monitoring_loop(self) -> None:
        """Background metacognitive monitoring loop"""
        while self.is_initialized:
            try:
                if self.metacognitive_monitor and self.active_thoughts:
                    await self.metacognitive_monitor.monitor_thought_process(self.active_thoughts)
                await asyncio.sleep(30)  # Monitor every 30 seconds
            except Exception as e:
                self.logger.error(f"Metacognitive monitoring error: {e}")
                await asyncio.sleep(10)

    async def _reasoning_optimization_loop(self) -> None:
        """Background reasoning optimization loop"""
        while self.is_initialized:
            try:
                if self.reasoning_engine:
                    await self.reasoning_engine.optimize_reasoning_strategies()
                await asyncio.sleep(300)  # Optimize every 5 minutes
            except Exception as e:
                self.logger.error(f"Reasoning optimization error: {e}")
                await asyncio.sleep(30)

    async def _shutdown_component(self, component_name: str) -> None:
        """Safely shutdown a component"""
        try:
            component = getattr(self, component_name)
            if hasattr(component, "shutdown"):
                await component.shutdown()
        except Exception as e:
            self.logger.error(f"Error shutting down {component_name}: {e}")

    # Public API methods

    async def process_thought(
        self, thought_content: str, thought_type: str = "analytical"
    ) -> Dict[str, Any]:
        """Process a thought through the cognitive pipeline"""
        if not self.cognitive_processor:
            raise CognitiveException("Cognitive processor not initialized")

        import uuid

        thought = Thought(
            thought_id=str(uuid.uuid4()),
            content=thought_content,
            thought_type=ThoughtType(thought_type),
            confidence=0.5,
            metadata={},
        )

        result = await self.cognitive_processor.process_thought(thought)
        self.active_thoughts.append(result)

        return {
            "original_thought": thought_content,
            "processed_thought": result.content,
            "confidence": result.confidence,
            "reasoning_path": await self.reasoning_engine.get_reasoning_path()
            if self.reasoning_engine
            else [],
        }

    async def store_memory(
        self, content: Any, memory_type: str = "episodic", importance: float = 0.5
    ) -> str:
        """Store a memory"""
        if not self.memory_manager:
            raise CognitiveException("Memory manager not initialized")

        memory = Memory(
            content=content,
            memory_type=memory_type,
            importance=importance,
            associations=[],
            retrieval_count=0,
            last_accessed=asyncio.get_event_loop().time(),
            metadata={},
        )

        memory_id = await self.memory_manager.store_memory(memory)
        return memory_id

    async def retrieve_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a memory by ID"""
        if not self.memory_manager:
            raise CognitiveException("Memory manager not initialized")

        memory = await self.memory_manager.retrieve_memory(memory_id)
        if memory:
            return {
                "content": memory.content,
                "memory_type": memory.memory_type,
                "importance": memory.importance,
                "retrieval_count": memory.retrieval_count,
                "last_accessed": memory.last_accessed,
            }
        return None

    async def reason(self, premise: str, context: Dict[str, Any]) -> str:
        """Perform reasoning on a premise"""
        if not self.reasoning_engine:
            raise CognitiveException("Reasoning engine not initialized")

        return await self.reasoning_engine.reason(premise, context)

    async def get_cognitive_state(self) -> Dict[str, Any]:
        """Get current cognitive state"""
        return {
            "state": self.cognitive_state,
            "active_thoughts_count": len(self.active_thoughts),
            "memory_count": len(self.memory_store) if self.memory_store else 0,
            "components_status": {
                name: "initialized" if component else "not_initialized"
                for name, component in [
                    ("memory_manager", self.memory_manager),
                    ("reasoning_engine", self.reasoning_engine),
                    ("metacognitive_monitor", self.metacognitive_monitor),
                    ("cognitive_processor", self.cognitive_processor),
                ]
            },
        }


class CognitiveException(DomainException):
    """Cognitive domain specific exceptions"""

    pass
