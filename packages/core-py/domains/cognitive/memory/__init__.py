"""
Memory Manager Implementation

This module provides advanced memory management capabilities including
episodic, semantic, and procedural memory systems.
"""

import asyncio
import hashlib
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ...__init__ import BaseComponent, ComponentException, IMemoryManager, Memory


class MemoryType:
    """Memory type constants"""

    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    WORKING = "working"
    LONG_TERM = "long_term"


@dataclass
class MemoryAssociation:
    """Memory association for linking related memories"""

    source_memory_id: str
    target_memory_id: str
    association_strength: float
    association_type: str
    created_at: float


@dataclass
class MemoryConsolidation:
    """Memory consolidation information"""

    memory_id: str
    consolidation_score: float
    last_consolidated: float
    consolidation_count: int


class MemoryManager(BaseComponent, IMemoryManager):
    """Advanced memory management system"""

    def __init__(self) -> None:
        super().__init__("memory_manager")
        self.logger = logging.getLogger(f"sloughgpt.{self.component_name}")

        # Memory stores
        self.episodic_memory: Dict[str, Memory] = {}
        self.semantic_memory: Dict[str, Memory] = {}
        self.procedural_memory: Dict[str, Memory] = {}
        self.working_memory: Dict[str, Memory] = {}

        # Memory associations
        self.associations: Dict[str, List[MemoryAssociation]] = defaultdict(list)

        # Consolidation tracking
        self.consolidation_info: Dict[str, MemoryConsolidation] = {}

        # Configuration
        self.working_memory_capacity = 50
        self.consolidation_threshold = 0.7
        self.forgetting_rate = 0.001

        # Background tasks
        self.consolidation_task: Optional[asyncio.Task[None]] = None
        self.forgetting_task: Optional[asyncio.Task[None]] = None

        self.is_initialized = False

    async def initialize(self) -> None:
        """Initialize the memory manager"""
        try:
            self.logger.info("Initializing Memory Manager...")

            # Start background processes
            self.consolidation_task = asyncio.create_task(self._consolidation_loop())
            self.forgetting_task = asyncio.create_task(self._forgetting_loop())

            self.is_initialized = True
            self.logger.info("Memory Manager initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize Memory Manager: {e}")
            raise ComponentException(f"Memory Manager initialization failed: {e}")

    async def shutdown(self) -> None:
        """Shutdown the memory manager"""
        try:
            self.logger.info("Shutting down Memory Manager...")

            # Cancel background tasks
            if self.consolidation_task:
                self.consolidation_task.cancel()
            if self.forgetting_task:
                self.forgetting_task.cancel()

            # Wait for tasks to complete
            tasks_to_wait: List[asyncio.Task[None]] = []
            if self.consolidation_task is not None:
                tasks_to_wait.append(self.consolidation_task)
            if self.forgetting_task is not None:
                tasks_to_wait.append(self.forgetting_task)

            if tasks_to_wait:
                await asyncio.gather(*tasks_to_wait, return_exceptions=True)

            self.is_initialized = False
            self.logger.info("Memory Manager shutdown successfully")

        except Exception as e:
            self.logger.error(f"Failed to shutdown Memory Manager: {e}")
            raise ComponentException(f"Memory Manager shutdown failed: {e}")

    async def store_memory(self, memory: Memory) -> str:
        """Store a memory and return its ID"""
        try:
            # Generate memory ID if not provided
            memory_id = self._generate_memory_id(memory)

            # Add metadata if not present
            if not hasattr(memory, "created_at"):
                # Add timestamp to metadata dict
                if not hasattr(memory, "metadata"):
                    memory.metadata = {}
                memory.metadata["created_at"] = time.time()

            # Store in appropriate memory store
            if memory.memory_type == MemoryType.EPISODIC:
                self.episodic_memory[memory_id] = memory
            elif memory.memory_type == MemoryType.SEMANTIC:
                self.semantic_memory[memory_id] = memory
            elif memory.memory_type == MemoryType.PROCEDURAL:
                self.procedural_memory[memory_id] = memory
            elif memory.memory_type == MemoryType.WORKING:
                # Check working memory capacity
                if len(self.working_memory) >= self.working_memory_capacity:
                    await self._evict_working_memory()
                self.working_memory[memory_id] = memory
            else:
                # Default to long-term storage
                memory.memory_type = MemoryType.LONG_TERM
                self.episodic_memory[memory_id] = memory

            # Initialize consolidation info
            self.consolidation_info[memory_id] = MemoryConsolidation(
                memory_id=memory_id,
                consolidation_score=memory.importance,
                last_consolidated=time.time(),
                consolidation_count=0,
            )

            # Create associations if needed
            await self._create_memory_associations(memory_id, memory)

            self.logger.debug(f"Stored memory {memory_id} of type {memory.memory_type}")
            return memory_id

        except Exception as e:
            self.logger.error(f"Failed to store memory: {e}")
            raise ComponentException(f"Memory storage failed: {e}")

    async def retrieve_memory(self, memory_id: str) -> Optional[Memory]:
        """Retrieve a memory by ID"""
        try:
            # Search in all memory stores
            memory = None

            if memory_id in self.episodic_memory:
                memory = self.episodic_memory[memory_id]
                # mem_store = self.episodic_memory  # noqa: F841
            elif memory_id in self.semantic_memory:
                memory = self.semantic_memory[memory_id]
                # mem_store = self.semantic_memory  # noqa: F841
            elif memory_id in self.procedural_memory:
                memory = self.procedural_memory[memory_id]
                # mem_store = self.procedural_memory  # noqa: F841
            elif memory_id in self.working_memory:
                memory = self.working_memory[memory_id]
                # mem_store = self.working_memory  # noqa: F841

            if memory:
                # Update access statistics
                memory.retrieval_count += 1
                memory.last_accessed = time.time()

                # Update consolidation info
                if memory_id in self.consolidation_info:
                    self.consolidation_info[memory_id].consolidation_score += 0.1

                self.logger.debug(f"Retrieved memory {memory_id}")
                return memory

            return None

        except Exception as e:
            self.logger.error(f"Failed to retrieve memory {memory_id}: {e}")
            raise ComponentException(f"Memory retrieval failed: {e}")

    async def search_memories(self, query: str, limit: int = 10) -> List[Memory]:
        """Search memories by content"""
        try:
            results = []
            query_lower = query.lower()

            # Search in all memory stores
            all_memories = {}
            all_memories.update(self.episodic_memory)
            all_memories.update(self.semantic_memory)
            all_memories.update(self.procedural_memory)
            all_memories.update(self.working_memory)

            # Simple text search (can be enhanced with vector similarity)
            scored_memories = []
            for memory_id, memory in all_memories.items():
                score = 0.0

                # Search in content
                if isinstance(memory.content, str):
                    if query_lower in memory.content.lower():
                        score += 1.0

                # Search in associations
                if memory_id in self.associations:
                    for assoc in self.associations[memory_id]:
                        if query_lower in assoc.association_type.lower():
                            score += 0.5

                # Boost by importance and recent access
                score += float(memory.importance) * 0.3
                time_factor = max(
                    0.0, 1.0 - (time.time() - memory.last_accessed) / 86400.0
                )  # 24 hours
                score += float(time_factor) * 0.2

                if score > 0:
                    scored_memories.append((memory, score))

            # Sort by score and limit results
            scored_memories.sort(key=lambda x: x[1], reverse=True)
            results = [memory for memory, score in scored_memories[:limit]]

            self.logger.debug(f"Search for '{query}' returned {len(results)} results")
            return results

        except Exception as e:
            self.logger.error(f"Failed to search memories: {e}")
            raise ComponentException(f"Memory search failed: {e}")

    async def forget_memory(self, memory_id: str) -> bool:
        """Remove a memory"""
        try:
            removed = False

            # Remove from all memory stores
            if memory_id in self.episodic_memory:
                del self.episodic_memory[memory_id]
                removed = True
            if memory_id in self.semantic_memory:
                del self.semantic_memory[memory_id]
                removed = True
            if memory_id in self.procedural_memory:
                del self.procedural_memory[memory_id]
                removed = True
            if memory_id in self.working_memory:
                del self.working_memory[memory_id]
                removed = True

            # Remove associations
            if memory_id in self.associations:
                del self.associations[memory_id]

            # Remove consolidation info
            if memory_id in self.consolidation_info:
                del self.consolidation_info[memory_id]

            # Remove from other memories' associations
            for associations in self.associations.values():
                associations[:] = [
                    assoc for assoc in associations if assoc.target_memory_id != memory_id
                ]

            if removed:
                self.logger.debug(f"Forgot memory {memory_id}")

            return removed

        except Exception as e:
            self.logger.error(f"Failed to forget memory {memory_id}: {e}")
            raise ComponentException(f"Memory forgetting failed: {e}")

    async def consolidate_memories(self) -> Dict[str, Any]:
        """Consolidate memories based on importance and usage"""
        try:
            consolidation_results = {"consolidated": 0, "promoted": 0, "demoted": 0, "forgotten": 0}

            current_time = time.time()

            # Process all memories with consolidation info
            for memory_id, consolidation in self.consolidation_info.items():
                memory = await self.retrieve_memory(memory_id)
                if not memory:
                    continue

                # Calculate consolidation score
                time_since_last = current_time - consolidation.last_consolidated
                usage_factor = memory.retrieval_count * 0.1
                importance_factor = memory.importance

                consolidation.consolidation_score = (
                    importance_factor
                    + usage_factor
                    - (time_since_last / 86400) * 0.1  # Decay over time
                )

                # Consolidation actions
                if consolidation.consolidation_score > self.consolidation_threshold:
                    # Promote to long-term memory
                    if memory.memory_type == MemoryType.WORKING:
                        memory.memory_type = MemoryType.LONG_TERM
                        self.episodic_memory[memory_id] = memory
                        del self.working_memory[memory_id]
                        consolidation_results["promoted"] += 1

                    consolidation.consolidation_count += 1
                    consolidation.last_consolidated = current_time
                    consolidation_results["consolidated"] += 1

                elif consolidation.consolidation_score < 0.2:
                    # Forget low-importance memories
                    if memory.importance < 0.3:
                        await self.forget_memory(memory_id)
                        consolidation_results["forgotten"] += 1

            self.logger.info(f"Memory consolidation completed: {consolidation_results}")
            return consolidation_results

        except Exception as e:
            self.logger.error(f"Failed to consolidate memories: {e}")
            raise ComponentException(f"Memory consolidation failed: {e}")

    async def get_memory_statistics(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        stats = {
            "episodic_count": len(self.episodic_memory),
            "semantic_count": len(self.semantic_memory),
            "procedural_count": len(self.procedural_memory),
            "working_count": len(self.working_memory),
            "total_memories": sum(
                [
                    len(self.episodic_memory),
                    len(self.semantic_memory),
                    len(self.procedural_memory),
                    len(self.working_memory),
                ]
            ),
            "total_associations": sum(len(assocs) for assocs in self.associations.values()),
            "consolidation_records": len(self.consolidation_info),
            "working_memory_capacity": self.working_memory_capacity,
            "working_memory_utilization": len(self.working_memory) / self.working_memory_capacity,
        }

        # Calculate average importance
        all_memories = {}
        all_memories.update(self.episodic_memory)
        all_memories.update(self.semantic_memory)
        all_memories.update(self.procedural_memory)
        all_memories.update(self.working_memory)

        if all_memories:
            stats["average_importance"] = sum(m.importance for m in all_memories.values()) / len(
                all_memories
            )
            stats["average_retrieval_count"] = sum(
                m.retrieval_count for m in all_memories.values()
            ) / len(all_memories)

        return stats

    # Private helper methods

    def _generate_memory_id(self, memory: Memory) -> str:
        """Generate a unique memory ID"""
        content_str = json.dumps(memory.content, sort_keys=True, default=str)
        content_hash = hashlib.md5(content_str.encode()).hexdigest()
        timestamp = str(int(time.time() * 1000))
        return f"mem_{content_hash[:8]}_{timestamp}"

    async def _create_memory_associations(self, memory_id: str, memory: Memory) -> None:
        """Create associations with existing memories"""
        try:
            # Simple association based on content similarity
            all_memories = {}
            all_memories.update(self.episodic_memory)
            all_memories.update(self.semantic_memory)
            all_memories.update(self.procedural_memory)

            for existing_id, existing_memory in all_memories.items():
                if existing_id == memory_id:
                    continue

                # Calculate similarity score (simplified)
                similarity = await self._calculate_similarity(memory, existing_memory)

                if similarity > 0.5:  # Association threshold
                    association = MemoryAssociation(
                        source_memory_id=memory_id,
                        target_memory_id=existing_id,
                        association_strength=similarity,
                        association_type="semantic_similarity",
                        created_at=time.time(),
                    )

                    self.associations[memory_id].append(association)

                    # Create bidirectional association
                    reverse_association = MemoryAssociation(
                        source_memory_id=existing_id,
                        target_memory_id=memory_id,
                        association_strength=similarity,
                        association_type="semantic_similarity",
                        created_at=time.time(),
                    )

                    self.associations[existing_id].append(reverse_association)

        except Exception as e:
            self.logger.warning(f"Failed to create associations for memory {memory_id}: {e}")

    async def _calculate_similarity(self, memory1: Memory, memory2: Memory) -> float:
        """Calculate similarity between two memories"""
        try:
            # Simple text-based similarity (can be enhanced with embeddings)
            if isinstance(memory1.content, str) and isinstance(memory2.content, str):
                content1 = memory1.content.lower()
                content2 = memory2.content.lower()

                # Jaccard similarity on word sets
                words1 = set(content1.split())
                words2 = set(content2.split())

                intersection = len(words1.intersection(words2))
                union = len(words1.union(words2))

                if union > 0:
                    return intersection / union

            return 0.0

        except Exception:
            return 0.0

    async def _evict_working_memory(self) -> None:
        """Evict least important memory from working memory"""
        if not self.working_memory:
            return

        # Find least important memory
        least_important_id = min(
            self.working_memory.keys(), key=lambda mid: self.working_memory[mid].importance
        )

        # Move to episodic memory instead of deleting
        memory = self.working_memory[least_important_id]
        memory.memory_type = MemoryType.EPISODIC
        self.episodic_memory[least_important_id] = memory
        del self.working_memory[least_important_id]

        self.logger.debug(f"Evicted working memory {least_important_id} to episodic memory")

    async def _consolidation_loop(self) -> None:
        """Background memory consolidation loop"""
        while self.is_initialized:
            try:
                await self.consolidate_memories()
                await asyncio.sleep(300)  # Consolidate every 5 minutes
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Consolidation loop error: {e}")
                await asyncio.sleep(60)

    async def _forgetting_loop(self) -> None:
        """Background memory forgetting loop"""
        while self.is_initialized:
            try:
                await self._apply_forgetting()
                await asyncio.sleep(3600)  # Apply forgetting every hour
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Forgetting loop error: {e}")
                await asyncio.sleep(300)

    async def _apply_forgetting(self) -> None:
        """Apply natural forgetting to memories"""
        current_time = time.time()

        for memory_id, consolidation in list(self.consolidation_info.items()):
            memory = await self.retrieve_memory(memory_id)
            if not memory:
                continue

            # Reduce importance based on time and retrieval count
            time_factor = (current_time - memory.last_accessed) / 86400  # Days
            forgetting_amount = self.forgetting_rate * time_factor

            if memory.retrieval_count == 0:  # Never accessed
                forgetting_amount *= 2

            memory.importance = max(0, memory.importance - forgetting_amount)

            # Forget if importance is too low
            if memory.importance < 0.1:
                await self.forget_memory(memory_id)
