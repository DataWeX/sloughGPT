"""
Domains Package - Clean domain-based architecture

Subdomains:
- chat: Chat generation and logging
- benchmark: Quality metrics
- companion: Personality management (from companion.py)
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Protocol, runtime_checkable

from .chat.domain import ChatDomain, get_chat_domain
from .benchmark.domain import BenchmarkDomain, get_benchmark_domain
from .infrastructure.errors import AppError

# Companion is in companion.py (not companion/)
from .companion import get_companion, CompanionSystem


class BaseComponent:
    def __init__(self, component_name: str):
        self.component_name = component_name
        self.is_initialized = False

    async def initialize(self) -> None:
        self.is_initialized = True

    async def shutdown(self) -> None:
        self.is_initialized = False

class ComponentException(AppError):
    """Component-level error — raised by domain components (cache, deployment, etc.)."""
    code: str = "E_COMPONENT"
    http_status: int = 500
    user_message: str = "A component error occurred."


# --- Cognitive domain shared types ---

@runtime_checkable
class ICognitiveProcessor(Protocol):
    """Protocol for cognitive processing."""
    async def process(self, input_data: Any) -> Any: ...

@runtime_checkable
class IMemoryManager(Protocol):
    """Protocol for memory management."""
    async def store(self, key: str, value: Any) -> None: ...
    async def recall(self, key: str) -> Any: ...

@runtime_checkable
class IMetacognitiveMonitor(Protocol):
    """Protocol for metacognitive monitoring."""
    async def monitor(self, state: Any) -> None: ...

@runtime_checkable
class IReasoningEngine(Protocol):
    """Protocol for reasoning."""
    async def reason(self, context: Any) -> Any: ...


class Memory:
    """Represents a memory entry."""
    def __init__(self, key: str, value: Any, memory_type: str = "episodic", importance: float = 0.5):
        self.key = key
        self.value = value
        self.content = value
        self.memory_type = memory_type
        self.importance = importance
        self.retrieval_count = 0
        self.last_accessed = time.time()
        self.metadata: Dict[str, Any] = {}


class ThoughtType:
    """Enum-like class for thought types."""
    PERCEPTION = "perception"
    REASONING = "reasoning"
    CREATIVITY = "creativity"
    REFLECTION = "reflection"
    DECISION = "decision"


class Thought:
    """Represents a cognitive thought."""
    def __init__(self, thought_id: str, content: str, thought_type: str = "reasoning",
                 confidence: float = 0.5, metadata: Optional[Dict[str, Any]] = None):
        self.thought_id = thought_id
        self.content = content
        self.thought_type = thought_type
        self.confidence = confidence
        self.metadata = metadata or {}


class BaseDomain:
    """Base domain class. Every domain extends this."""
    def __init__(self, domain_name: str) -> None:
        self.domain_name = domain_name

class DomainException(AppError):
    """Base exception for domain errors."""
    code: str = "E_DOMAIN"
    http_status: int = 400
    user_message: str = "A domain error occurred."



__all__ = [
    "ChatDomain",
    "get_chat_domain",
    "get_companion",
    "CompanionSystem",
    "BaseComponent",
    "ComponentException",
    "ICognitiveProcessor",
    "IMemoryManager",
    "IMetacognitiveMonitor",
    "IReasoningEngine",
    "Memory",
    "Thought",
    "ThoughtType",
    "BaseDomain",
    "DomainException",
]
