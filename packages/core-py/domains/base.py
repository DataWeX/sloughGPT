"""Core domain base classes and simple interfaces.

These are deliberately lightweight – they provide just enough structure for the
cognitive, rag, and other higher‑level domains to import without pulling in heavy
dependencies. Real implementations live in the concrete sub‑domains.
"""

from __future__ import annotations

import abc
from typing import Any, Dict


class DomainException(Exception):
    """Base exception for domain‑level errors."""


class BaseDomain(abc.ABC):
    """Minimal abstract base for all domains.

    Sub‑classes should implement ``_on_initialize`` and ``_on_shutdown`` which
    are called by ``initialize``/``shutdown`` helpers.
    """

    def __init__(self, name: str):
        self.domain_name = name
        self.components: Dict[str, Any] = {}
        self.is_initialized: bool = False

    @abc.abstractmethod
    async def _on_initialize(self) -> None:  # pragma: no cover – implementation provided by sub‑domains
        ...

    @abc.abstractmethod
    async def _on_shutdown(self) -> None:  # pragma: no cover – implementation provided by sub‑domains
        ...

    async def initialize(self) -> None:
        await self._on_initialize()
        self.is_initialized = True

    async def shutdown(self) -> None:
        await self._on_shutdown()
        self.is_initialized = False


# ---------- Simple interface placeholders used by cognitive domain ----------

class ICognitiveProcessor(abc.ABC):
    @abc.abstractmethod
    async def process_thought(self, thought: Any) -> Any:
        ...


class IMemoryManager(abc.ABC):
    @abc.abstractmethod
    async def initialize(self) -> None:
        ...


class IMetacognitiveMonitor(abc.ABC):
    @abc.abstractmethod
    async def initialize(self) -> None:
        ...


class IReasoningEngine(abc.ABC):
    @abc.abstractmethod
    async def initialize(self) -> None:
        ...


# ---------- Lightweight data structures used in cognitive pipelines ----------

class Thought:
    def __init__(self, thought_id: str, content: str, thought_type: Any, confidence: float, metadata: dict):
        self.thought_id = thought_id
        self.content = content
        self.thought_type = thought_type
        self.confidence = confidence
        self.metadata = metadata


class ThoughtType(str):
    """Simple wrapper so ``ThoughtType`` can be used as an enum‑like string."""


class Memory(dict):
    """A thin wrapper around ``dict`` for memory storage; kept for type safety."""
