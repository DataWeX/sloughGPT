"""
Domains Package - Clean domain-based architecture

Subdomains:
- chat: Chat generation and logging
- benchmark: Quality metrics
- companion: Personality management (from companion.py)
"""

from .chat.domain import ChatDomain, get_chat_domain
from .benchmark.domain import BenchmarkDomain, get_benchmark_domain

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

class ComponentException(Exception):
    pass


__all__ = [
    "ChatDomain",
    "get_chat_domain",
    "get_companion",
    "CompanionSystem",
    "BaseComponent",
    "ComponentException",
]
