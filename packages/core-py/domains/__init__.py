"""
Domains Package - Clean domain-based architecture

Subdomains:
- chat: Chat generation and logging  
- benchmark: Quality metrics  
- companion: Personality management (from companion.py)
"""

from .chat.domain import ChatDomain, get_chat_domain

# Companion is in companion.py (not companion/)
from .companion import get_companion, CompanionSystem

# Core domain base classes and interfaces (used by cognitive, rag, etc.)
from .base import (
    BaseDomain,
    DomainException,
    ICognitiveProcessor,
    IMemoryManager,
    IMetacognitiveMonitor,
    IReasoningEngine,
    Thought,
    ThoughtType,
    Memory,
)

# Simple component base for creativity / rag components
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

# ── Re-export missing framework symbols needed by enterprise / infrastructure / UI sub‑modules ──
from ._framework import (
    IAuthenticationService,
    IChatInterface,
    IUIController,
    IWebInterface,
    ICacheManager,
    IDatabaseManager,
    IDeploymentManager,
    IDataRepository,
    ICostOptimizer,
    IMonitoringService,
    IUserManager,
    DatabaseConfig,
    DatabaseType,
    UIRequest,
    UIResponse,
    ResponseFormat,
    UIType,
    SecurityLevel,
    User,
    UserRole,
)


__all__ = [
    "ChatDomain",
    "get_chat_domain",
    "get_companion",
    "CompanionSystem",
    # Core domain exports
    "BaseDomain",
    "DomainException",
    "ICognitiveProcessor",
    "IMemoryManager",
    "IMetacognitiveMonitor",
    "IReasoningEngine",
    "Thought",
    "ThoughtType",
    "Memory",
    # Framework re-exports
    "IAuthenticationService",
    "IChatInterface",
    "IUIController",
    "IWebInterface",
    "ICacheManager",
    "IDatabaseManager",
    "IDeploymentManager",
    "IDataRepository",
    "ICostOptimizer",
    "IMonitoringService",
    "IUserManager",
    "DatabaseConfig",
    "DatabaseType",
    "UIRequest",
    "UIResponse",
    "ResponseFormat",
    "UIType",
    "SecurityLevel",
    "User",
    "UserRole",
]
