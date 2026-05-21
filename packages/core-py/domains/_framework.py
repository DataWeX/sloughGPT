"""
Domain Framework — base classes and interfaces for the Clean Domain Architecture.

Provides BaseDomain, BaseComponent, ComponentException, DomainException,
and all protocol/interface classes expected by cognitive, infrastructure,
enterprise, and UI sub-packages.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum
import time


# ── Base Exceptions ──────────────────────────────────────────────────────────

class DomainException(Exception):
    """Base exception for all domain-level errors."""
    pass


class ComponentException(DomainException):
    """Exception raised when a managed component fails."""
    pass


# ── Base Domain (for cognitive/base.py) ──────────────────────────────────────

class BaseDomain(ABC):
    """Abstract base for all domains in the clean architecture.

    Subclasses implement _on_initialize and _on_shutdown.
    """

    def __init__(self, domain_name: str) -> None:
        self.domain_name = domain_name
        self.is_initialized = False
        self.components: Dict[str, Any] = {}

    async def initialize(self) -> None:
        self.is_initialized = True
        await self._on_initialize()

    async def shutdown(self) -> None:
        self.is_initialized = False
        await self._on_shutdown()

    async def _on_initialize(self) -> None:
        pass

    async def _on_shutdown(self) -> None:
        pass


# ── Base Component ──────────────────────────────────────────────────────────

class BaseComponent(ABC):
    """Abstract base for managed sub-components (config, cache, db, …)."""

    def __init__(self, component_name: str) -> None:
        self.component_name = component_name

    @abstractmethod
    async def initialize(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def shutdown(self) -> None:
        raise NotImplementedError


# ── Interfaces / Protocols ──────────────────────────────────────────────────

class IMemoryManager(ABC):
    @abstractmethod
    async def store_memory(self, memory: "Memory") -> str:
        raise NotImplementedError

    @abstractmethod
    async def retrieve_memory(self, memory_id: str) -> Optional["Memory"]:
        raise NotImplementedError

    @abstractmethod
    async def search_memories(self, query: str, limit: int = 10) -> List["Memory"]:
        raise NotImplementedError

    @abstractmethod
    async def forget_memory(self, memory_id: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    async def consolidate_memories(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    async def get_memory_statistics(self) -> Dict[str, Any]:
        raise NotImplementedError


class ICognitiveProcessor(ABC):
    @abstractmethod
    async def process_thought(self, thought: "Thought") -> "Thought":
        raise NotImplementedError

    @abstractmethod
    async def get_state(self) -> Dict[str, Any]:
        raise NotImplementedError


class IMetacognitiveMonitor(ABC):
    @abstractmethod
    async def monitor_thought_process(self, thoughts: List["Thought"]) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    async def generate_reflection(self, context: Dict[str, Any]) -> str:
        raise NotImplementedError


class IReasoningEngine(ABC):
    @abstractmethod
    async def reason(self, premise: str, context: Dict[str, Any]) -> str:
        raise NotImplementedError

    @abstractmethod
    async def get_reasoning_path(self) -> List[Dict[str, Any]]:
        raise NotImplementedError


class IDatabaseManager(ABC):
    @abstractmethod
    async def connect(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def disconnect(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        raise NotImplementedError


class IDataRepository(ABC):
    @abstractmethod
    async def create(self, data: Dict[str, Any]) -> Any:
        raise NotImplementedError

    @abstractmethod
    async def read(self, id: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    async def update(self, id: str, data: Dict[str, Any]) -> bool:
        raise NotImplementedError

    @abstractmethod
    async def delete(self, id: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    async def query(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        raise NotImplementedError


class IDeploymentManager(ABC):
    @abstractmethod
    async def deploy(self, config: Dict[str, Any]) -> str:
        raise NotImplementedError

    @abstractmethod
    async def rollback(self, deployment_id: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    async def get_status(self, deployment_id: str) -> Dict[str, Any]:
        raise NotImplementedError


class ICacheManager(ABC):
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        raise NotImplementedError

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        raise NotImplementedError

    @abstractmethod
    async def delete(self, key: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    async def clear(self) -> None:
        raise NotImplementedError


class IMonitoringService(ABC):
    @abstractmethod
    async def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        raise NotImplementedError

    @abstractmethod
    async def get_metrics(self, name: str, since: float) -> List[Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    async def check_health(self) -> Dict[str, Any]:
        raise NotImplementedError


class ICostOptimizer(ABC):
    @abstractmethod
    async def estimate_cost(self, resource: str, usage: Dict[str, Any]) -> float:
        raise NotImplementedError

    @abstractmethod
    async def optimize(self, current_config: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    async def get_savings_report(self) -> Dict[str, Any]:
        raise NotImplementedError


class IUserManager(ABC):
    @abstractmethod
    async def create_user(self, user_data: Dict[str, Any]) -> Any:
        raise NotImplementedError

    @abstractmethod
    async def get_user(self, user_id: str) -> Optional[Any]:
        raise NotImplementedError

    @abstractmethod
    async def update_user(self, user_id: str, updates: Dict[str, Any]) -> bool:
        raise NotImplementedError

    @abstractmethod
    async def delete_user(self, user_id: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    async def list_users(self, filters: Optional[Dict[str, Any]] = None, limit: int = 100) -> list:
        raise NotImplementedError


class IAuthenticationService(ABC):
    @abstractmethod
    async def authenticate(self, credentials: Dict[str, Any]) -> Optional[Any]:
        raise NotImplementedError

    @abstractmethod
    async def authorize(self, user: Any, resource: str, action: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    async def generate_token(self, user: Any) -> str:
        raise NotImplementedError

    @abstractmethod
    async def validate_token(self, token: str) -> Optional[Any]:
        raise NotImplementedError


# ── UI Interfaces ───────────────────────────────────────────────────────────

class IUIController(ABC):
    @abstractmethod
    async def render(self, request: "UIRequest") -> "UIResponse":
        raise NotImplementedError

    @abstractmethod
    async def handle_input(self, input_data: Any) -> None:
        raise NotImplementedError


class IWebInterface(ABC):
    @abstractmethod
    async def start_server(self, config: Dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    async def stop_server(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def handle_request(self, request: "UIRequest") -> "UIResponse":
        raise NotImplementedError


class IChatInterface(ABC):
    @abstractmethod
    async def send_message(self, message: str, channel: str) -> None:
        raise NotImplementedError

    @abstractmethod
    async def receive_messages(self, channel: str) -> List[Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    async def get_conversation_history(self, channel: str) -> List[Dict[str, Any]]:
        raise NotImplementedError


class UserRole(Enum):
    ADMIN = "admin"
    USER = "user"
    GUEST = "guest"
    MODERATOR = "moderator"


class SecurityLevel(Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class UIType(Enum):
    WEB = "web"
    CLI = "cli"
    API = "api"
    CHAT = "chat"
    DESKTOP = "desktop"


@dataclass
class UIRequest:
    endpoint: str
    method: str = "GET"
    headers: Dict[str, str] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    body: Any = None
    user: Optional[Any] = None


@dataclass
class UIResponse:
    status_code: int = 200
    body: Any = None
    headers: Dict[str, str] = field(default_factory=dict)
    content_type: str = "application/json"


class ResponseFormat(Enum):
    JSON = "json"
    HTML = "html"
    TEXT = "text"
    XML = "xml"
    STREAM = "stream"


@dataclass
class User:
    user_id: str
    username: str
    role: UserRole = UserRole.USER
    email: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


# ── Data Types ──────────────────────────────────────────────────────────────

class ThoughtType(Enum):
    ANALYTICAL = "analytical"
    CREATIVE = "creative"
    CRITICAL = "critical"
    STRATEGIC = "strategic"
    REFLECTIVE = "reflective"


@dataclass
class Thought:
    thought_id: str
    content: str
    thought_type: ThoughtType
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    retrieval_count: int = 0
    last_accessed: float = field(default_factory=time.time)


@dataclass
class Memory:
    content: Any
    memory_type: str = "episodic"
    importance: float = 0.5
    associations: List[str] = field(default_factory=list)
    retrieval_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryConfig:
    working_capacity: int = 50
    consolidation_threshold: float = 0.7
    forgetting_rate: float = 0.001


class DatabaseType(Enum):
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    REDIS = "redis"
    MONGODB = "mongodb"


@dataclass
class DatabaseConfig:
    db_type: DatabaseType = DatabaseType.SQLITE
    host: str = "localhost"
    port: int = 5432
    username: str = ""
    password: str = ""
    database: str = "sloughgpt"
    pool_size: int = 10
    timeout: int = 30


__all__ = [
    # Exceptions
    "DomainException",
    "ComponentException",
    # Base classes
    "BaseDomain",
    "BaseComponent",
    # Interfaces
    "IMemoryManager",
    "ICognitiveProcessor",
    "IMetacognitiveMonitor",
    "IReasoningEngine",
    "IDatabaseManager",
    "IDataRepository",
    "IDeploymentManager",
    "ICacheManager",
    "IMonitoringService",
    "ICostOptimizer",
    "IUserManager",
    "IAuthenticationService",
    # UI interfaces
    "IUIController",
    "IWebInterface",
    "IChatInterface",
    "UserRole",
    "SecurityLevel",
    "UIType",
    "UIRequest",
    "UIResponse",
    "ResponseFormat",
    "User",
    # Data types
    "ThoughtType",
    "Thought",
    "Memory",
    "MemoryConfig",
    "DatabaseType",
    "DatabaseConfig",
]
