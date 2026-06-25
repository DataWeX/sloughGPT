"""
Database types — shared enums, protocols, and config used by database/__init__.py.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


class DatabaseType(Enum):
    """Supported database backends."""

    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    REDIS = "redis"


@dataclass
class DatabaseConfig:
    """Configuration for a database connection."""

    db_type: DatabaseType = DatabaseType.SQLITE
    host: str = "localhost"
    port: int = 0
    database: str = "sloughgpt.db"
    username: Optional[str] = None
    password: Optional[str] = None
    ssl_enabled: bool = False
    pool_min: int = 1
    pool_max: int = 10
    connect_timeout: int = 10


class BaseComponent:
    """Minimal base component with lifecycle hooks."""

    def __init__(self, component_name: str):
        self.component_name = component_name
        self.is_initialized = False

    async def initialize(self) -> None:
        self.is_initialized = True

    async def shutdown(self) -> None:
        self.is_initialized = False


class ComponentException(Exception):
    """Exception raised by components."""


@runtime_checkable
class IDataRepository(Protocol):
    """Protocol for data repository implementations."""

    async def create(self, data: Dict[str, Any]) -> str:
        """Create a new record and return its ID."""
        ...

    async def read(self, record_id: str) -> Optional[Dict[str, Any]]:
        """Read a record by ID."""
        ...

    async def update(self, record_id: str, data: Dict[str, Any]) -> bool:
        """Update a record by ID."""
        ...

    async def delete(self, record_id: str) -> bool:
        """Delete a record by ID."""
        ...

    async def query(self, filters: Dict[str, Any], limit: int = 100) -> List[Dict[str, Any]]:
        """Query records matching filters."""
        ...


@runtime_checkable
class IDatabaseManager(Protocol):
    """Protocol for database manager implementations."""

    async def connect(self, config: DatabaseConfig) -> bool:
        """Connect to a database."""
        ...

    async def disconnect(self) -> None:
        """Disconnect from all databases."""
        ...

    async def get_repository(self, collection_name: str) -> IDataRepository:
        """Get a repository for a collection."""
        ...

    async def execute_query(self, query: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute a raw query."""
        ...

    async def get_database_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        ...


__all__ = [
    "DatabaseType",
    "DatabaseConfig",
    "BaseComponent",
    "ComponentException",
    "IDataRepository",
    "IDatabaseManager",
]
