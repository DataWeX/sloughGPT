"""
Database Manager Implementation

This module provides comprehensive database management capabilities
supporting multiple database types with unified interfaces.
"""

import asyncio
import json
import logging
import sqlite3
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ...__init__ import (
    BaseComponent,
    ComponentException,
    DatabaseConfig,
    DatabaseType,
    IDatabaseManager,
    IDataRepository,
)


@dataclass
class DatabaseConnection:
    """Database connection information"""

    connection_id: str
    db_type: DatabaseType
    connection: Any
    is_active: bool
    created_at: float
    last_used: float


class BaseRepository(IDataRepository, ABC):
    """Base repository implementation"""

    def __init__(self, collection_name: str, connection: Any):
        self.collection_name = collection_name
        self.connection = connection
        self.logger = logging.getLogger(f"sloughgpt.repository.{collection_name}")

    @abstractmethod
    async def create(self, data: Dict[str, Any]) -> str:
        """Create a new record"""
        pass

    @abstractmethod
    async def read(self, record_id: str) -> Optional[Dict[str, Any]]:
        """Read a record by ID"""
        pass

    @abstractmethod
    async def update(self, record_id: str, data: Dict[str, Any]) -> bool:
        """Update a record"""
        pass

    @abstractmethod
    async def delete(self, record_id: str) -> bool:
        """Delete a record"""
        pass

    @abstractmethod
    async def query(self, filters: Dict[str, Any], limit: int = 100) -> List[Dict[str, Any]]:
        """Query records with filters"""
        pass


class SQLiteRepository(BaseRepository):
    """SQLite repository implementation"""

    async def create(self, data: Dict[str, Any]) -> str:
        """Create a new record in SQLite"""
        try:
            record_id = str(uuid.uuid4())
            data["id"] = record_id
            data["created_at"] = time.time()
            data["updated_at"] = time.time()

            # Convert data to JSON for storage
            json_data = json.dumps(data)

            cursor = self.connection.cursor()
            cursor.execute(
                f"INSERT INTO {self.collection_name} "
                f"(id, data, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (record_id, json_data, data["created_at"], data["updated_at"]),
            )
            self.connection.commit()

            self.logger.debug(f"Created record {record_id} in {self.collection_name}")
            return record_id

        except Exception as e:
            self.logger.error(f"Failed to create record in {self.collection_name}: {e}")
            raise ComponentException(f"Record creation failed: {e}")

    async def read(self, record_id: str) -> Optional[Dict[str, Any]]:
        """Read a record by ID from SQLite"""
        try:
            cursor = self.connection.cursor()
            cursor.execute(f"SELECT data FROM {self.collection_name} WHERE id = ?", (record_id,))
            result = cursor.fetchone()

            if result:
                data = json.loads(result[0])
                self.logger.debug(f"Read record {record_id} from {self.collection_name}")
                return data

            return None

        except Exception as e:
            self.logger.error(f"Failed to read record {record_id} from {self.collection_name}: {e}")
            raise ComponentException(f"Record read failed: {e}")

    async def update(self, record_id: str, data: Dict[str, Any]) -> bool:
        """Update a record in SQLite"""
        try:
            data["updated_at"] = time.time()
            json_data = json.dumps(data)

            cursor = self.connection.cursor()
            cursor.execute(
                f"UPDATE {self.collection_name} SET data = ?, updated_at = ? WHERE id = ?",
                (json_data, data["updated_at"], record_id),
            )
            self.connection.commit()

            updated = cursor.rowcount > 0
            if updated:
                self.logger.debug(f"Updated record {record_id} in {self.collection_name}")

            return bool(updated)

        except Exception as e:
            self.logger.error(f"Failed to update record {record_id} in {self.collection_name}: {e}")
            raise ComponentException(f"Record update failed: {e}")

    async def delete(self, record_id: str) -> bool:
        """Delete a record from SQLite"""
        try:
            cursor = self.connection.cursor()
            cursor.execute(f"DELETE FROM {self.collection_name} WHERE id = ?", (record_id,))
            self.connection.commit()

            deleted = cursor.rowcount > 0
            if deleted:
                self.logger.debug(f"Deleted record {record_id} from {self.collection_name}")

            return bool(deleted)

        except Exception as e:
            self.logger.error(
                f"Failed to delete record {record_id} from {self.collection_name}: {e}"
            )
            raise ComponentException(f"Record deletion failed: {e}")

    async def query(self, filters: Dict[str, Any], limit: int = 100) -> List[Dict[str, Any]]:
        """Query records from SQLite"""
        try:
            cursor = self.connection.cursor()

            # Build query based on filters
            if filters:
                # Simple implementation - in production, use proper query building
                where_clause = " AND ".join(
                    [f"JSON_EXTRACT(data, '$.{key}') = ?" for key in filters.keys()]
                )
                values = list(filters.values())
                cursor.execute(
                    f"SELECT data FROM {self.collection_name} WHERE {where_clause} LIMIT ?",
                    values + [limit],
                )
            else:
                cursor.execute(f"SELECT data FROM {self.collection_name} LIMIT ?", (limit,))

            results = []
            for row in cursor.fetchall():
                data = json.loads(row[0])
                results.append(data)

            self.logger.debug(f"Queried {len(results)} records from {self.collection_name}")
            return results

        except Exception as e:
            self.logger.error(f"Failed to query {self.collection_name}: {e}")
            raise ComponentException(f"Record query failed: {e}")


class DatabaseManager(BaseComponent, IDatabaseManager):
    """Advanced database management system"""

    def __init__(self):
        super().__init__("database_manager")
        self.logger = logging.getLogger(f"sloughgpt.{self.component_name}")

        # Database connections
        self.connections: Dict[str, DatabaseConnection] = {}
        self.repositories: Dict[str, BaseRepository] = {}

        # Configuration
        self.default_config = DatabaseConfig(
            db_type=DatabaseType.SQLITE,
            host="localhost",
            port=0,
            database="sloughgpt.db",
            username=None,
            password=None,
            ssl_enabled=False,
        )

        # Connection pool management
        self.connection_pools: Dict[DatabaseType, List[Any]] = {}
        self.pool_locks: Dict[DatabaseType, asyncio.Lock] = {}

        # Statistics
        stats_dict: Dict[str, Any] = {
            "total_connections": 0,
            "active_connections": 0,
            "total_queries": 0,
            "query_errors": 0,
            "connection_errors": 0,
            "connections_by_type": {},
            "repositories": [],
            "repository_count": 0,
        }
        self.stats = stats_dict

        self.is_initialized = False

    async def initialize(self) -> None:
        """Initialize the database manager"""
        try:
            self.logger.info("Initializing Database Manager...")

            # Initialize connection pools
            await self._initialize_connection_pools()

            # Connect to default database
            await self.connect(self.default_config)

            # Start connection monitoring
            asyncio.create_task(self._connection_monitoring_loop())

            self.is_initialized = True
            self.logger.info("Database Manager initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize Database Manager: {e}")
            raise ComponentException(f"Database Manager initialization failed: {e}")

    async def shutdown(self) -> None:
        """Shutdown the database manager"""
        try:
            self.logger.info("Shutting down Database Manager...")

            # Close all connections
            for connection_id in list(self.connections.keys()):
                await self._close_connection(connection_id)

            # Clear pools
            self.connection_pools.clear()
            self.pool_locks.clear()

            self.is_initialized = False
            self.logger.info("Database Manager shutdown successfully")

        except Exception as e:
            self.logger.error(f"Failed to shutdown Database Manager: {e}")
            raise ComponentException(f"Database Manager shutdown failed: {e}")

    async def connect(self, config: DatabaseConfig) -> bool:
        """Connect to database"""
        try:
            connection_id = str(uuid.uuid4())

            if config.db_type == DatabaseType.SQLITE:
                connection = await self._connect_sqlite(config)
            elif config.db_type == DatabaseType.POSTGRESQL:
                connection = await self._connect_postgresql(config)
            elif config.db_type == DatabaseType.REDIS:
                connection = await self._connect_redis(config)
            else:
                raise ComponentException(f"Unsupported database type: {config.db_type}")

            # Store connection
            db_connection = DatabaseConnection(
                connection_id=connection_id,
                db_type=config.db_type,
                connection=connection,
                is_active=True,
                created_at=time.time(),
                last_used=time.time(),
            )

            self.connections[connection_id] = db_connection
            self.stats["total_connections"] += 1
            self.stats["active_connections"] += 1

            self.logger.info(
                f"Connected to {config.db_type.value} database with ID: {connection_id}"
            )
            return True

        except Exception as e:
            self.logger.error(f"Failed to connect to database: {e}")
            self.stats["connection_errors"] += 1
            raise ComponentException(f"Database connection failed: {e}")

    async def disconnect(self) -> None:
        """Disconnect from database"""
        try:
            # Close all connections
            for connection_id in list(self.connections.keys()):
                await self._close_connection(connection_id)

            self.logger.info("Disconnected from all databases")

        except Exception as e:
            self.logger.error(f"Failed to disconnect from database: {e}")
            raise ComponentException(f"Database disconnection failed: {e}")

    async def get_repository(self, collection_name: str) -> IDataRepository:
        """Get a repository for a collection"""
        try:
            # Check if repository already exists
            if collection_name in self.repositories:
                return self.repositories[collection_name]

            # Get a connection for the repository
            connection = await self._get_connection()
            if not connection:
                raise ComponentException("No available database connections")

            # Create repository based on database type
            if connection.db_type == DatabaseType.SQLITE:
                repository = SQLiteRepository(collection_name, connection.connection)
                await self._ensure_table_exists(collection_name, connection.connection)
            else:
                raise ComponentException(f"Repository not implemented for {connection.db_type}")

            # Store repository
            self.repositories[collection_name] = repository

            self.logger.debug(f"Created repository for collection: {collection_name}")
            return repository

        except Exception as e:
            self.logger.error(f"Failed to get repository for {collection_name}: {e}")
            raise ComponentException(f"Repository creation failed: {e}")

    async def execute_query(self, query: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute a raw query"""
        try:
            # Get a connection
            connection = await self._get_connection()
            if not connection:
                raise ComponentException("No available database connections")

            # Execute query based on database type
            if connection.db_type == DatabaseType.SQLITE:
                results = await self._execute_sqlite_query(connection.connection, query, params)
            else:
                raise ComponentException(
                    f"Query execution not implemented for {connection.db_type}"
                )

            self.stats["total_queries"] += 1
            return results

        except Exception as e:
            self.logger.error(f"Failed to execute query: {e}")
            self.stats["query_errors"] += 1
            raise ComponentException(f"Query execution failed: {e}")

    async def get_database_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        stats = self.stats.copy()

        # Add connection details
        connections_by_type: Dict[str, int] = {}
        for connection in self.connections.values():
            db_type = connection.db_type.value
            current_count = connections_by_type.get(db_type, 0)
            connections_by_type[db_type] = current_count + 1

        stats["connections_by_type"] = connections_by_type  # type: ignore

        # Add repository details
        stats["repositories"] = list(self.repositories.keys())  # type: ignore
        stats["repository_count"] = len(self.repositories)

        return stats

    # Private helper methods

    async def _initialize_connection_pools(self) -> None:
        """Initialize connection pools"""
        for db_type in DatabaseType:
            self.connection_pools[db_type] = []
            self.pool_locks[db_type] = asyncio.Lock()

    async def _connect_sqlite(self, config: DatabaseConfig) -> sqlite3.Connection:
        """Connect to SQLite database"""
        connection = sqlite3.connect(config.database, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        return connection

    async def _connect_postgresql(self, config: DatabaseConfig) -> Any:
        """Connect to PostgreSQL database"""
        # Placeholder for PostgreSQL connection
        raise ComponentException("PostgreSQL connection not implemented")

    async def _connect_redis(self, config: DatabaseConfig) -> Any:
        """Connect to Redis database"""
        # Placeholder for Redis connection
        raise ComponentException("Redis connection not implemented")

    async def _get_connection(self) -> Optional[DatabaseConnection]:
        """Get an available connection"""
        for connection in self.connections.values():
            if connection.is_active:
                connection.last_used = time.time()
                return connection
        return None

    async def _close_connection(self, connection_id: str) -> None:
        """Close a specific connection"""
        if connection_id not in self.connections:
            return

        connection = self.connections[connection_id]

        try:
            if connection.db_type == DatabaseType.SQLITE:
                connection.connection.close()
            # Add other database types as needed

            connection.is_active = False
            del self.connections[connection_id]
            self.stats["active_connections"] -= 1

            self.logger.debug(f"Closed connection {connection_id}")

        except Exception as e:
            self.logger.error(f"Failed to close connection {connection_id}: {e}")

    async def _ensure_table_exists(self, table_name: str, connection: sqlite3.Connection) -> None:
        """Ensure table exists in SQLite"""
        cursor = connection.cursor()
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """)
        connection.commit()

    async def _execute_sqlite_query(
        self, connection: sqlite3.Connection, query: str, params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Execute SQLite query"""
        cursor = connection.cursor()
        cursor.execute(query, params)

        results = []
        for row in cursor.fetchall():
            results.append(dict(row))

        return results

    async def _connection_monitoring_loop(self) -> None:
        """Background connection monitoring loop"""
        while self.is_initialized:
            try:
                await self._monitor_connections()
                await asyncio.sleep(60)  # Monitor every minute
            except Exception as e:
                self.logger.error(f"Connection monitoring error: {e}")
                await asyncio.sleep(10)

    async def _monitor_connections(self) -> None:
        """Monitor connection health"""
        current_time = time.time()
        timeout_threshold = 300  # 5 minutes

        for connection_id, connection in list(self.connections.items()):
            # Check for idle connections
            if current_time - connection.last_used > timeout_threshold:
                self.logger.debug(f"Closing idle connection {connection_id}")
                await self._close_connection(connection_id)


__all__ = ["DatabaseConnection", "BaseRepository", "SQLiteRepository", "DatabaseManager"]
