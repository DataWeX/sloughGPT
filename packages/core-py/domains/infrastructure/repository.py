"""
Data Repository — abstracted CRUD with TTL caching and migration support.

Provides a common interface for all data stores (sessions, datasets, feedback)
and replaces ad-hoc json.load / os.listdir / file I/O scattered everywhere.

Usage:
    from domains.infrastructure.repository import FileRepository

    repo = FileRepository[SessionData]("data/sessions", serializer=SessionData)
    repo.save("session_1", session)
    s = repo.get("session_1")
    all_sessions = repo.list()
    repo.delete("session_1")
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Generic, Optional, Protocol, TypeVar, runtime_checkable

logger = logging.getLogger("man.repository")

T = TypeVar("T")


# ── Serializer protocol ──


@runtime_checkable
class Serializer(Protocol[T]):
    """Protocol for serializing/deserializing data objects."""

    def serialize(self, obj: T) -> dict[str, Any]:
        ...

    def deserialize(self, data: dict[str, Any]) -> T:
        ...


class JsonSerializer(Generic[T]):
    """Generic JSON serializer for dataclass-like objects."""

    def __init__(self, cls: type[T], date_format: str = "iso"):
        self._cls = cls

    def serialize(self, obj: T) -> dict[str, Any]:
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if hasattr(obj, "_asdict"):
            return obj._asdict()
        if hasattr(obj, "__dataclass_fields__"):
            import dataclasses
            return dataclasses.asdict(obj)
        if isinstance(obj, dict):
            return dict(obj)
        return {"_value": str(obj)}

    def deserialize(self, data: dict[str, Any]) -> T:
        if hasattr(self._cls, "model_validate"):
            return self._cls.model_validate(data)
        if hasattr(self._cls, "__dataclass_fields__"):
            return self._cls(**data)
        return self._cls(data)


# ── Migrations ──


class Migration:
    """A single schema migration step."""

    def __init__(self, version: int, description: str, fn: Callable[[dict], dict]):
        self.version = version
        self.description = description
        self.fn = fn

    def apply(self, data: dict) -> dict:
        return self.fn(data)


class MigrationRunner:
    """Runs a chain of migrations to bring data up to the current schema."""

    def __init__(self, migrations: list[Migration] | None = None):
        self._migrations = sorted(migrations or [], key=lambda m: m.version)

    def add(self, migration: Migration):
        self._migrations.append(migration)
        self._migrations.sort(key=lambda m: m.version)

    @property
    def latest_version(self) -> int:
        if not self._migrations:
            return 0
        return self._migrations[-1].version

    def run(self, data: dict) -> dict:
        current_version = data.get("_schema_version", 0)
        for m in self._migrations:
            if m.version > current_version:
                try:
                    data = m.apply(data)
                    data["_schema_version"] = m.version
                    logger.debug("Migration v%d applied: %s", m.version, m.description)
                except Exception:
                    logger.exception("Migration v%d failed: %s", m.version, m.description)
                    raise
        return data


# ── Repository ──


class Repository(Protocol[T]):
    """Protocol for all repositories."""

    def get(self, key: str) -> T | None:
        ...

    def list(self) -> list[T]:
        ...

    def save(self, key: str, obj: T) -> bool:
        ...

    def delete(self, key: str) -> bool:
        ...

    def search(self, query: str, fields: list[str] | None = None) -> list[T]:
        ...


class FileRepository(Generic[T]):
    """JSON file-backed repository. Each record is one JSON file in a directory."""

    def __init__(
        self,
        directory: str | Path,
        *,
        serializer: Serializer[T] | type | None = None,
        migration_runner: MigrationRunner | None = None,
        key_suffix: str = ".json",
    ):
        self._directory = Path(directory)
        self._directory.mkdir(parents=True, exist_ok=True)
        self._suffix = key_suffix
        self._migrations = migration_runner or MigrationRunner()
        self._serializer = serializer
        self._cache: dict[str, _CacheEntry[T]] = {}
        self._cache_ttl: float = 0  # 0 = no cache
        self._lock = threading.Lock()

    # ── Serializer ──

    def _resolve_serializer(self) -> Serializer[T]:
        if isinstance(self._serializer, Serializer):
            return self._serializer
        if self._serializer is not None:
            return JsonSerializer(self._serializer)
        return JsonSerializer(dict)  # type: ignore

    # ── Cache ──

    def enable_cache(self, ttl_seconds: float = 5.0):
        self._cache_ttl = ttl_seconds

    def disable_cache(self):
        self._cache_ttl = 0
        self._cache.clear()

    def invalidate(self, key: str | None = None):
        if key is None:
            self._cache.clear()
        else:
            self._cache.pop(key, None)

    def _cache_get(self, key: str) -> T | None:
        entry = self._cache.get(key)
        if entry is None:
            return None
        if self._cache_ttl > 0 and time.monotonic() - entry.ts > self._cache_ttl:
            del self._cache[key]
            return None
        return entry.value

    def _cache_set(self, key: str, value: T):
        if self._cache_ttl > 0:
            self._cache[key] = _CacheEntry(value=value, ts=time.monotonic())

    # ── CRUD ──

    def get(self, key: str) -> T | None:
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        path = self._path(key)
        if not path.exists():
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            data = self._migrations.run(data)
            data.pop("_schema_version", None)
            obj = self._resolve_serializer().deserialize(data)
            self._cache_set(key, obj)
            return obj
        except Exception:
            logger.exception("Failed to read %s", path)
            return None

    def list(self) -> list[T]:
        results: list[T] = []
        for path in self._directory.glob(f"*{self._suffix}"):
            key = path.stem
            obj = self.get(key)
            if obj is not None:
                results.append(obj)
        return results

    def save(self, key: str, obj: T) -> bool:
        path = self._path(key)
        try:
            data = self._resolve_serializer().serialize(obj)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(data, f, indent=2, default=str)
            self._cache_set(key, obj)
            return True
        except Exception:
            logger.exception("Failed to save %s", key)
            return False

    def delete(self, key: str) -> bool:
        path = self._path(key)
        try:
            if path.exists():
                path.unlink()
            self.invalidate(key)
            return True
        except Exception:
            logger.exception("Failed to delete %s", key)
            return False

    def search(self, query: str, fields: list[str] | None = None) -> list[T]:
        q = query.lower()
        results: list[T] = []
        for obj in self.list():
            data = self._resolve_serializer().serialize(obj)
            for field in fields or list(data.keys()):
                val = str(data.get(field, ""))
                if q in val.lower():
                    results.append(obj)
                    break
        return results

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def count(self) -> int:
        return len(list(self._directory.glob(f"*{self._suffix}")))

    def keys(self) -> list[str]:
        return [p.stem for p in self._directory.glob(f"*{self._suffix}")]

    def _path(self, key: str) -> Path:
        return self._directory / f"{key}{self._suffix}"


@dataclass
class _CacheEntry(Generic[T]):
    value: T
    ts: float


# ── Caching wrapper ──


class CachedRepository(Generic[T]):
    """Wraps any Repository with an LRU-like TTL cache."""

    def __init__(self, inner: Repository[T], ttl: float = 5.0):
        self._inner = inner
        self._ttl = ttl
        self._cache: dict[str, tuple[T, float]] = {}
        self._list_cache: tuple[list[T], float] | None = None

    def get(self, key: str) -> T | None:
        cached = self._cache.get(key)
        if cached is not None and time.monotonic() - cached[1] < self._ttl:
            return cached[0]
        obj = self._inner.get(key)
        if obj is not None:
            self._cache[key] = (obj, time.monotonic())
        return obj

    def list(self) -> list[T]:
        if self._list_cache is not None and time.monotonic() - self._list_cache[1] < self._ttl:
            return self._list_cache[0]
        objs = self._inner.list()
        self._list_cache = (objs, time.monotonic())
        return objs

    def save(self, key: str, obj: T) -> bool:
        self._cache.pop(key, None)
        self._list_cache = None
        return self._inner.save(key, obj)

    def delete(self, key: str) -> bool:
        self._cache.pop(key, None)
        self._list_cache = None
        return self._inner.delete(key)

    def search(self, query: str, fields: list[str] | None = None) -> list[T]:
        return self._inner.search(query, fields)

    def invalidate(self):
        self._cache.clear()
        self._list_cache = None


# ── Memory repository (for testing) ──


class MemoryRepository(Generic[T]):
    """In-memory repository for testing."""

    def __init__(self):
        self._data: dict[str, T] = {}

    def get(self, key: str) -> T | None:
        return self._data.get(key)

    def list(self) -> list[T]:
        return list(self._data.values())

    def save(self, key: str, obj: T) -> bool:
        self._data[key] = obj
        return True

    def delete(self, key: str) -> bool:
        return self._data.pop(key, None) is not None

    def search(self, query: str, fields: list[str] | None = None) -> list[T]:
        q = query.lower()
        results: list[T] = []
        for obj in self._data.values():
            s = str(obj).lower()
            if q in s:
                results.append(obj)
        return results

    def clear(self):
        self._data.clear()
