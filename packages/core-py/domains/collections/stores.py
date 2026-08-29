from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from pathlib import Path
from typing import Iterator, Protocol, runtime_checkable

from .sources import Record

logger = logging.getLogger(__name__)


@runtime_checkable
class Store(Protocol):
    def write(self, record: Record) -> None: ...

    def read_all(self) -> Iterator[Record]: ...

    def count(self) -> int: ...


class FileStore:
    def __init__(self, path: str, name: str = ""):
        self.path = Path(path)
        self.name = name or f"file:{self.path.name}"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: Record) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            row = {"content": record.content, **record.metadata}
            f.write(json.dumps(row, default=str) + "\n")

    def read_all(self) -> Iterator[Record]:
        if not self.path.exists():
            return
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    content = data.pop("content", "")
                    yield Record(content=content, metadata=data)
                except json.JSONDecodeError as e:
                    logger.debug("Skipping malformed store line: %s", e)
                    continue

    def count(self) -> int:
        if not self.path.exists():
            return 0
        with open(self.path, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())

    def clear(self) -> None:
        if self.path.exists():
            self.path.write_text("")


class MemoryStore:
    def __init__(self, max_size: int = 10000, name: str = ""):
        self.name = name or "memory"
        self._buffer: deque[Record] = deque(maxlen=max_size)
        self._lock = threading.Lock()

    def write(self, record: Record) -> None:
        with self._lock:
            self._buffer.append(record)

    def read_all(self) -> Iterator[Record]:
        with self._lock:
            yield from list(self._buffer)

    def count(self) -> int:
        with self._lock:
            return len(self._buffer)

    def take(self, n: int = 1) -> list[Record]:
        with self._lock:
            result = []
            for _ in range(min(n, len(self._buffer))):
                result.append(self._buffer.popleft())
            return result

    def peek(self, n: int = 10) -> list[Record]:
        with self._lock:
            return list(self._buffer)[:n]

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()


class CallbackStore:
    def __init__(self, callback, name: str = ""):
        self.name = name or "callback"
        self._callback = callback
        self._count = 0

    def write(self, record: Record) -> None:
        self._callback(record)
        self._count += 1

    def read_all(self) -> Iterator[Record]:
        return iter([])

    def count(self) -> int:
        return self._count


class ChainedStore:
    def __init__(self, stores: list[Store], name: str = ""):
        self.name = name or "chained"
        self._stores = stores

    def write(self, record: Record) -> None:
        for store in self._stores:
            store.write(record)

    def read_all(self) -> Iterator[Record]:
        seen = set()
        for store in self._stores:
            for record in store.read_all():
                key = (record.content, id(store))
                if key not in seen:
                    seen.add(key)
                    yield record

    def count(self) -> int:
        return sum(s.count() for s in self._stores)


class StatsStore:
    def __init__(self, inner: Store, name: str = ""):
        self.name = name or f"stats:{inner.name}"
        self._inner = inner
        self.total_written = 0
        self.total_bytes = 0
        self.by_source: dict[str, int] = {}

    def write(self, record: Record) -> None:
        self._inner.write(record)
        self.total_written += 1
        self.total_bytes += len(record.content)
        source = record.metadata.get("source", "unknown")
        self.by_source[source] = self.by_source.get(source, 0) + 1

    def read_all(self) -> Iterator[Record]:
        return self._inner.read_all()

    def count(self) -> int:
        return self._inner.count()

    def stats(self) -> dict:
        return {
            "total_written": self.total_written,
            "total_bytes": self.total_bytes,
            "avg_bytes": self.total_bytes / max(self.total_written, 1),
            "by_source": dict(self.by_source),
            "inner_count": self._inner.count(),
        }
