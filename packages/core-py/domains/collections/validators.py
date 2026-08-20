from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Protocol, runtime_checkable

from .sources import Record


@dataclass
class Schema:
    required_fields: list[str] = field(default_factory=list)
    field_types: dict[str, type] = field(default_factory=dict)
    field_validators: dict[str, Callable[[Any], bool]] = field(default_factory=dict)
    max_content_length: int = 1000000
    min_content_length: int = 1

    def validate(self, record: Record) -> tuple[bool, str]:
        if len(record.content) > self.max_content_length:
            return False, f"Content too long: {len(record.content)} > {self.max_content_length}"
        if len(record.content) < self.min_content_length:
            return False, f"Content too short: {len(record.content)} < {self.min_content_length}"

        for field_name in self.required_fields:
            if field_name not in record.metadata:
                return False, f"Missing required field: {field_name}"

        for field_name, expected_type in self.field_types.items():
            if field_name in record.metadata:
                if not isinstance(record.metadata[field_name], expected_type):
                    return False, f"Field {field_name} has wrong type: expected {expected_type.__name__}"

        for field_name, validator in self.field_validators.items():
            if field_name in record.metadata:
                if not validator(record.metadata[field_name]):
                    return False, f"Field {field_name} failed validation"

        return True, ""


class DataValidator:
    def __init__(self, schema: Schema):
        self.schema = schema
        self.stats = {"valid": 0, "invalid": 0, "errors": {}}

    def validate(self, record: Record) -> bool:
        valid, error = self.schema.validate(record)
        if valid:
            self.stats["valid"] += 1
            return True
        self.stats["invalid"] += 1
        self.stats["errors"][error] = self.stats["errors"].get(error, 0) + 1
        return False

    def validate_all(self, records: list[Record]) -> list[Record]:
        return [r for r in records if self.validate(r)]

    def reset_stats(self):
        self.stats = {"valid": 0, "invalid": 0, "errors": {}}


@dataclass
class EnrichmentRule:
    key: str
    value: Any = None
    value_fn: Callable[[Record], Any] | None = None
    overwrite: bool = False

    def apply(self, record: Record) -> Record:
        if self.key in record.metadata and not self.overwrite:
            return record
        if self.value_fn is not None:
            record.metadata[self.key] = self.value_fn(record)
        else:
            record.metadata[self.key] = self.value
        return record


class DataEnricher:
    def __init__(self, rules: list[EnrichmentRule] | None = None):
        self.rules = rules or []
        self.stats = {"enriched": 0, "skipped": 0}

    def add_rule(self, rule: EnrichmentRule) -> DataEnricher:
        self.rules.append(rule)
        return self

    def enrich(self, record: Record) -> Record:
        modified = False
        for rule in self.rules:
            before = len(record.metadata)
            rule.apply(record)
            if len(record.metadata) > before or rule.overwrite:
                modified = True
        if modified:
            self.stats["enriched"] += 1
        else:
            self.stats["skipped"] += 1
        return record

    def enrich_all(self, records: list[Record]) -> list[Record]:
        return [self.enrich(r) for r in records]

    def reset_stats(self):
        self.stats = {"enriched": 0, "skipped": 0}


class RateLimiter:
    def __init__(self, max_per_second: float = 10.0, burst_size: int = 1):
        self.max_per_second = max_per_second
        self.burst_size = burst_size
        self._tokens = burst_size
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()
        self.stats = {"allowed": 0, "delayed": 0}

    def _refill(self):
        now = time.monotonic()
        elapsed = now - self._last_refill
        new_tokens = elapsed * self.max_per_second
        self._tokens = min(self.burst_size, self._tokens + new_tokens)
        self._last_refill = now

    def acquire(self) -> bool:
        with self._lock:
            self._refill()
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                self.stats["allowed"] += 1
                return True
            self.stats["delayed"] += 1
            return False

    def wait(self, timeout: float = 5.0) -> bool:
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            if self.acquire():
                return True
            time.sleep(0.01)
        return False

    def reset(self):
        with self._lock:
            self._tokens = self.burst_size
            self._last_refill = time.monotonic()


@runtime_checkable
class SourceAdapter(Protocol):
    def __call__(self) -> Iterator[Record]: ...


@runtime_checkable
class StoreAdapter(Protocol):
    def __call__(self, record: Record) -> None: ...


class CallableSource:
    def __init__(self, fn: SourceAdapter, name: str = ""):
        self._fn = fn
        self.name = name or getattr(fn, "__name__", "callable_source")

    def read(self) -> Iterator[Record]:
        return self._fn()


class CallableStore:
    def __init__(self, fn: StoreAdapter, name: str = ""):
        self._fn = fn
        self.name = name or getattr(fn, "__name__", "callable_store")
        self._count = 0

    def write(self, record: Record) -> None:
        self._fn(record)
        self._count += 1

    def read_all(self) -> Iterator[Record]:
        return iter([])

    def count(self) -> int:
        return self._count


class CollectorRunner:
    def __init__(self):
        self._collectors: dict[str, Any] = {}
        self._stats: dict[str, dict] = {}

    def add(self, name: str, collector: Any) -> CollectorRunner:
        self._collectors[name] = collector
        self._stats[name] = {"runs": 0, "total_collected": 0, "errors": 0, "last_run": None}
        return self

    def remove(self, name: str) -> bool:
        if name in self._collectors:
            del self._collectors[name]
            del self._stats[name]
            return True
        return False

    def run(self, name: str) -> int:
        if name not in self._collectors:
            return 0
        try:
            count = self._collectors[name].collect()
            self._stats[name]["runs"] += 1
            self._stats[name]["total_collected"] += count
            self._stats[name]["last_run"] = time.time()
            return count
        except Exception:
            self._stats[name]["errors"] += 1
            return 0

    def run_all(self) -> dict[str, int]:
        results = {}
        for name in self._collectors:
            results[name] = self.run(name)
        return results

    def run_threaded(self) -> dict[str, int]:
        results = {}
        threads = []
        lock = threading.Lock()

        def run_one(name):
            count = self.run(name)
            with lock:
                results[name] = count

        for name in self._collectors:
            t = threading.Thread(target=run_one, args=(name,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        return results

    def stats(self) -> dict:
        return dict(self._stats)

    def list(self) -> list[str]:
        return list(self._collectors.keys())

    def get(self, name: str):
        return self._collectors.get(name)
