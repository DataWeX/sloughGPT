from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from .sources import Record


@runtime_checkable
class Filter(Protocol):
    def accept(self, record: Record) -> bool: ...


@dataclass
class LengthFilter:
    min_length: int = 10
    max_length: int = 100000

    def accept(self, record: Record) -> bool:
        return self.min_length <= len(record.content) <= self.max_length


@dataclass
class DedupFilter:
    _seen: set[str] = field(default_factory=set)

    def accept(self, record: Record) -> bool:
        h = hashlib.md5(record.content.encode("utf-8")).hexdigest()
        if h in self._seen:
            return False
        self._seen.add(h)
        return True

    def reset(self):
        self._seen.clear()


@dataclass
class KeywordFilter:
    keywords: list[str] = field(default_factory=list)
    mode: str = "include"

    def accept(self, record: Record) -> bool:
        if not self.keywords:
            return True
        content_lower = record.content.lower()
        has_keyword = any(kw.lower() in content_lower for kw in self.keywords)
        return has_keyword if self.mode == "include" else not has_keyword


@dataclass
class RegexFilter:
    pattern: str = ""
    mode: str = "include"

    def __post_init__(self):
        self._compiled = re.compile(self.pattern, re.IGNORECASE) if self.pattern else None

    def accept(self, record: Record) -> bool:
        if self._compiled is None:
            return True
        matches = bool(self._compiled.search(record.content))
        return matches if self.mode == "include" else not matches


@dataclass
class LanguageFilter:
    allowed_chars_ratio: float = 0.8

    def accept(self, record: Record) -> bool:
        if not record.content:
            return False
        ascii_count = sum(1 for c in record.content if ord(c) < 128)
        return (ascii_count / len(record.content)) >= self.allowed_chars_ratio


class FilterChain:
    def __init__(self, filters: list[Filter] | None = None):
        self.filters = filters or []
        self.stats = {"accepted": 0, "rejected": 0}

    def add(self, f: Filter):
        self.filters.append(f)
        return self

    def accept(self, record: Record) -> bool:
        for f in self.filters:
            if not f.accept(record):
                self.stats["rejected"] += 1
                return False
        self.stats["accepted"] += 1
        return True

    def filter_records(self, records) -> list[Record]:
        return [r for r in records if self.accept(r)]
