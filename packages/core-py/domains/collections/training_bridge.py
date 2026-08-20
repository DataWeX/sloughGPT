from __future__ import annotations

import json
import hashlib
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from .sources import Record, Source
from .stores import Store, MemoryStore
from .collector import Collector


@dataclass
class TrainingDataConfig:
    block_size: int = 128
    separator: str = "\n"
    include_metadata: bool = False
    max_records: int | None = None
    deduplicate: bool = True
    min_length: int = 10
    max_length: int = 100000


class TrainingDataAdapter:
    def __init__(self, config: TrainingDataConfig | None = None):
        self.config = config or TrainingDataConfig()
        self._seen_hashes: set[str] = set()
        self.stats = {"total": 0, "accepted": 0, "deduplicated": 0, "too_short": 0, "too_long": 0}

    def records_to_text(self, records: list[Record]) -> str:
        parts = []
        for record in records:
            text = self._record_to_text(record)
            if text is not None:
                parts.append(text)
        return self.config.separator.join(parts)

    def _record_to_text(self, record: Record) -> str | None:
        self.stats["total"] += 1

        if self.config.min_length and len(record.content) < self.config.min_length:
            self.stats["too_short"] += 1
            return None

        if self.config.max_length and len(record.content) > self.config.max_length:
            self.stats["too_long"] += 1
            return None

        if self.config.deduplicate:
            h = hashlib.md5(record.content.encode("utf-8")).hexdigest()
            if h in self._seen_hashes:
                self.stats["deduplicated"] += 1
                return None
            self._seen_hashes.add(h)

        self.stats["accepted"] += 1
        return record.content

    def records_to_training_data(self, records: list[Record]) -> tuple[np.ndarray, dict[str, int], dict[int, str]]:
        text = self.records_to_text(records)
        chars = sorted(set(text))
        stoi = {c: i for i, c in enumerate(chars)}
        itos = {i: c for i, c in enumerate(chars)}
        data = np.array([stoi[c] for c in text], dtype=np.int64)
        return data, stoi, itos

    def records_to_text_file(self, records: list[Record], path: str) -> int:
        text = self.records_to_text(records)
        Path(path).write_text(text, encoding="utf-8")
        return len(text)

    def records_to_numpy(self, records: list[Record]) -> tuple[np.ndarray, int]:
        text = self.records_to_text(records)
        chars = sorted(set(text))
        stoi = {c: i for i, c in enumerate(chars)}
        data = np.array([stoi[c] for c in text], dtype=np.int64)
        return data, len(chars)

    def reset(self):
        self._seen_hashes.clear()
        self.stats = {"total": 0, "accepted": 0, "deduplicated": 0, "too_short": 0, "too_long": 0}


class RecordToTrainingSource:
    def __init__(self, records: list[Record], config: TrainingDataConfig | None = None):
        self._records = records
        self._adapter = TrainingDataAdapter(config)
        self.name = "training_records"

    def read(self) -> Iterator[Record]:
        return iter(self._records)


class TrainingDatasetBuilder:
    def __init__(self, config: TrainingDataConfig | None = None):
        self._config = config or TrainingDataConfig()
        self._adapter = TrainingDataAdapter(self._config)
        self._records: list[Record] = []

    def add_records(self, records: list[Record]) -> TrainingDatasetBuilder:
        self._records.extend(records)
        return self

    def add_from_source(self, source: Source) -> TrainingDatasetBuilder:
        for record in source.read():
            self._records.append(record)
        return self

    def add_from_file(self, path: str) -> TrainingDatasetBuilder:
        text = Path(path).read_text(encoding="utf-8")
        for line in text.split("\n"):
            if line.strip():
                self._records.append(Record(content=line))
        return self

    def add_from_text(self, text: str, separator: str = "\n") -> TrainingDatasetBuilder:
        for part in text.split(separator):
            if part.strip():
                self._records.append(Record(content=part))
        return self

    def build_text(self) -> str:
        return self._adapter.records_to_text(self._records)

    def build_numpy(self) -> tuple[np.ndarray, int]:
        return self._adapter.records_to_numpy(self._records)

    def build_dataset(self) -> tuple[np.ndarray, dict[str, int], dict[int, str]]:
        return self._adapter.records_to_training_data(self._records)

    def save_text(self, path: str) -> int:
        return self._adapter.records_to_text_file(self._records, path)

    def save_numpy(self, path: str) -> int:
        data, vocab_size = self.build_numpy()
        np.save(path, data)
        return len(data)

    @property
    def record_count(self) -> int:
        return len(self._records)

    @property
    def stats(self) -> dict:
        return self._adapter.stats

    def reset(self):
        self._records.clear()
        self._adapter.reset()


class CollectorTrainingBridge:
    def __init__(self, collector: Collector, config: TrainingDataConfig | None = None):
        self._collector = collector
        self._config = config or TrainingDataConfig()
        self._adapter = TrainingDataAdapter(self._config)
        self._records: list[Record] = []
        self.stats = {"collected": 0, "training_records": 0}

    def collect_and_prepare(self) -> tuple[np.ndarray, int]:
        self._collector.collect()
        records = list(self._collector.store.read_all())
        self._records.extend(records)
        self.stats["collected"] += len(records)
        data, vocab_size = self._adapter.records_to_numpy(self._records)
        self.stats["training_records"] = self._adapter.stats["accepted"]
        return data, vocab_size

    def collect_and_save_text(self, path: str) -> int:
        self._collector.collect()
        records = list(self._collector.store.read_all())
        self._records.extend(records)
        self.stats["collected"] += len(records)
        count = self._adapter.records_to_text_file(self._records, path)
        self.stats["training_records"] = self._adapter.stats["accepted"]
        return count

    def get_text(self) -> str:
        return self._adapter.records_to_text(self._records)

    def get_numpy(self) -> tuple[np.ndarray, int]:
        return self._adapter.records_to_numpy(self._records)

    @property
    def adapter(self) -> TrainingDataAdapter:
        return self._adapter

    def reset(self):
        self._records.clear()
        self._adapter.reset()
