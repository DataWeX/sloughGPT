"""Tests for collections.training_bridge — training data preparation and dataset building."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from domains.collections.training_bridge import (
    TrainingDataConfig,
    TrainingDataAdapter,
    RecordToTrainingSource,
    TrainingDatasetBuilder,
    CollectorTrainingBridge,
)
from domains.collections.sources import Record
from domains.collections.collector import Collector


# ── Helpers ───────────────────────────────────────────────────────────────


class NameSource:
    def __init__(self, name: str, records: list[Record]):
        self.name = name
        self._records = records

    def read(self):
        return iter(self._records)


class ListStore:
    def __init__(self):
        self._items: list[Record] = []

    def write(self, record: Record):
        self._items.append(record)

    def count(self) -> int:
        return len(self._items)

    def read_all(self):
        return iter(self._items)


# ── TrainingDataConfig ────────────────────────────────────────────────────


class TestTrainingDataConfig:

    def test_defaults(self):
        cfg = TrainingDataConfig()
        assert cfg.block_size == 128
        assert cfg.separator == "\n"
        assert cfg.include_metadata is False
        assert cfg.deduplicate is True
        assert cfg.min_length == 10
        assert cfg.max_length == 100000

    def test_custom(self):
        cfg = TrainingDataConfig(block_size=64, min_length=5)
        assert cfg.block_size == 64
        assert cfg.min_length == 5


# ── TrainingDataAdapter ───────────────────────────────────────────────────


class TestTrainingDataAdapter:

    def test_records_to_text(self):
        adapter = TrainingDataAdapter(TrainingDataConfig(min_length=1))
        records = [Record(content="hello"), Record(content="world")]
        text = adapter.records_to_text(records)
        assert text == "hello\nworld"

    def test_filters_too_short(self):
        adapter = TrainingDataAdapter(TrainingDataConfig(min_length=10))
        records = [Record(content="hi"), Record(content="long enough text")]
        text = adapter.records_to_text(records)
        assert text == "long enough text"
        assert adapter.stats["too_short"] == 1

    def test_filters_too_long(self):
        adapter = TrainingDataAdapter(TrainingDataConfig(min_length=1, max_length=5))
        records = [Record(content="short"), Record(content="this is way too long")]
        text = adapter.records_to_text(records)
        assert text == "short"
        assert adapter.stats["too_long"] == 1

    def test_deduplication(self):
        adapter = TrainingDataAdapter(TrainingDataConfig(min_length=1))
        records = [Record(content="hello"), Record(content="hello")]
        text = adapter.records_to_text(records)
        assert text == "hello"
        assert adapter.stats["deduplicated"] == 1

    def test_dedup_disabled(self):
        cfg = TrainingDataConfig(min_length=1, deduplicate=False)
        adapter = TrainingDataAdapter(cfg)
        records = [Record(content="hello"), Record(content="hello")]
        text = adapter.records_to_text(records)
        assert text == "hello\nhello"
        assert adapter.stats["deduplicated"] == 0

    def test_records_to_training_data(self):
        adapter = TrainingDataAdapter(TrainingDataConfig(min_length=1))
        records = [Record(content="abc")]
        data, stoi, itos = adapter.records_to_training_data(records)
        assert isinstance(data, np.ndarray)
        assert len(stoi) == 3
        assert len(itos) == 3

    def test_records_to_text_file(self, tmp_path):
        adapter = TrainingDataAdapter(TrainingDataConfig(min_length=1))
        records = [Record(content="hello"), Record(content="world")]
        out = tmp_path / "out.txt"
        chars = adapter.records_to_text_file(records, str(out))
        assert chars > 0
        assert out.read_text() == "hello\nworld"

    def test_records_to_numpy(self):
        adapter = TrainingDataAdapter(TrainingDataConfig(min_length=1))
        records = [Record(content="abc")]
        data, vocab_size = adapter.records_to_numpy(records)
        assert isinstance(data, np.ndarray)
        assert vocab_size == 3

    def test_reset(self):
        adapter = TrainingDataAdapter(TrainingDataConfig(min_length=1))
        adapter.records_to_text([Record(content="test")])
        adapter.reset()
        assert adapter.stats["total"] == 0
        assert len(adapter._seen_hashes) == 0

    def test_empty_records(self):
        adapter = TrainingDataAdapter()
        text = adapter.records_to_text([])
        assert text == ""


# ── RecordToTrainingSource ────────────────────────────────────────────────


class TestRecordToTrainingSource:

    def test_read_returns_records(self):
        records = [Record(content="a"), Record(content="b")]
        src = RecordToTrainingSource(records)
        assert src.name == "training_records"
        result = list(src.read())
        assert len(result) == 2


# ── TrainingDatasetBuilder ────────────────────────────────────────────────


class TestTrainingDatasetBuilder:

    def test_add_records(self):
        builder = TrainingDatasetBuilder(TrainingDataConfig(min_length=1))
        builder.add_records([Record(content="a"), Record(content="b")])
        assert builder.record_count == 2

    def test_add_from_source(self):
        src = NameSource("s", [Record(content="a"), Record(content="b")])
        builder = TrainingDatasetBuilder(TrainingDataConfig(min_length=1))
        builder.add_from_source(src)
        assert builder.record_count == 2

    def test_add_from_text(self):
        builder = TrainingDatasetBuilder(TrainingDataConfig(min_length=1))
        builder.add_from_text("line1\nline2\nline3")
        assert builder.record_count == 3

    def test_add_from_file(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("line1\nline2\nline3")
        builder = TrainingDatasetBuilder(TrainingDataConfig(min_length=1))
        builder.add_from_file(str(f))
        assert builder.record_count == 3

    def test_build_text(self):
        builder = TrainingDatasetBuilder(TrainingDataConfig(min_length=1))
        builder.add_records([Record(content="hello"), Record(content="world")])
        text = builder.build_text()
        assert text == "hello\nworld"

    def test_build_numpy(self):
        builder = TrainingDatasetBuilder(TrainingDataConfig(min_length=1))
        builder.add_records([Record(content="abc")])
        data, vocab = builder.build_numpy()
        assert isinstance(data, np.ndarray)
        assert vocab == 3

    def test_build_dataset(self):
        builder = TrainingDatasetBuilder(TrainingDataConfig(min_length=1))
        builder.add_records([Record(content="abc")])
        data, stoi, itos = builder.build_dataset()
        assert isinstance(data, np.ndarray)
        assert "a" in stoi

    def test_save_text(self, tmp_path):
        builder = TrainingDatasetBuilder(TrainingDataConfig(min_length=1))
        builder.add_records([Record(content="hello")])
        out = tmp_path / "out.txt"
        chars = builder.save_text(str(out))
        assert chars > 0
        assert out.read_text() == "hello"

    def test_save_numpy(self, tmp_path):
        builder = TrainingDatasetBuilder(TrainingDataConfig(min_length=1))
        builder.add_records([Record(content="abc")])
        out = tmp_path / "out.npy"
        length = builder.save_numpy(str(out))
        assert length == 3
        assert out.exists()

    def test_reset(self):
        builder = TrainingDatasetBuilder(TrainingDataConfig(min_length=1))
        builder.add_records([Record(content="test")])
        builder.reset()
        assert builder.record_count == 0
        assert builder.stats["total"] == 0

    def test_fluent_api(self):
        builder = TrainingDatasetBuilder(TrainingDataConfig(min_length=1))
        result = builder.add_records([Record(content="a")]).add_records([Record(content="b")])
        assert result is builder
        assert builder.record_count == 2


# ── CollectorTrainingBridge ───────────────────────────────────────────────


class TestCollectorTrainingBridge:

    def test_collect_and_prepare(self):
        src = NameSource("s", [Record(content="hello world test")])
        store = ListStore()
        collector = Collector(src, store)
        bridge = CollectorTrainingBridge(collector, TrainingDataConfig(min_length=1))
        data, vocab = bridge.collect_and_prepare()
        assert isinstance(data, np.ndarray)
        assert bridge.stats["collected"] > 0

    def test_get_text(self):
        src = NameSource("s", [])
        store = ListStore()
        collector = Collector(src, store)
        bridge = CollectorTrainingBridge(collector, TrainingDataConfig(min_length=1))
        bridge._records = [Record(content="hello"), Record(content="world")]
        text = bridge.get_text()
        assert text == "hello\nworld"

    def test_get_numpy(self):
        src = NameSource("s", [])
        store = ListStore()
        collector = Collector(src, store)
        bridge = CollectorTrainingBridge(collector, TrainingDataConfig(min_length=1))
        bridge._records = [Record(content="abc")]
        data, vocab = bridge.get_numpy()
        assert isinstance(data, np.ndarray)
        assert vocab == 3

    def test_adapter_property(self):
        src = NameSource("s", [])
        store = ListStore()
        collector = Collector(src, store)
        bridge = CollectorTrainingBridge(collector)
        assert bridge.adapter is bridge._adapter

    def test_reset(self):
        src = NameSource("s", [])
        store = ListStore()
        collector = Collector(src, store)
        bridge = CollectorTrainingBridge(collector, TrainingDataConfig(min_length=1))
        bridge._records = [Record(content="test")]
        bridge.reset()
        assert len(bridge._records) == 0

    def test_collect_and_save_text(self, tmp_path):
        src = NameSource("s", [Record(content="hello world test")])
        store = ListStore()
        collector = Collector(src, store)
        bridge = CollectorTrainingBridge(collector, TrainingDataConfig(min_length=1))
        out = tmp_path / "out.txt"
        chars = bridge.collect_and_save_text(str(out))
        assert chars > 0
        assert out.exists()
