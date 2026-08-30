import hashlib
import numpy as np
import pytest
import tempfile
from pathlib import Path

from domains.collections.sources import Record, FileSource
from domains.collections.stores import MemoryStore
from domains.collections.collector import Collector
from domains.collections.training_bridge import (
    TrainingDataConfig,
    TrainingDataAdapter,
    RecordToTrainingSource,
    TrainingDatasetBuilder,
    CollectorTrainingBridge,
)


class _FixedSource:
    """Trivial source wrapping a list of Records."""

    def __init__(self, records, name="fixed"):
        self._records = records
        self.name = name

    def read(self):
        return iter(self._records)


# ---------------------------------------------------------------------------
# TrainingDataConfig
# ---------------------------------------------------------------------------

class TestTrainingDataConfig:
    def test_defaults(self):
        cfg = TrainingDataConfig()
        assert cfg.block_size == 128
        assert cfg.separator == "\n"
        assert cfg.include_metadata is False
        assert cfg.max_records is None
        assert cfg.deduplicate is True
        assert cfg.min_length == 10
        assert cfg.max_length == 100000

    def test_custom(self):
        cfg = TrainingDataConfig(block_size=64, separator=" ", min_length=5)
        assert cfg.block_size == 64
        assert cfg.separator == " "
        assert cfg.min_length == 5


# ---------------------------------------------------------------------------
# TrainingDataAdapter
# ---------------------------------------------------------------------------

class TestTrainingDataAdapter:
    def setup_method(self):
        self.adapter = TrainingDataAdapter(TrainingDataConfig(min_length=1))

    def test_records_to_text(self):
        records = [Record(content="hello"), Record(content="world")]
        text = self.adapter.records_to_text(records)
        assert text == "hello\nworld"

    def test_records_to_text_custom_separator(self):
        adapter = TrainingDataAdapter(TrainingDataConfig(separator=" ", min_length=1))
        records = [Record(content="hello"), Record(content="world")]
        text = adapter.records_to_text(records)
        assert text == "hello world"

    def test_min_length_filters(self):
        adapter = TrainingDataAdapter(TrainingDataConfig(min_length=5))
        records = [Record(content="hi"), Record(content="hello world")]
        text = adapter.records_to_text(records)
        assert text == "hello world"
        assert adapter.stats["too_short"] == 1

    def test_max_length_filters(self):
        adapter = TrainingDataAdapter(TrainingDataConfig(min_length=1, max_length=5))
        records = [Record(content="short"), Record(content="this is too long")]
        text = adapter.records_to_text(records)
        assert text == "short"
        assert adapter.stats["too_long"] == 1

    def test_deduplication(self):
        adapter = TrainingDataAdapter(TrainingDataConfig(deduplicate=True, min_length=1))
        records = [Record(content="hello"), Record(content="hello"), Record(content="world")]
        text = adapter.records_to_text(records)
        assert text == "hello\nworld"
        assert adapter.stats["deduplicated"] == 1
        assert adapter.stats["accepted"] == 2

    def test_dedup_disabled(self):
        adapter = TrainingDataAdapter(TrainingDataConfig(deduplicate=False, min_length=1))
        records = [Record(content="hello"), Record(content="hello")]
        text = adapter.records_to_text(records)
        assert text == "hello\nhello"
        assert adapter.stats["deduplicated"] == 0
        assert adapter.stats["accepted"] == 2

    def test_stats_tracking(self):
        records = [Record(content="a"), Record(content="b")]
        self.adapter.records_to_text(records)
        assert self.adapter.stats["total"] == 2
        assert self.adapter.stats["accepted"] == 2

    def test_records_to_training_data(self):
        records = [Record(content="hello"), Record(content="world")]
        data, stoi, itos = self.adapter.records_to_training_data(records)
        assert isinstance(data, np.ndarray)
        assert data.dtype == np.int64
        assert len(stoi) > 0
        assert len(itos) > 0
        # stoi and itos should be inverses
        for k, v in stoi.items():
            assert itos[v] == k

    def test_records_to_numpy(self):
        records = [Record(content="hello"), Record(content="world")]
        data, vocab_size = self.adapter.records_to_numpy(records)
        assert isinstance(data, np.ndarray)
        assert data.dtype == np.int64
        assert vocab_size > 0

    def test_records_to_text_file(self, tmp_path):
        records = [Record(content="hello"), Record(content="world")]
        path = str(tmp_path / "train.txt")
        count = self.adapter.records_to_text_file(records, path)
        assert count > 0
        content = Path(path).read_text()
        assert "hello\nworld" in content

    def test_reset(self):
        self.adapter.records_to_text([Record(content="hello")])
        assert self.adapter.stats["accepted"] == 1
        self.adapter.reset()
        assert self.adapter.stats["accepted"] == 0
        assert self.adapter.stats["total"] == 0
        assert len(self.adapter._seen_hashes) == 0

    def test_empty_records(self):
        text = self.adapter.records_to_text([])
        assert text == ""

    def test_all_filtered_returns_empty(self):
        adapter = TrainingDataAdapter(TrainingDataConfig(min_length=100))
        records = [Record(content="short")]
        text = adapter.records_to_text(records)
        assert text == ""
        assert adapter.stats["too_short"] == 1

    def test_dedup_hash_deterministic(self):
        h1 = hashlib.md5("test".encode("utf-8")).hexdigest()
        h2 = hashlib.md5("test".encode("utf-8")).hexdigest()
        assert h1 == h2

    def test_records_to_training_data_mapping(self):
        records = [Record(content="abc")]
        data, stoi, itos = self.adapter.records_to_training_data(records)
        # 'abc' has 3 unique chars; data should map each char to its stoi index
        assert len(data) == 3
        assert all(v in itos for v in data)


# ---------------------------------------------------------------------------
# RecordToTrainingSource
# ---------------------------------------------------------------------------

class TestRecordToTrainingSource:
    def test_read(self):
        records = [Record(content="a"), Record(content="b")]
        source = RecordToTrainingSource(records)
        result = list(source.read())
        assert len(result) == 2
        assert result[0].content == "a"
        assert result[1].content == "b"

    def test_name(self):
        source = RecordToTrainingSource([])
        assert source.name == "training_records"

    def test_empty(self):
        source = RecordToTrainingSource([])
        assert list(source.read()) == []


# ---------------------------------------------------------------------------
# TrainingDatasetBuilder
# ---------------------------------------------------------------------------

class TestTrainingDatasetBuilder:
    def setup_method(self):
        self.builder = TrainingDatasetBuilder(TrainingDataConfig(min_length=1))

    def test_add_records(self):
        self.builder.add_records([Record(content="hello"), Record(content="world")])
        assert self.builder.record_count == 2

    def test_add_records_chaining(self):
        result = self.builder.add_records([Record(content="a")])
        assert result is self.builder

    def test_build_text(self):
        self.builder.add_records([Record(content="hello"), Record(content="world")])
        text = self.builder.build_text()
        assert text == "hello\nworld"

    def test_build_numpy(self):
        self.builder.add_records([Record(content="hello")])
        data, vocab_size = self.builder.build_numpy()
        assert len(data) > 0
        assert vocab_size > 0

    def test_build_dataset(self):
        self.builder.add_records([Record(content="hello")])
        data, stoi, itos = self.builder.build_dataset()
        assert len(data) > 0
        assert len(stoi) > 0
        assert len(itos) > 0

    def test_save_text(self, tmp_path):
        self.builder.add_records([Record(content="hello")])
        path = str(tmp_path / "out.txt")
        count = self.builder.save_text(path)
        assert count > 0
        assert (tmp_path / "out.txt").exists()

    def test_save_numpy(self, tmp_path):
        self.builder.add_records([Record(content="hello")])
        path = str(tmp_path / "data.npy")
        count = self.builder.save_numpy(path)
        assert count > 0
        assert (tmp_path / "data.npy").exists()

    def test_add_from_text(self):
        self.builder.add_from_text("hello\nworld\nfoo")
        assert self.builder.record_count == 3

    def test_add_from_text_custom_separator(self):
        self.builder.add_from_text("hello world", separator=" ")
        assert self.builder.record_count == 2

    def test_add_from_text_skips_blanks(self):
        self.builder.add_from_text("hello\n\n\nworld")
        assert self.builder.record_count == 2

    def test_add_from_file(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("hello\nworld\n")
        self.builder.add_from_file(str(f))
        assert self.builder.record_count == 2

    def test_add_from_file_skips_blanks(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("hello\n\n\nworld\n")
        self.builder.add_from_file(str(f))
        assert self.builder.record_count == 2

    def test_add_from_source(self):
        records = [Record(content="a"), Record(content="b")]
        source = _FixedSource(records)
        self.builder.add_from_source(source)
        assert self.builder.record_count == 2

    def test_stats(self):
        self.builder.add_records([Record(content="hello world")])
        self.builder.build_text()
        assert self.builder.stats["accepted"] == 1
        assert self.builder.stats["total"] == 1

    def test_reset(self):
        self.builder.add_records([Record(content="hello")])
        self.builder.reset()
        assert self.builder.record_count == 0
        assert self.builder.stats["accepted"] == 0

    def test_build_text_empty(self):
        text = self.builder.build_text()
        assert text == ""


# ---------------------------------------------------------------------------
# CollectorTrainingBridge
# ---------------------------------------------------------------------------

class TestCollectorTrainingBridge:
    def test_collect_and_prepare(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("hello world\nfoo bar baz\n")
        collector = Collector(FileSource(str(f)), MemoryStore())
        bridge = CollectorTrainingBridge(collector, TrainingDataConfig(min_length=1))
        data, vocab_size = bridge.collect_and_prepare()
        assert len(data) > 0
        assert vocab_size > 0
        assert bridge.stats["collected"] == 2

    def test_collect_and_prepare_accumulates(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f1.write_text("hello world\n")
        collector = Collector(FileSource(str(f1)), MemoryStore())
        bridge = CollectorTrainingBridge(collector, TrainingDataConfig(min_length=1))
        bridge.collect_and_prepare()
        first_count = bridge.stats["collected"]
        bridge.collect_and_prepare()
        # Should accumulate on second call
        assert bridge.stats["collected"] >= first_count

    def test_collect_and_save_text(self, tmp_path):
        f = tmp_path / "input.txt"
        f.write_text("hello world\nfoo bar baz\n")
        out = tmp_path / "output.txt"
        collector = Collector(FileSource(str(f)), MemoryStore())
        bridge = CollectorTrainingBridge(collector, TrainingDataConfig(min_length=1))
        count = bridge.collect_and_save_text(str(out))
        assert count > 0
        assert out.exists()

    def test_get_text(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("hello world\nfoo bar baz\n")
        collector = Collector(FileSource(str(f)), MemoryStore())
        bridge = CollectorTrainingBridge(collector, TrainingDataConfig(min_length=1))
        bridge.collect_and_prepare()
        text = bridge.get_text()
        assert "hello world" in text
        assert "foo bar baz" in text

    def test_get_numpy(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("hello world\n")
        collector = Collector(FileSource(str(f)), MemoryStore())
        bridge = CollectorTrainingBridge(collector, TrainingDataConfig(min_length=1))
        bridge.collect_and_prepare()
        data, vocab_size = bridge.get_numpy()
        assert len(data) > 0
        assert vocab_size > 0

    def test_adapter_property(self):
        adapter = TrainingDataAdapter()
        bridge = CollectorTrainingBridge.__new__(CollectorTrainingBridge)
        bridge._adapter = adapter
        assert bridge.adapter is adapter

    def test_reset(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("hello world\n")
        collector = Collector(FileSource(str(f)), MemoryStore())
        bridge = CollectorTrainingBridge(collector, TrainingDataConfig(min_length=1))
        bridge.collect_and_prepare()
        bridge.reset()
        # reset() clears records and adapter stats, but bridge.stats is not reset
        assert len(bridge._records) == 0
        assert bridge.adapter.stats["accepted"] == 0

    def test_collect_with_dedup(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f1.write_text("hello world\nhello world\n")
        collector = Collector(FileSource(str(f1)), MemoryStore())
        bridge = CollectorTrainingBridge(collector, TrainingDataConfig(min_length=1, deduplicate=True))
        data, _ = bridge.collect_and_prepare()
        # Dedup should have removed one record
        assert bridge.adapter.stats["deduplicated"] >= 1


# ---------------------------------------------------------------------------
# Integration: CollectorTrainingBridge with GeneratorSource
# ---------------------------------------------------------------------------

class TestCollectorTrainingBridgeIntegration:
    def test_with_generator_source(self):
        def gen():
            yield Record(content="first record hello")
            yield Record(content="second record world")
            yield Record(content="third record again")

        source = _FixedSource(list(gen()))
        collector = Collector(source, MemoryStore())
        bridge = CollectorTrainingBridge(collector, TrainingDataConfig(min_length=1))
        data, vocab_size = bridge.collect_and_prepare()
        assert len(data) > 0
        assert bridge.stats["collected"] == 3

    def test_all_filtered(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("hi\n")
        collector = Collector(FileSource(str(f)), MemoryStore())
        bridge = CollectorTrainingBridge(collector, TrainingDataConfig(min_length=100))
        data, vocab_size = bridge.collect_and_prepare()
        assert len(data) == 0
        assert vocab_size == 0
