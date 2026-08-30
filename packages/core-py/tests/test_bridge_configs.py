"""Tests for domains.collections.training_bridge — TrainingDataConfig, TrainingDataAdapter; domains.collections.world_bridge — WorldFeedConfig, RecordToWorldMapper."""

import numpy as np
import pytest
from domains.collections.sources import Record
from domains.collections.training_bridge import TrainingDataConfig, TrainingDataAdapter
from domains.collections.world_bridge import (
    WorldFeedConfig,
    RecordToWorldMapper,
    MATERIAL_SIGNAL,
    MATERIAL_FOOD,
    MATERIAL_TOXIC,
)


class TestTrainingDataConfigDefaults:
    def test_block_size(self):
        tdc = TrainingDataConfig()
        assert tdc.block_size == 128

    def test_separator(self):
        tdc = TrainingDataConfig()
        assert tdc.separator == "\n"

    def test_include_metadata(self):
        tdc = TrainingDataConfig()
        assert tdc.include_metadata is False

    def test_max_records(self):
        tdc = TrainingDataConfig()
        assert tdc.max_records is None

    def test_deduplicate(self):
        tdc = TrainingDataConfig()
        assert tdc.deduplicate is True

    def test_min_length(self):
        tdc = TrainingDataConfig()
        assert tdc.min_length == 10

    def test_max_length(self):
        tdc = TrainingDataConfig()
        assert tdc.max_length == 100000


class TestTrainingDataConfigCustom:
    def test_block_size_custom(self):
        tdc = TrainingDataConfig(block_size=64)
        assert tdc.block_size == 64

    def test_separator_custom(self):
        tdc = TrainingDataConfig(separator="|")
        assert tdc.separator == "|"

    def test_include_metadata_true(self):
        tdc = TrainingDataConfig(include_metadata=True)
        assert tdc.include_metadata is True

    def test_max_records_custom(self):
        tdc = TrainingDataConfig(max_records=500)
        assert tdc.max_records == 500

    def test_deduplicate_false(self):
        tdc = TrainingDataConfig(deduplicate=False)
        assert tdc.deduplicate is False

    def test_min_length_custom(self):
        tdc = TrainingDataConfig(min_length=5)
        assert tdc.min_length == 5

    def test_max_length_custom(self):
        tdc = TrainingDataConfig(max_length=500)
        assert tdc.max_length == 500

    def test_all_custom(self):
        tdc = TrainingDataConfig(
            block_size=32, separator=";", include_metadata=True,
            max_records=10, deduplicate=False, min_length=1, max_length=100,
        )
        assert tdc.block_size == 32
        assert tdc.separator == ";"
        assert tdc.include_metadata is True
        assert tdc.max_records == 10
        assert tdc.deduplicate is False
        assert tdc.min_length == 1
        assert tdc.max_length == 100


class TestTrainingDataAdapter:
    def test_records_to_text_basic(self):
        adapter = TrainingDataAdapter(TrainingDataConfig(min_length=0))
        records = [Record(content="hello"), Record(content="world")]
        result = adapter.records_to_text(records)
        assert result == "hello\nworld"

    def test_records_to_text_custom_separator(self):
        adapter = TrainingDataAdapter(TrainingDataConfig(separator=" | ", min_length=0))
        records = [Record(content="a"), Record(content="b")]
        result = adapter.records_to_text(records)
        assert result == "a | b"

    def test_records_to_text_empty(self):
        adapter = TrainingDataAdapter()
        assert adapter.records_to_text([]) == ""

    def test_records_to_text_filters_short(self):
        adapter = TrainingDataAdapter(TrainingDataConfig(min_length=10))
        records = [Record(content="hi"), Record(content="hello world")]
        result = adapter.records_to_text(records)
        assert result == "hello world"

    def test_records_to_text_filters_long(self):
        adapter = TrainingDataAdapter(TrainingDataConfig(min_length=0, max_length=5))
        records = [Record(content="ok"), Record(content="too long")]
        result = adapter.records_to_text(records)
        assert result == "ok"

    def test_records_to_text_deduplication(self):
        adapter = TrainingDataAdapter(TrainingDataConfig(deduplicate=True, min_length=0))
        records = [Record(content="same"), Record(content="same")]
        result = adapter.records_to_text(records)
        assert result == "same"

    def test_records_to_text_no_dedup(self):
        adapter = TrainingDataAdapter(TrainingDataConfig(deduplicate=False, min_length=0))
        records = [Record(content="same"), Record(content="same")]
        result = adapter.records_to_text(records)
        assert result == "same\nsame"

    def test_stats_tracking(self):
        adapter = TrainingDataAdapter(TrainingDataConfig(min_length=6))
        records = [
            Record(content="short"),
            Record(content="hello world"),
            Record(content="hello world"),
        ]
        adapter.records_to_text(records)
        assert adapter.stats["total"] == 3
        assert adapter.stats["too_short"] == 1
        assert adapter.stats["deduplicated"] == 1
        assert adapter.stats["accepted"] == 1

    def test_records_to_training_data(self):
        adapter = TrainingDataAdapter(TrainingDataConfig(min_length=0))
        data, stoi, itos = adapter.records_to_training_data([Record(content="abc")])
        assert isinstance(stoi, dict)
        assert isinstance(itos, dict)
        assert len(stoi) == 3

    def test_records_to_numpy(self):
        adapter = TrainingDataAdapter(TrainingDataConfig(min_length=0))
        records = [Record(content="abc")]
        data, vocab_size = adapter.records_to_numpy(records)
        assert isinstance(data, np.ndarray)
        assert vocab_size == 3

    def test_records_to_numpy_empty(self):
        adapter = TrainingDataAdapter(TrainingDataConfig(min_length=0))
        data, vocab_size = adapter.records_to_numpy([])
        assert len(data) == 0
        assert vocab_size == 0

    def test_reset(self):
        adapter = TrainingDataAdapter(TrainingDataConfig(min_length=0))
        adapter.records_to_text([Record(content="test")])
        adapter.reset()
        assert adapter.stats["total"] == 0
        assert adapter.stats["accepted"] == 0
        assert len(adapter._seen_hashes) == 0

    def test_records_to_text_file(self, tmp_path):
        adapter = TrainingDataAdapter(TrainingDataConfig(min_length=0))
        records = [Record(content="line1"), Record(content="line2")]
        out = tmp_path / "out.txt"
        count = adapter.records_to_text_file(records, str(out))
        assert count > 0
        assert out.read_text() == "line1\nline2"


class TestWorldFeedConfigDefaults:
    def test_grid_size(self):
        wfc = WorldFeedConfig()
        assert wfc.grid_size == (64, 32, 64)

    def test_energy_scale(self):
        wfc = WorldFeedConfig()
        assert wfc.energy_scale == 1.0

    def test_temperature_scale(self):
        wfc = WorldFeedConfig()
        assert wfc.temperature_scale == 1.0

    def test_signal_scale(self):
        wfc = WorldFeedConfig()
        assert wfc.signal_scale == 1.0

    def test_feed_radius(self):
        wfc = WorldFeedConfig()
        assert wfc.feed_radius == 5

    def test_max_records(self):
        wfc = WorldFeedConfig()
        assert wfc.max_records == 1000


class TestWorldFeedConfigCustom:
    def test_grid_size_custom(self):
        wfc = WorldFeedConfig(grid_size=(32, 16, 32))
        assert wfc.grid_size == (32, 16, 32)

    def test_energy_scale_custom(self):
        wfc = WorldFeedConfig(energy_scale=2.0)
        assert wfc.energy_scale == 2.0

    def test_temperature_scale_custom(self):
        wfc = WorldFeedConfig(temperature_scale=0.5)
        assert wfc.temperature_scale == 0.5

    def test_signal_scale_custom(self):
        wfc = WorldFeedConfig(signal_scale=3.0)
        assert wfc.signal_scale == 3.0

    def test_feed_radius_custom(self):
        wfc = WorldFeedConfig(feed_radius=10)
        assert wfc.feed_radius == 10

    def test_max_records_custom(self):
        wfc = WorldFeedConfig(max_records=5000)
        assert wfc.max_records == 5000

    def test_all_custom(self):
        wfc = WorldFeedConfig(
            grid_size=(10, 10, 10), energy_scale=2.5, temperature_scale=0.5,
            signal_scale=1.5, feed_radius=20, max_records=200,
        )
        assert wfc.grid_size == (10, 10, 10)
        assert wfc.energy_scale == 2.5
        assert wfc.temperature_scale == 0.5
        assert wfc.signal_scale == 1.5
        assert wfc.feed_radius == 20
        assert wfc.max_records == 200


class TestMaterialConstants:
    def test_material_signal(self):
        assert MATERIAL_SIGNAL == 4

    def test_material_food(self):
        assert MATERIAL_FOOD == 2

    def test_material_toxic(self):
        assert MATERIAL_TOXIC == 3

    def test_all_different(self):
        assert MATERIAL_SIGNAL != MATERIAL_FOOD
        assert MATERIAL_SIGNAL != MATERIAL_TOXIC
        assert MATERIAL_FOOD != MATERIAL_TOXIC


class TestRecordToWorldMapper:
    def test_record_to_cell_signal(self):
        mapper = RecordToWorldMapper()
        record = Record(content="test content")
        signal = mapper.record_to_cell_signal(record)
        assert "energy" in signal
        assert "temperature" in signal
        assert "signal" in signal
        assert isinstance(signal["energy"], float)
        assert isinstance(signal["temperature"], float)
        assert isinstance(signal["signal"], float)

    def test_record_to_cell_signal_energy_range(self):
        mapper = RecordToWorldMapper()
        record = Record(content="test")
        signal = mapper.record_to_cell_signal(record)
        assert 0.0 <= signal["energy"] <= 1.0

    def test_record_to_cell_signal_with_metadata_energy(self):
        mapper = RecordToWorldMapper()
        record = Record(content="test", metadata={"energy": 0.5})
        signal = mapper.record_to_cell_signal(record)
        assert signal["energy"] == 0.5

    def test_record_to_cell_signal_with_metadata_temp(self):
        mapper = RecordToWorldMapper()
        record = Record(content="test", metadata={"temperature": 37.0})
        signal = mapper.record_to_cell_signal(record)
        assert signal["temperature"] == 37.0

    def test_record_to_cell_signal_with_metadata_signal(self):
        mapper = RecordToWorldMapper()
        record = Record(content="test", metadata={"signal": 0.9})
        signal = mapper.record_to_cell_signal(record)
        assert signal["signal"] == 0.9

    def test_record_to_cell_signal_energy_scale(self):
        mapper = RecordToWorldMapper(WorldFeedConfig(energy_scale=2.0))
        record = Record(content="test", metadata={"energy": 0.5})
        signal = mapper.record_to_cell_signal(record)
        assert signal["energy"] == 1.0

    def test_record_to_cell_signal_signal_scale(self):
        mapper = RecordToWorldMapper(WorldFeedConfig(signal_scale=3.0))
        record = Record(content="test", metadata={"signal": 0.5})
        signal = mapper.record_to_cell_signal(record)
        assert signal["signal"] == 1.5

    def test_record_to_cell_signal_deterministic(self):
        mapper = RecordToWorldMapper()
        record = Record(content="same content")
        s1 = mapper.record_to_cell_signal(record)
        s2 = mapper.record_to_cell_signal(record)
        assert s1 == s2

    def test_record_to_cell_signal_different_content(self):
        mapper = RecordToWorldMapper()
        s1 = mapper.record_to_cell_signal(Record(content="alpha"))
        s2 = mapper.record_to_cell_signal(Record(content="beta"))
        assert s1 != s2

    def test_record_to_position_deterministic(self):
        mapper = RecordToWorldMapper()
        record = Record(content="test")
        p1 = mapper.record_to_position(record, 0)
        p2 = mapper.record_to_position(record, 0)
        assert p1 == p2

    def test_record_to_position_different_index(self):
        mapper = RecordToWorldMapper()
        record = Record(content="test")
        p1 = mapper.record_to_position(record, 0)
        p2 = mapper.record_to_position(record, 1)
        assert p1 != p2

    def test_record_to_position_within_grid(self):
        config = WorldFeedConfig(grid_size=(10, 10, 10))
        mapper = RecordToWorldMapper(config)
        for i in range(20):
            pos = mapper.record_to_position(Record(content=f"item_{i}"), i)
            assert 0 <= pos[0] < 10
            assert 0 <= pos[1] < 10
            assert 0 <= pos[2] < 10

    def test_record_to_position_metadata_override(self):
        mapper = RecordToWorldMapper()
        record = Record(content="test", metadata={"position": [5, 10, 15]})
        pos = mapper.record_to_position(record, 0)
        assert pos == (5, 10, 15)

    def test_records_to_world_ops(self):
        mapper = RecordToWorldMapper()
        records = [Record(content="a"), Record(content="b")]
        ops = mapper.records_to_world_ops(records)
        assert len(ops) == 2
        for op in ops:
            assert op["type"] == "place_cell"
            assert op["material"] == MATERIAL_SIGNAL
            assert "energy" in op
            assert "temperature" in op
            assert "signal_amplitude" in op

    def test_records_to_world_ops_empty(self):
        mapper = RecordToWorldMapper()
        assert mapper.records_to_world_ops([]) == []

    def test_records_to_world_ops_content_truncated(self):
        mapper = RecordToWorldMapper()
        long_content = "x" * 200
        ops = mapper.records_to_world_ops([Record(content=long_content)])
        assert len(ops[0]["record_content"]) <= 100

    def test_records_to_world_ops_metadata_preserved(self):
        mapper = RecordToWorldMapper()
        record = Record(content="test", metadata={"custom": "value"})
        ops = mapper.records_to_world_ops([record])
        assert ops[0]["record_metadata"]["custom"] == "value"

    def test_mapper_custom_config(self):
        config = WorldFeedConfig(energy_scale=5.0, temperature_scale=2.0)
        mapper = RecordToWorldMapper(config)
        record = Record(content="test", metadata={"energy": 0.2})
        signal = mapper.record_to_cell_signal(record)
        assert signal["energy"] == 1.0
