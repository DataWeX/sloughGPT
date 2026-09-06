"""Tests for collections.world_bridge — WorldFeedConfig, RecordToWorldMapper, WorldGridBridge."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import numpy as np
import pytest

from domains.collections.world_bridge import (
    WorldFeedConfig,
    RecordToWorldMapper,
    WorldGridBridge,
    WorldGridSource,
    WorldStoreAdapter,
    CollectionWorldPipeline,
    MATERIAL_SIGNAL,
)
from domains.collections.sources import Record


# ── WorldFeedConfig ───────────────────────────────────────────────────────


class TestWorldFeedConfig:

    def test_defaults(self):
        cfg = WorldFeedConfig()
        assert cfg.grid_size == (64, 32, 64)
        assert cfg.energy_scale == 1.0
        assert cfg.temperature_scale == 1.0
        assert cfg.signal_scale == 1.0
        assert cfg.feed_radius == 5
        assert cfg.max_records == 1000


# ── RecordToWorldMapper ──────────────────────────────────────────────────


class TestRecordToWorldMapper:

    def test_record_to_cell_signal(self):
        mapper = RecordToWorldMapper()
        record = Record(content="hello world")
        signal = mapper.record_to_cell_signal(record)
        assert "energy" in signal
        assert "temperature" in signal
        assert "signal" in signal
        assert isinstance(signal["energy"], float)

    def test_record_to_cell_signal_metadata_override(self):
        mapper = RecordToWorldMapper()
        record = Record(content="test", metadata={"energy": 0.5, "temperature": 25.0, "signal": 0.8})
        signal = mapper.record_to_cell_signal(record)
        assert signal["energy"] == 0.5
        assert signal["temperature"] == 25.0
        assert signal["signal"] == 0.8

    def test_record_to_cell_signal_scales(self):
        cfg = WorldFeedConfig(energy_scale=2.0, signal_scale=3.0)
        mapper = RecordToWorldMapper(cfg)
        record = Record(content="test", metadata={"energy": 0.5, "signal": 0.5})
        signal = mapper.record_to_cell_signal(record)
        assert signal["energy"] == 1.0
        assert signal["signal"] == 1.5

    def test_record_to_position_from_metadata(self):
        mapper = RecordToWorldMapper()
        record = Record(content="test", metadata={"position": [10, 20, 30]})
        pos = mapper.record_to_position(record, 0)
        assert pos == (10, 20, 30)

    def test_record_to_position_hash(self):
        mapper = RecordToWorldMapper()
        record = Record(content="test")
        pos = mapper.record_to_position(record, 0)
        assert len(pos) == 3
        assert all(isinstance(v, int) for v in pos)

    def test_records_to_world_ops(self):
        mapper = RecordToWorldMapper()
        records = [Record(content="hello"), Record(content="world")]
        ops = mapper.records_to_world_ops(records)
        assert len(ops) == 2
        assert ops[0]["type"] == "place_cell"
        assert ops[0]["material"] == MATERIAL_SIGNAL
        assert "energy" in ops[0]
        assert "temperature" in ops[0]
        assert "signal_amplitude" in ops[0]

    def test_record_content_truncated(self):
        mapper = RecordToWorldMapper()
        record = Record(content="x" * 200)
        ops = mapper.records_to_world_ops([record])
        assert len(ops[0]["record_content"]) <= 100


# ── WorldGridBridge ───────────────────────────────────────────────────────


class TestWorldGridBridge:

    def test_no_grid_returns_empty(self):
        bridge = WorldGridBridge()
        assert bridge.inject_records([Record(content="test")]) == 0
        assert bridge.read_grid_as_records() == []

    def test_inject_records(self):
        grid = MagicMock()
        bridge = WorldGridBridge(grid)
        records = [Record(content="hello"), Record(content="world")]
        count = bridge.inject_records(records)
        assert count == 2
        assert bridge.stats["injected"] == 2
        assert grid.place_material.call_count == 2

    def test_inject_error_counted(self):
        grid = MagicMock()
        grid.place_material.side_effect = RuntimeError("boom")
        bridge = WorldGridBridge(grid)
        count = bridge.inject_records([Record(content="test")])
        assert count == 0
        assert bridge.stats["errors"] == 1

    def test_read_grid_as_records(self):
        grid = MagicMock()
        grid.get_nearby_cells.return_value = {
            "count": 1,
            "material": [1],
            "energy": [0.5],
            "temperature": [22.0],
        }
        bridge = WorldGridBridge(grid)
        records = bridge.read_grid_as_records()
        assert len(records) == 1
        data = json.loads(records[0].content)
        assert data["material"] == 1
        assert data["energy"] == 0.5
        assert bridge.stats["read"] == 1

    def test_read_grid_custom_center(self):
        grid = MagicMock()
        grid.get_nearby_cells.return_value = {"count": 0}
        bridge = WorldGridBridge(grid)
        records = bridge.read_grid_as_records(center=(10, 20, 30), radius=10)
        grid.get_nearby_cells.assert_called_with(10, 20, 30, 10)

    def test_set_grid(self):
        bridge = WorldGridBridge()
        new_grid = MagicMock()
        bridge.set_grid(new_grid)
        bridge.inject_records([Record(content="test")])
        new_grid.place_material.assert_called_once()

    def test_grid_to_source(self):
        bridge = WorldGridBridge()
        source = bridge.grid_to_source()
        assert isinstance(source, WorldGridSource)
        assert source.name == "world_grid"


# ── WorldGridSource ───────────────────────────────────────────────────────


class TestWorldGridSource:

    def test_read(self):
        grid = MagicMock()
        grid.get_nearby_cells.return_value = {"count": 0}
        bridge = WorldGridBridge(grid)
        source = WorldGridSource(bridge)
        records = list(source.read())
        assert len(records) == 0


# ── WorldStoreAdapter ─────────────────────────────────────────────────────


class TestWorldStoreAdapter:

    def test_write(self):
        bridge = MagicMock()
        bridge.inject_records.return_value = 1
        adapter = WorldStoreAdapter(bridge)
        adapter.write(Record(content="test"))
        assert adapter.count() == 1
        bridge.inject_records.assert_called_once()

    def test_read_all(self):
        bridge = MagicMock()
        bridge.read_grid_as_records.return_value = [Record(content="data")]
        adapter = WorldStoreAdapter(bridge)
        records = list(adapter.read_all())
        assert len(records) == 1

    def test_count(self):
        bridge = MagicMock()
        adapter = WorldStoreAdapter(bridge)
        assert adapter.count() == 0


# ── CollectionWorldPipeline ───────────────────────────────────────────────


class TestCollectionWorldPipeline:

    def _make_source(self, records):
        src = MagicMock()
        src.read.return_value = iter(records)
        return src

    def test_run_injects(self):
        grid = MagicMock()
        src = self._make_source([Record(content="a"), Record(content="b")])
        pipeline = CollectionWorldPipeline(src, grid)
        count = pipeline.run()
        assert count == 2
        assert pipeline.stats["total_collected"] == 2
        assert pipeline.stats["total_injected"] == 2

    def test_run_filters(self):
        from domains.collections.filters import LengthFilter
        grid = MagicMock()
        src = self._make_source([Record(content="hi"), Record(content="long enough text")])
        pipeline = CollectionWorldPipeline(src, grid, filters=[LengthFilter(min_length=10)])
        count = pipeline.run()
        assert pipeline.stats["filtered"] == 1
        assert pipeline.stats["total_collected"] == 1

    def test_set_grid(self):
        pipeline = CollectionWorldPipeline(self._make_source([]))
        new_grid = MagicMock()
        pipeline.set_grid(new_grid)
        assert pipeline.bridge._grid is new_grid

    def test_bridge_property(self):
        pipeline = CollectionWorldPipeline(self._make_source([]))
        assert isinstance(pipeline.bridge, WorldGridBridge)
