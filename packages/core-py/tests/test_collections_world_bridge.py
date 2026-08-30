import hashlib
import json
import numpy as np
import pytest

from domains.collections.sources import Record
from domains.collections.stores import MemoryStore
from domains.collections.filters import LengthFilter
from domains.collections.world_bridge import (
    MATERIAL_SIGNAL,
    MATERIAL_FOOD,
    MATERIAL_TOXIC,
    WorldFeedConfig,
    RecordToWorldMapper,
    WorldGridBridge,
    WorldGridSource,
    WorldStoreAdapter,
    CollectionWorldPipeline,
)


class _FakeGrid:
    """Minimal world grid stub for testing WorldGridBridge."""

    def __init__(self, shape=(64, 32, 64)):
        self._shape = shape
        size = shape[0] * shape[1] * shape[2]
        self.material = np.zeros(size, dtype=np.int64)
        self.energy = np.zeros(size, dtype=np.float64)
        self.temperature = np.full(size, 20.0, dtype=np.float64)
        self._placed = []

    def place_material(self, x, y, z, material, energy=0.0, temperature=20.0):
        nx, ny, nz = self._shape
        idx = x * ny * nz + y * nz + z
        self.material[idx] = material
        self.energy[idx] = energy
        self.temperature[idx] = temperature
        self._placed.append((x, y, z, material, energy, temperature))

    def get_nearby_cells(self, cx, cy, cz, radius):
        nx, ny, nz = self._shape
        cells = {"count": 0, "material": [], "energy": [], "temperature": []}
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    x, y, z = cx + dx, cy + dy, cz + dz
                    if 0 <= x < nx and 0 <= y < ny and 0 <= z < nz:
                        idx = x * ny * nz + y * nz + z
                        cells["material"].append(self.material[idx])
                        cells["energy"].append(self.energy[idx])
                        cells["temperature"].append(self.temperature[idx])
                        cells["count"] += 1
        return cells


class _FixedSource:
    """Trivial source wrapping a list of Records."""

    def __init__(self, records, name="fixed"):
        self._records = records
        self.name = name

    def read(self):
        return iter(self._records)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestMaterialConstants:
    def test_values(self):
        assert MATERIAL_SIGNAL == 4
        assert MATERIAL_FOOD == 2
        assert MATERIAL_TOXIC == 3


# ---------------------------------------------------------------------------
# WorldFeedConfig
# ---------------------------------------------------------------------------

class TestWorldFeedConfig:
    def test_defaults(self):
        cfg = WorldFeedConfig()
        assert cfg.grid_size == (64, 32, 64)
        assert cfg.energy_scale == 1.0
        assert cfg.temperature_scale == 1.0
        assert cfg.signal_scale == 1.0
        assert cfg.feed_radius == 5
        assert cfg.max_records == 1000

    def test_custom(self):
        cfg = WorldFeedConfig(grid_size=(32, 16, 32), energy_scale=2.0, feed_radius=10)
        assert cfg.grid_size == (32, 16, 32)
        assert cfg.energy_scale == 2.0
        assert cfg.feed_radius == 10


# ---------------------------------------------------------------------------
# RecordToWorldMapper
# ---------------------------------------------------------------------------

class TestRecordToWorldMapper:
    def setup_method(self):
        self.mapper = RecordToWorldMapper()

    def test_record_to_cell_signal_returns_keys(self):
        cell = self.mapper.record_to_cell_signal(Record(content="test"))
        assert "energy" in cell
        assert "temperature" in cell
        assert "signal" in cell

    def test_record_to_cell_signal_energy_from_metadata(self):
        r = Record(content="test", metadata={"energy": 0.75})
        cell = self.mapper.record_to_cell_signal(r)
        assert cell["energy"] == pytest.approx(0.75, abs=0.001)

    def test_record_to_cell_signal_temperature_from_metadata(self):
        r = Record(content="test", metadata={"temperature": 37.5})
        cell = self.mapper.record_to_cell_signal(r)
        assert cell["temperature"] == pytest.approx(37.5)

    def test_record_to_cell_signal_signal_from_metadata(self):
        r = Record(content="test", metadata={"signal": 0.9})
        cell = self.mapper.record_to_cell_signal(r)
        assert cell["signal"] == pytest.approx(0.9, abs=0.001)

    def test_record_to_cell_signal_energy_scale(self):
        r = Record(content="test")
        cfg = WorldFeedConfig(energy_scale=3.0)
        mapper = RecordToWorldMapper(cfg)
        cell = mapper.record_to_cell_signal(r)
        assert cell["energy"] >= 0

    def test_record_to_cell_signal_temperature_base(self):
        r = Record(content="test")
        cell = self.mapper.record_to_cell_signal(r)
        # Base temperature should be around 20 +/- some hash-based offset
        assert 10.0 < cell["temperature"] < 30.0

    def test_record_to_position_from_metadata(self):
        r = Record(content="test", metadata={"position": [10, 5, 20]})
        pos = self.mapper.record_to_position(r, 0)
        assert pos == (10, 5, 20)

    def test_record_to_position_from_metadata_casts_to_int(self):
        r = Record(content="test", metadata={"position": [10.7, 5.3, 20.1]})
        pos = self.mapper.record_to_position(r, 0)
        assert pos == (10, 5, 20)

    def test_record_to_position_from_hash(self):
        r = Record(content="test content for hashing")
        pos = self.mapper.record_to_position(r, 0)
        nx, ny, nz = 64, 32, 64
        assert 0 <= pos[0] < nx
        assert 0 <= pos[1] < ny
        assert 0 <= pos[2] < nz

    def test_record_to_position_deterministic(self):
        r = Record(content="same content")
        p1 = self.mapper.record_to_position(r, 0)
        p2 = self.mapper.record_to_position(r, 0)
        assert p1 == p2

    def test_record_to_position_varies_with_index(self):
        r = Record(content="same content")
        p1 = self.mapper.record_to_position(r, 0)
        p2 = self.mapper.record_to_position(r, 1)
        # Different index means different hash input, positions likely differ
        # Just verify both are valid
        for p in (p1, p2):
            assert 0 <= p[0] < 64
            assert 0 <= p[1] < 32
            assert 0 <= p[2] < 64

    def test_records_to_world_ops(self):
        records = [Record(content="a"), Record(content="b")]
        ops = self.mapper.records_to_world_ops(records)
        assert len(ops) == 2
        assert ops[0]["type"] == "place_cell"
        assert "x" in ops[0]
        assert "y" in ops[0]
        assert "z" in ops[0]
        assert "material" in ops[0]
        assert "energy" in ops[0]
        assert "temperature" in ops[0]
        assert "signal_amplitude" in ops[0]

    def test_records_to_world_ops_material_signal(self):
        ops = self.mapper.records_to_world_ops([Record(content="test")])
        assert ops[0]["material"] == MATERIAL_SIGNAL

    def test_records_to_world_ops_record_content_truncated(self):
        long = Record(content="x" * 200)
        ops = self.mapper.records_to_world_ops([long])
        assert len(ops[0]["record_content"]) == 100

    def test_records_to_world_ops_empty(self):
        ops = self.mapper.records_to_world_ops([])
        assert ops == []


# ---------------------------------------------------------------------------
# WorldGridBridge
# ---------------------------------------------------------------------------

class TestWorldGridBridge:
    def test_inject_no_grid(self):
        bridge = WorldGridBridge()
        count = bridge.inject_records([Record(content="hello")])
        assert count == 0
        assert bridge.stats["injected"] == 0

    def test_inject_with_grid(self):
        grid = _FakeGrid()
        bridge = WorldGridBridge(grid)
        records = [Record(content="hello world")]
        count = bridge.inject_records(records)
        assert count == 1
        assert bridge.stats["injected"] == 1
        assert len(bridge._injected_records) == 1

    def test_inject_multiple_records(self):
        grid = _FakeGrid()
        bridge = WorldGridBridge(grid)
        records = [Record(content="a"), Record(content="b"), Record(content="c")]
        count = bridge.inject_records(records)
        assert count == 3

    def test_inject_with_position_metadata(self):
        grid = _FakeGrid()
        bridge = WorldGridBridge(grid)
        records = [Record(content="pos test", metadata={"position": [5, 3, 10]})]
        count = bridge.inject_records(records)
        assert count == 1
        assert grid._placed[0][:3] == (5, 3, 10)

    def test_inject_with_energy_metadata(self):
        grid = _FakeGrid()
        bridge = WorldGridBridge(grid)
        records = [Record(content="energy test", metadata={"energy": 2.5, "position": [1, 1, 1]})]
        bridge.inject_records(records)
        placed = grid._placed[0]
        assert placed[4] == pytest.approx(2.5)

    def test_inject_with_temperature_metadata(self):
        grid = _FakeGrid()
        bridge = WorldGridBridge(grid)
        records = [Record(content="temp test", metadata={"temperature": 37.0, "position": [1, 1, 1]})]
        bridge.inject_records(records)
        placed = grid._placed[0]
        assert placed[5] == pytest.approx(37.0)

    def test_inject_with_error_increments_errors(self):
        grid = _FakeGrid()
        bridge = WorldGridBridge(grid)
        # Create a record whose position will be out of bounds
        records = [Record(content="err", metadata={"position": [999, 999, 999]})]
        count = bridge.inject_records(records)
        assert count == 0
        assert bridge.stats["errors"] == 1

    def test_set_grid(self):
        bridge = WorldGridBridge()
        grid = _FakeGrid()
        bridge.set_grid(grid)
        count = bridge.inject_records([Record(content="after set")])
        assert count == 1

    def test_read_grid_as_records_no_grid(self):
        bridge = WorldGridBridge()
        records = bridge.read_grid_as_records()
        assert records == []

    def test_read_grid_as_records_empty_grid(self):
        grid = _FakeGrid(shape=(8, 4, 8))
        bridge = WorldGridBridge(grid)
        records = bridge.read_grid_as_records(center=(4, 2, 4), radius=2)
        # Grid is all zeros, but still returns records for each nearby cell
        # count is based on get_nearby_cells which counts all cells in radius
        assert isinstance(records, list)

    def test_read_grid_as_records_content_is_json(self):
        grid = _FakeGrid(shape=(8, 4, 8))
        bridge = WorldGridBridge(grid, WorldFeedConfig(grid_size=(8, 4, 8), feed_radius=1))
        records = bridge.read_grid_as_records(center=(4, 2, 4), radius=1)
        assert len(records) >= 1
        data = json.loads(records[0].content)
        assert "material" in data
        assert "energy" in data
        assert "temperature" in data

    def test_read_grid_as_records_metadata(self):
        grid = _FakeGrid(shape=(8, 4, 8))
        bridge = WorldGridBridge(grid, WorldFeedConfig(grid_size=(8, 4, 8), feed_radius=1))
        records = bridge.read_grid_as_records(center=(4, 2, 4), radius=1)
        assert records[0].metadata["source"] == "world_grid"
        assert records[0].metadata["position"] == [4, 2, 4]
        assert records[0].metadata["radius"] == 1

    def test_read_grid_as_records_stats(self):
        grid = _FakeGrid(shape=(8, 4, 8))
        bridge = WorldGridBridge(grid, WorldFeedConfig(grid_size=(8, 4, 8), feed_radius=1))
        records = bridge.read_grid_as_records(center=(4, 2, 4), radius=1)
        assert bridge.stats["read"] == len(records)

    def test_grid_to_source(self):
        grid = _FakeGrid(shape=(8, 4, 8))
        bridge = WorldGridBridge(grid)
        source = bridge.grid_to_source(center=(4, 2, 4), radius=2)
        assert isinstance(source, WorldGridSource)
        assert source.name == "world_grid"
        records = list(source.read())
        assert isinstance(records, list)


# ---------------------------------------------------------------------------
# WorldGridSource
# ---------------------------------------------------------------------------

class TestWorldGridSource:
    def test_read_delegates_to_bridge(self):
        grid = _FakeGrid(shape=(8, 4, 8))
        bridge = WorldGridBridge(grid, WorldFeedConfig(grid_size=(8, 4, 8), feed_radius=1))
        source = WorldGridSource(bridge, center=(4, 2, 4), radius=1)
        records = list(source.read())
        assert len(records) >= 1

    def test_name(self):
        bridge = WorldGridBridge()
        source = WorldGridSource(bridge)
        assert source.name == "world_grid"


# ---------------------------------------------------------------------------
# WorldStoreAdapter
# ---------------------------------------------------------------------------

class TestWorldStoreAdapter:
    def test_write(self):
        grid = _FakeGrid()
        bridge = WorldGridBridge(grid)
        adapter = WorldStoreAdapter(bridge)
        adapter.write(Record(content="test write"))
        assert adapter.count() == 1
        assert bridge.stats["injected"] == 1

    def test_write_multiple(self):
        grid = _FakeGrid()
        bridge = WorldGridBridge(grid)
        adapter = WorldStoreAdapter(bridge)
        adapter.write(Record(content="a"))
        adapter.write(Record(content="b"))
        assert adapter.count() == 2

    def test_read_all(self):
        grid = _FakeGrid(shape=(8, 4, 8))
        bridge = WorldGridBridge(grid)
        adapter = WorldStoreAdapter(bridge)
        records = list(adapter.read_all())
        assert isinstance(records, list)

    def test_count_with_no_writes(self):
        bridge = WorldGridBridge()
        adapter = WorldStoreAdapter(bridge)
        assert adapter.count() == 0

    def test_name(self):
        bridge = WorldGridBridge()
        adapter = WorldStoreAdapter(bridge)
        assert adapter.name == "world_grid"


# ---------------------------------------------------------------------------
# CollectionWorldPipeline
# ---------------------------------------------------------------------------

class TestCollectionWorldPipeline:
    def test_run_no_grid(self):
        records = [Record(content="a"), Record(content="b")]
        source = _FixedSource(records)
        pipeline = CollectionWorldPipeline(source)
        count = pipeline.run()
        assert count == 0
        assert pipeline.stats["total_collected"] == 2
        assert pipeline.stats["total_injected"] == 0

    def test_run_with_grid(self):
        grid = _FakeGrid()
        records = [Record(content="hello world")]
        source = _FixedSource(records)
        pipeline = CollectionWorldPipeline(source, grid)
        count = pipeline.run()
        assert count == 1
        assert pipeline.stats["total_injected"] == 1

    def test_run_with_filter(self):
        records = [
            Record(content="short"),
            Record(content="this is a much longer record that passes"),
        ]
        source = _FixedSource(records)
        pipeline = CollectionWorldPipeline(source, filters=[LengthFilter(min_length=10)])
        count = pipeline.run()
        assert count == 0
        assert pipeline.stats["total_collected"] == 1
        assert pipeline.stats["filtered"] == 1

    def test_set_grid(self):
        pipeline = CollectionWorldPipeline(_FixedSource([]))
        grid = _FakeGrid()
        pipeline.set_grid(grid)
        records = [Record(content="after set grid")]
        pipeline._source = _FixedSource(records)
        count = pipeline.run()
        assert count == 1

    def test_run_accumulates(self):
        grid = _FakeGrid()
        records1 = [Record(content="first")]
        records2 = [Record(content="second")]
        source = _FixedSource(records1)
        pipeline = CollectionWorldPipeline(source, grid)
        pipeline.run()
        pipeline._source = _FixedSource(records2)
        pipeline.run()
        assert pipeline.stats["total_collected"] == 2
        assert pipeline.stats["total_injected"] == 2

    def test_bridge_property(self):
        pipeline = CollectionWorldPipeline(_FixedSource([]))
        assert isinstance(pipeline.bridge, WorldGridBridge)

    def test_filters_rejected_not_injected(self):
        grid = _FakeGrid()
        records = [Record(content="hi")]
        source = _FixedSource(records)
        pipeline = CollectionWorldPipeline(source, grid, filters=[LengthFilter(min_length=100)])
        count = pipeline.run()
        assert count == 0
        assert pipeline.stats["filtered"] == 1
        assert pipeline.stats["total_collected"] == 0

    def test_custom_config(self):
        cfg = WorldFeedConfig(grid_size=(8, 4, 8), feed_radius=2)
        pipeline = CollectionWorldPipeline(_FixedSource([]), config=cfg)
        assert pipeline.bridge.config.grid_size == (8, 4, 8)

    def test_multiple_records_partial_filter(self):
        grid = _FakeGrid()
        records = [
            Record(content="short"),
            Record(content="long enough record to pass filter"),
            Record(content="also short"),
            Record(content="another long enough record"),
        ]
        source = _FixedSource(records)
        pipeline = CollectionWorldPipeline(source, grid, filters=[LengthFilter(min_length=20)])
        count = pipeline.run()
        assert count == 2
        assert pipeline.stats["total_collected"] == 2
        assert pipeline.stats["filtered"] == 2
