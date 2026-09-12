from __future__ import annotations

import json
import hashlib
import numpy as np
from dataclasses import dataclass
from typing import Any, Iterator

from .sources import Record, Source
from .collector import Collector
from .filters import FilterChain


MATERIAL_SIGNAL = 4
MATERIAL_FOOD = 2
MATERIAL_TOXIC = 3


@dataclass
class WorldFeedConfig:
    grid_size: tuple[int, int, int] = (64, 32, 64)
    energy_scale: float = 1.0
    temperature_scale: float = 1.0
    signal_scale: float = 1.0
    feed_radius: int = 5
    max_records: int = 1000


class RecordToWorldMapper:
    def __init__(self, config: WorldFeedConfig | None = None):
        self.config = config or WorldFeedConfig()

    def record_to_cell_signal(self, record: Record) -> dict[str, float]:
        content_hash = hashlib.md5(record.content.encode("utf-8")).digest()
        hash_floats = np.frombuffer(content_hash, dtype=np.uint8).astype(np.float32) / 255.0

        energy = float(hash_floats[0]) * self.config.energy_scale
        temperature = 20.0 + float(hash_floats[1] - 0.5) * 20.0 * self.config.temperature_scale
        signal = float(hash_floats[2]) * self.config.signal_scale

        metadata_energy = record.metadata.get("energy")
        if metadata_energy is not None:
            energy = float(metadata_energy) * self.config.energy_scale

        metadata_temp = record.metadata.get("temperature")
        if metadata_temp is not None:
            temperature = float(metadata_temp)

        metadata_signal = record.metadata.get("signal")
        if metadata_signal is not None:
            signal = float(metadata_signal) * self.config.signal_scale

        return {"energy": energy, "temperature": temperature, "signal": signal}

    def record_to_position(self, record: Record, index: int) -> tuple[int, int, int]:
        pos = record.metadata.get("position")
        if pos and len(pos) == 3:
            return int(pos[0]), int(pos[1]), int(pos[2])

        content_hash = hashlib.md5(
            (record.content + str(index)).encode("utf-8")
        ).digest()
        h = np.frombuffer(content_hash[:12], dtype=np.uint32)

        nx, ny, nz = self.config.grid_size
        return int(h[0] % nx), int(h[1] % ny), int(h[2] % nz)

    def records_to_world_ops(self, records: list[Record]) -> list[dict[str, Any]]:
        ops = []
        for i, record in enumerate(records):
            pos = self.record_to_position(record, i)
            cell = self.record_to_cell_signal(record)
            ops.append({
                "type": "place_cell",
                "x": pos[0], "y": pos[1], "z": pos[2],
                "material": MATERIAL_SIGNAL,
                "energy": cell["energy"],
                "temperature": cell["temperature"],
                "signal_amplitude": cell["signal"],
                "record_content": record.content[:100],
                "record_metadata": record.metadata,
            })
        return ops


class WorldGridBridge:
    def __init__(self, world_grid=None, config: WorldFeedConfig | None = None):
        self._grid = world_grid
        self.config = config or WorldFeedConfig()
        self._mapper = RecordToWorldMapper(self.config)
        self._injected_records: list[Record] = []
        self.stats = {"injected": 0, "read": 0, "errors": 0}

    def set_grid(self, world_grid) -> None:
        self._grid = world_grid

    def inject_records(self, records: list[Record]) -> int:
        if self._grid is None:
            return 0
        count = 0
        for record in records:
            try:
                ops = self._mapper.records_to_world_ops([record])
                for op in ops:
                    self._grid.place_material(
                        op["x"], op["y"], op["z"],
                        op["material"],
                        energy=op["energy"],
                        temperature=op["temperature"],
                    )
                self._injected_records.append(record)
                self.stats["injected"] += 1
                count += 1
            except Exception:
                self.stats["errors"] += 1
        return count

    def inject_from_collector(self, collector: Collector) -> int:
        records = list(collector.source.read())
        return self.inject_records(records)

    def read_grid_as_records(self, center: tuple[int, int, int] | None = None,
                              radius: int | None = None) -> list[Record]:
        if self._grid is None:
            return []
        radius = radius or self.config.feed_radius
        cx, cy, cz = center or (self.config.grid_size[0] // 2,
                                 self.config.grid_size[1] // 2,
                                 self.config.grid_size[2] // 2)

        cells = self._grid.get_nearby_cells(cx, cy, cz, radius)
        records = []
        count = int(cells["count"]) if "count" in cells else 0

        for i in range(min(count, self.config.max_records)):
            content = json.dumps({
                "material": int(cells["material"][i]) if "material" in cells else 0,
                "energy": float(cells["energy"][i]) if "energy" in cells else 0.0,
                "temperature": float(cells["temperature"][i]) if "temperature" in cells else 0.0,
            })
            records.append(Record(
                content=content,
                metadata={
                    "source": "world_grid",
                    "position": [cx, cy, cz],
                    "radius": radius,
                },
            ))

        self.stats["read"] += len(records)
        return records

    def grid_to_source(self, center: tuple[int, int, int] | None = None,
                        radius: int | None = None) -> WorldGridSource:
        return WorldGridSource(self, center=center, radius=radius)


class WorldGridSource:
    def __init__(self, bridge: WorldGridBridge, center: tuple[int, int, int] | None = None,
                 radius: int | None = None):
        self._bridge = bridge
        self._center = center
        self._radius = radius
        self.name = "world_grid"

    def read(self) -> Iterator[Record]:
        return iter(self._bridge.read_grid_as_records(
            center=self._center, radius=self._radius
        ))


class WorldStoreAdapter:
    def __init__(self, bridge: WorldGridBridge):
        self._bridge = bridge
        self.name = "world_grid"
        self._count = 0

    def write(self, record: Record) -> None:
        self._bridge.inject_records([record])
        self._count += 1

    def read_all(self) -> Iterator[Record]:
        return iter(self._bridge.read_grid_as_records())

    def count(self) -> int:
        return self._count


class CollectionWorldPipeline:
    def __init__(self, source: Source, world_grid=None, config: WorldFeedConfig | None = None,
                 filters=None):
        self._source = source
        self._bridge = WorldGridBridge(world_grid, config)
        self._filters = FilterChain(filters or [])
        self.stats = {"total_collected": 0, "total_injected": 0, "filtered": 0}

    def set_grid(self, world_grid) -> None:
        self._bridge.set_grid(world_grid)

    def run(self) -> int:
        records = []
        for record in self._source.read():
            if self._filters.accept(record):
                records.append(record)
                self.stats["total_collected"] += 1
            else:
                self.stats["filtered"] += 1

        injected = self._bridge.inject_records(records)
        self.stats["total_injected"] += injected
        return injected

    def run_continuous(self, interval: float = 60.0, max_rounds: int | None = None) -> None:
        import time
        rounds = 0
        while max_rounds is None or rounds < max_rounds:
            self.run()
            rounds += 1
            time.sleep(interval)

    @property
    def bridge(self) -> WorldGridBridge:
        return self._bridge
