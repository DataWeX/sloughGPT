from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


from .sources import Record, Source
from .stores import Store, MemoryStore

from .collector import Collector



@dataclass
class PerceptionConfig:
    grid_size: tuple[int, int, int] = (64, 4, 64)
    center: tuple[int, int, int] = (32, 0, 32)
    radius: int = 15
    material_map: dict[str, int] = field(default_factory=lambda: {
        "text": 1, "code": 2, "image": 3, "audio": 4,
        "news": 5, "question": 6, "answer": 7, "event": 8,
    })
    energy_scale: float = 1.0
    decay_rate: float = 0.95
    max_records: int = 1000
    dedup_window: int = 100


@dataclass
class PerceptionEvent:
    record: Record
    grid_pos: tuple[int, int, int]
    material_type: int
    energy: float
    timestamp: float
    metadata: dict = field(default_factory=dict)


class RecordToMaterial:
    def __init__(self, config: PerceptionConfig | None = None):
        self.config = config or PerceptionConfig()

    def classify(self, record: Record) -> str:
        text = record.content.lower()
        if any(kw in text for kw in ["<code", "def ", "class ", "import ", "function"]):
            return "code"
        if any(kw in text for kw in ["<img", "photo", "image", "picture"]):
            return "image"
        if any(kw in text for kw in ["audio", "sound", "music", "listen"]):
            return "audio"
        if any(kw in text for kw in ["breaking", "news", "report", "announce"]):
            return "news"
        if "?" in text:
            return "question"
        return "text"

    def material_id(self, record: Record) -> int:
        material_type = self.classify(record)
        return self.config.material_map.get(material_type, 1)

    def energy_from_record(self, record: Record) -> float:
        text_len = len(record.content)
        base_energy = min(text_len / 100.0, 5.0)
        keywords = ["important", "critical", "urgent", "breaking", "novel"]
        keyword_boost = sum(0.5 for kw in keywords if kw in record.content.lower())
        return (base_energy + keyword_boost) * self.config.energy_scale

    def position_from_record(self, record: Record, tick: int = 0) -> tuple[int, int, int]:
        cx, cy, cz = self.config.center
        r = self.config.radius
        text_hash = hash(record.content) % 1000
        angle = (text_hash / 1000.0) * 2 * np.pi
        dist = (text_hash % r)
        x = int(cx + dist * np.cos(angle))
        z = int(cz + dist * np.sin(angle))
        x = max(0, min(x, self.config.grid_size[0] - 1))
        z = max(0, min(z, self.config.grid_size[2] - 1))
        return (x, cy, z)


class WorldPerception:
    def __init__(self, config: PerceptionConfig | None = None):
        self.config = config or PerceptionConfig()
        self._mapper = RecordToMaterial(self.config)
        self._events: list[PerceptionEvent] = []
        self._seen_hashes: set[int] = set()
        self._tick = 0

    def process_record(self, record: Record) -> PerceptionEvent | None:
        h = hash(record.content)
        if h in self._seen_hashes:
            return None
        self._seen_hashes.add(h)
        if len(self._seen_hashes) > self.config.dedup_window:
            self._seen_hashes = set(list(self._seen_hashes)[-self.config.dedup_window:])

        pos = self._mapper.position_from_record(record, self._tick)
        mat_id = self._mapper.material_id(record)
        energy = self._mapper.energy_from_record(record)

        event = PerceptionEvent(
            record=record,
            grid_pos=pos,
            material_type=mat_id,
            energy=energy,
            timestamp=self._tick,
            metadata={
                "material_class": self._mapper.classify(record),
                "text_length": len(record.content),
            },
        )
        self._events.append(event)
        return event

    def apply_to_grid(self, world_grid, events: list[PerceptionEvent] | None = None):
        events = events or self._events
        for event in events:
            x, y, z = event.grid_pos
            idx = world_grid.idx(x, y, z)
            world_grid.material[idx] = event.material_type
            world_grid.energy[idx] = event.energy

    def ingest_source(self, source: Source, store: Store | None = None) -> list[PerceptionEvent]:
        store = store or MemoryStore()
        collector = Collector(source, store)
        collector.collect()

        events = []
        for record in store.read_all():
            event = self.process_record(record)
            if event:
                events.append(event)
        return events

    def ingest_records(self, records: list[Record]) -> list[PerceptionEvent]:
        events = []
        for record in records:
            event = self.process_record(record)
            if event:
                events.append(event)
        return events

    def advance_tick(self):
        self._tick += 1

    def decay_energy(self, world_grid):
        solid_mask = world_grid.material != 0
        world_grid.energy[solid_mask] *= self.config.decay_rate

    def summary(self) -> dict:
        material_counts = {}
        for event in self._events:
            cls = event.metadata.get("material_class", "unknown")
            material_counts[cls] = material_counts.get(cls, 0) + 1

        return {
            "total_events": len(self._events),
            "unique_records": len(self._seen_hashes),
            "tick": self._tick,
            "material_counts": material_counts,
            "avg_energy": float(np.mean([e.energy for e in self._events])) if self._events else 0.0,
        }

    @property
    def events(self) -> list[PerceptionEvent]:
        return list(self._events)

    @property
    def tick(self) -> int:
        return self._tick


class PerceptionFeed:
    def __init__(self, perception: WorldPerception, source: Source):
        self.perception = perception
        self.source = source
        self._collector = Collector(source, MemoryStore())

    def run(self) -> list[PerceptionEvent]:
        self._collector.collect()
        records = list(self._collector.store.read_all())
        return self.perception.ingest_records(records)

    def run_and_apply(self, world_grid) -> list[PerceptionEvent]:
        events = self.run()
        self.perception.apply_to_grid(world_grid, events)
        return events


class PerceptionScheduler:
    def __init__(self, perception: WorldPerception):
        self.perception = perception
        self._feeds: list[PerceptionFeed] = []

    def add_feed(self, feed: PerceptionFeed):
        self._feeds.append(feed)

    def tick_all(self, world_grid) -> list[PerceptionEvent]:
        all_events = []
        for feed in self._feeds:
            events = feed.run_and_apply(world_grid)
            all_events.extend(events)
        self.perception.advance_tick()
        self.perception.decay_energy(world_grid)
        return all_events

    def summary(self) -> dict:
        return {
            "feeds": len(self._feeds),
            **self.perception.summary(),
        }
