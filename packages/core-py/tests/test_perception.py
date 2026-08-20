import numpy as np
import pytest

from domains.collections.perception import (
    PerceptionConfig, PerceptionEvent, RecordToMaterial,
    WorldPerception, PerceptionFeed, PerceptionScheduler,
)
from domains.collections.sources import Record, GeneratorSource
from domains.collections.stores import MemoryStore
from domains.shell.simulation import WorldGrid


class TestRecordToMaterial:
    def test_classify_text(self):
        mapper = RecordToMaterial()
        record = Record(content="Hello world, this is a test")
        assert mapper.classify(record) == "text"

    def test_classify_code(self):
        mapper = RecordToMaterial()
        record = Record(content="def hello():\n    print('hi')")
        assert mapper.classify(record) == "code"

    def test_classify_question(self):
        mapper = RecordToMaterial()
        record = Record(content="What is the meaning of life?")
        assert mapper.classify(record) == "question"

    def test_classify_news(self):
        mapper = RecordToMaterial()
        record = Record(content="Breaking news: major event announced today")
        assert mapper.classify(record) == "news"

    def test_material_id(self):
        mapper = RecordToMaterial()
        record = Record(content="Hello world")
        mat_id = mapper.material_id(record)
        assert 1 <= mat_id <= 8

    def test_energy_from_record(self):
        mapper = RecordToMaterial()
        short = Record(content="Hi")
        long = Record(content="This is a very long text with important content")
        assert mapper.energy_from_record(long) > mapper.energy_from_record(short)

    def test_position_in_bounds(self):
        config = PerceptionConfig(grid_size=(64, 4, 64), radius=15)
        mapper = RecordToMaterial(config)
        for i in range(100):
            record = Record(content=f"Test record {i}")
            pos = mapper.position_from_record(record)
            assert 0 <= pos[0] < 64
            assert 0 <= pos[2] < 64

    def test_position_deterministic(self):
        mapper = RecordToMaterial()
        record = Record(content="Same text")
        pos1 = mapper.position_from_record(record)
        pos2 = mapper.position_from_record(record)
        assert pos1 == pos2


class TestWorldPerception:
    def test_process_record(self):
        perception = WorldPerception()
        record = Record(content="Hello world")
        event = perception.process_record(record)
        assert event is not None
        assert event.record == record
        assert event.energy > 0

    def test_dedup(self):
        perception = WorldPerception()
        record = Record(content="Hello world")
        e1 = perception.process_record(record)
        e2 = perception.process_record(record)
        assert e1 is not None
        assert e2 is None

    def test_apply_to_grid(self):
        perception = WorldPerception()
        world = WorldGrid(size=(16, 4, 16))
        records = [Record(content=f"Record {i}") for i in range(5)]
        events = perception.ingest_records(records)
        perception.apply_to_grid(world, events)
        assert np.any(world.material != 0)

    def test_ingest_records(self):
        perception = WorldPerception()
        records = [Record(content=f"Record {i}") for i in range(10)]
        events = perception.ingest_records(records)
        assert len(events) == 10

    def test_tick(self):
        perception = WorldPerception()
        assert perception.tick == 0
        perception.advance_tick()
        assert perception.tick == 1

    def test_decay_energy(self):
        perception = WorldPerception()
        world = WorldGrid(size=(16, 4, 16))
        world.energy[world.idx(8, 0, 8)] = 10.0
        world.material[world.idx(8, 0, 8)] = 1
        perception.decay_energy(world)
        assert world.energy[world.idx(8, 0, 8)] < 10.0

    def test_summary(self):
        perception = WorldPerception()
        records = [Record(content=f"Record {i}") for i in range(5)]
        perception.ingest_records(records)
        s = perception.summary()
        assert s["total_events"] == 5
        assert s["unique_records"] == 5

    def test_events_property(self):
        perception = WorldPerception()
        records = [Record(content=f"Record {i}") for i in range(3)]
        perception.ingest_records(records)
        assert len(perception.events) == 3


class TestPerceptionFeed:
    def test_run(self):
        records = [Record(content=f"Feed record {i}") for i in range(5)]
        source = GeneratorSource(lambda: iter(records))
        perception = WorldPerception()
        feed = PerceptionFeed(perception, source)
        events = feed.run()
        assert len(events) == 5

    def test_run_and_apply(self):
        records = [Record(content=f"Feed record {i}") for i in range(5)]
        source = GeneratorSource(lambda: iter(records))
        perception = WorldPerception()
        feed = PerceptionFeed(perception, source)
        world = WorldGrid(size=(16, 4, 16))
        events = feed.run_and_apply(world)
        assert len(events) == 5
        assert np.any(world.material != 0)


class TestPerceptionScheduler:
    def test_add_feed(self):
        scheduler = PerceptionScheduler(WorldPerception())
        source = GeneratorSource(lambda: iter([Record(content="test")]))
        feed = PerceptionFeed(WorldPerception(), source)
        scheduler.add_feed(feed)
        assert len(scheduler._feeds) == 1

    def test_tick_all(self):
        scheduler = PerceptionScheduler(WorldPerception())
        records1 = [Record(content=f"A {i}") for i in range(3)]
        records2 = [Record(content=f"B {i}") for i in range(3)]
        source1 = GeneratorSource(lambda: iter(records1))
        source2 = GeneratorSource(lambda: iter(records2))
        feed1 = PerceptionFeed(WorldPerception(), source1)
        feed2 = PerceptionFeed(WorldPerception(), source2)
        scheduler.add_feed(feed1)
        scheduler.add_feed(feed2)
        world = WorldGrid(size=(16, 4, 16))
        events = scheduler.tick_all(world)
        assert len(events) == 6

    def test_summary(self):
        scheduler = PerceptionScheduler(WorldPerception())
        source = GeneratorSource(lambda: iter([Record(content="test")]))
        feed = PerceptionFeed(WorldPerception(), source)
        scheduler.add_feed(feed)
        s = scheduler.summary()
        assert s["feeds"] == 1
        assert s["total_events"] == 0
