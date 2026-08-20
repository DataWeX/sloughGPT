import numpy as np
import pytest

from domains.collections.baby_perception import (
    BabyPerceptionConfig, BabyPerception, BabyAction, BabyLearning,
    BabyPerceptionSystem,
)
from domains.collections.perception import WorldPerception, PerceptionEvent, Record
from domains.collections.sources import Record as SourceRecord
from domains.shell.simulation import WorldGrid, SimBaby, WorldParams


class FakeBaby:
    def __init__(self, position=None, energy=50.0):
        self.entity = type('Entity', (), {'id': 1, 'entity_type': 1})()
        self.position = np.array(position if position else [32, 0, 32], dtype=np.float64)
        self.energy = energy
        self.alive = True


class TestBabyPerception:
    def test_perceive_world(self):
        baby = FakeBaby()
        perception = WorldPerception()
        records = [SourceRecord(content=f"Record {i}") for i in range(5)]
        perception.ingest_records(records)

        bp = BabyPerception(baby)
        result = bp.perceive_world(WorldGrid(size=(64, 4, 64)), perception)
        assert "material_features" in result
        assert "energy_sum" in result
        assert "event_count" in result

    def test_material_preference(self):
        baby = FakeBaby()
        bp = BabyPerception(baby)
        bp._material_counts = {1: 5, 2: 3, 3: 1}
        assert bp.get_material_preference() == 1

    def test_novelty_score(self):
        baby = FakeBaby()
        bp = BabyPerception(baby)
        bp._material_counts = {1: 10}
        score = bp.get_novelty_score(1)
        assert score < 1.0
        assert bp.get_novelty_score(2) == 1.0

    def test_summary(self):
        baby = FakeBaby()
        bp = BabyPerception(baby)
        s = bp.summary()
        assert "total_perceived" in s
        assert "material_counts" in s


class TestBabyAction:
    def test_move_toward(self):
        baby = FakeBaby(position=[32, 0, 32])
        world = WorldGrid(size=(64, 4, 64))
        action = BabyAction(baby)
        target = np.array([33, 0, 32], dtype=np.float64)
        success = action.move_toward(target, world)
        assert success
        assert baby.position[0] > 32

    def test_consume_energy(self):
        baby = FakeBaby(energy=50.0)
        action = BabyAction(baby)
        success = action.consume_energy(10.0)
        assert success
        assert baby.energy == 40.0

    def test_consume_energy_insufficient(self):
        baby = FakeBaby(energy=5.0)
        action = BabyAction(baby)
        success = action.consume_energy(10.0)
        assert not success
        assert baby.energy == 5.0

    def test_write_to_grid(self):
        baby = FakeBaby(position=[32, 0, 32])
        world = WorldGrid(size=(64, 4, 64))
        action = BabyAction(baby)
        success = action.write_to_grid(world, 1, 5.0)
        assert success
        assert world.material[world.idx(32, 0, 32)] == 1

    def test_interact_with_event(self):
        baby = FakeBaby(position=[32, 0, 32], energy=50.0)
        world = WorldGrid(size=(64, 4, 64))
        event = PerceptionEvent(
            record=SourceRecord(content="test"),
            grid_pos=(33, 0, 32),
            material_type=1,
            energy=2.0,
            timestamp=0,
        )
        action = BabyAction(baby)
        result = action.interact_with_event(event, world)
        assert result["success"]

    def test_summary(self):
        baby = FakeBaby()
        action = BabyAction(baby)
        s = action.summary()
        assert "last_action" in s
        assert "action_counts" in s


class TestBabyLearning:
    def test_record_experience(self):
        baby = FakeBaby()
        learning = BabyLearning(baby)
        perc_result = {"event_count": 3, "energy_sum": 5.0}
        action_result = {"success": True, "action": "approach"}
        learning.record_experience(perc_result, action_result)
        assert len(learning._experiences) == 1

    def test_material_value(self):
        baby = FakeBaby()
        learning = BabyLearning(baby)
        learning._material_rewards = {1: 5.0, 2: 3.0}
        assert learning.get_material_value(1) == 5.0
        assert learning.get_material_value(3) == 0.0

    def test_preferred_material(self):
        baby = FakeBaby()
        learning = BabyLearning(baby)
        learning._material_rewards = {1: 5.0, 2: 3.0}
        assert learning.get_preferred_material() == 1

    def test_summary(self):
        baby = FakeBaby()
        learning = BabyLearning(baby)
        s = learning.summary()
        assert "total_experiences" in s
        assert "material_rewards" in s


class TestBabyPerceptionSystem:
    def test_register_baby(self):
        system = BabyPerceptionSystem()
        baby = FakeBaby()
        baby.entity.id = 1
        system.register_baby(baby)
        assert 1 in system._babies

    def test_unregister_baby(self):
        system = BabyPerceptionSystem()
        baby = FakeBaby()
        baby.entity.id = 1
        system.register_baby(baby)
        system.unregister_baby(baby)
        assert 1 not in system._babies

    def test_get_modules(self):
        system = BabyPerceptionSystem()
        baby = FakeBaby()
        baby.entity.id = 1
        system.register_baby(baby)
        modules = system.get_modules(baby)
        assert modules is not None
        assert "perception" in modules
        assert "action" in modules
        assert "learning" in modules

    def test_tick(self):
        system = BabyPerceptionSystem()
        baby = FakeBaby()
        baby.entity.id = 1
        system.register_baby(baby)

        perception = WorldPerception()
        records = [SourceRecord(content=f"Record {i}") for i in range(5)]
        perception.ingest_records(records)

        world = WorldGrid(size=(64, 4, 64))
        perception.apply_to_grid(world)

        results = system.tick(world, perception)
        assert len(results) == 1
        assert "perception" in results[0]
        assert "action" in results[0]

    def test_tick_dead_baby(self):
        system = BabyPerceptionSystem()
        baby = FakeBaby()
        baby.entity.id = 1
        baby.alive = False
        system.register_baby(baby)

        perception = WorldPerception()
        world = WorldGrid(size=(64, 4, 64))
        results = system.tick(world, perception)
        assert len(results) == 0

    def test_summary(self):
        system = BabyPerceptionSystem()
        baby = FakeBaby()
        baby.entity.id = 1
        system.register_baby(baby)
        s = system.summary()
        assert s["total_babies"] == 1
        assert 1 in s["babies"]
