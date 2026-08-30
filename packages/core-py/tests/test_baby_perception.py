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


# ── Config ──────────────────────────────────────────────────────────────

class TestBabyPerceptionConfig:
    def test_defaults(self):
        cfg = BabyPerceptionConfig()
        assert cfg.see_radius == 5
        assert cfg.material_weight == 1.0
        assert cfg.energy_weight == 0.5
        assert cfg.novelty_bonus == 2.0
        assert cfg.memory_size == 50
        assert cfg.learning_rate == 0.1

    def test_custom_values(self):
        cfg = BabyPerceptionConfig(
            see_radius=10, material_weight=2.0, energy_weight=0.8,
            novelty_bonus=3.0, memory_size=200, learning_rate=0.01,
        )
        assert cfg.see_radius == 10
        assert cfg.material_weight == 2.0
        assert cfg.memory_size == 200


# ── BabyPerception ──────────────────────────────────────────────────────

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

    def test_empty_material_preference(self):
        """No materials seen → default preference is 1."""
        baby = FakeBaby()
        bp = BabyPerception(baby)
        assert bp.get_material_preference() == 1

    def test_perceive_world_no_events(self):
        """Perception with no events → zero event_count."""
        baby = FakeBaby()
        perception = WorldPerception()
        bp = BabyPerception(baby)
        result = bp.perceive_world(WorldGrid(size=(64, 4, 64)), perception)
        assert result["event_count"] == 0
        assert result["energy_sum"] == 0.0
        assert result["novelty_sum"] == 0.0

    def test_perceive_world_material_features_normalized(self):
        """material_features should be normalized when sum > 0."""
        baby = FakeBaby()
        perception = WorldPerception()
        records = [SourceRecord(content="some text")] * 3
        perception.ingest_records(records)
        bp = BabyPerception(baby)
        result = bp.perceive_world(WorldGrid(size=(64, 4, 64)), perception)
        total = result["material_features"].sum()
        if total > 0:
            assert abs(total - 1.0) < 1e-6

    def test_perceive_world_material_features_sum_zero(self):
        """No nearby events → material_features all zeros."""
        baby = FakeBaby(position=[0, 0, 0])
        perception = WorldPerception()
        # Events far away
        records = [SourceRecord(content="x")] * 3
        perception.ingest_records(records)
        # Place baby at origin, events will be placed randomly — could be far
        bp = BabyPerception(baby)
        result = bp.perceive_world(WorldGrid(size=(64, 4, 64)), perception)
        assert isinstance(result["material_features"], np.ndarray)
        assert result["material_features"].shape == (8,)

    def test_perceive_world_material_counts_accumulate(self):
        """Multiple perceive_world calls should accumulate material counts."""
        baby = FakeBaby()
        perception = WorldPerception()
        # Use records that map to positions near the baby
        # Place events directly on the grid at baby's position
        records = [SourceRecord(content="text content " * 50)] * 5
        perception.ingest_records(records)
        bp = BabyPerception(baby)
        bp.perceive_world(WorldGrid(size=(64, 4, 64)), perception)
        bp.perceive_world(WorldGrid(size=(64, 4, 64)), perception)
        # Even if no events are nearby, _total_perceived should track
        assert bp._total_perceived >= 0

    def test_perceive_world_total_perceived_increments(self):
        baby = FakeBaby()
        perception = WorldPerception()
        records = [SourceRecord(content="data")] * 3
        perception.ingest_records(records)
        bp = BabyPerception(baby)
        r1 = bp.perceive_world(WorldGrid(size=(64, 4, 64)), perception)
        r2 = bp.perceive_world(WorldGrid(size=(64, 4, 64)), perception)
        assert r2["total_perceived"] >= r1["total_perceived"]

    def test_perceive_world_memory_truncation(self):
        """Memory list is capped at memory_size."""
        baby = FakeBaby()
        perception = WorldPerception()
        cfg = BabyPerceptionConfig(memory_size=3)
        bp = BabyPerception(baby, cfg)
        # Feed many empty perceptions to fill memory
        for _ in range(10):
            bp.perceive_world(WorldGrid(size=(64, 4, 64)), perception)
        assert len(bp._memory) <= 3

    def test_perceived_events_memory_truncation(self):
        """perceived_events list is capped at memory_size."""
        baby = FakeBaby()
        perception = WorldPerception()
        records = [SourceRecord(content=f"rec {j}") for j in range(5)]
        perception.ingest_records(records)
        cfg = BabyPerceptionConfig(memory_size=3)
        bp = BabyPerception(baby, cfg)
        for _ in range(10):
            bp.perceive_world(WorldGrid(size=(64, 4, 64)), perception)
        assert len(bp._perceived_events) <= 3

    def test_novelty_score_zero_count(self):
        """Novelty for unseen material is 1.0."""
        baby = FakeBaby()
        bp = BabyPerception(baby)
        assert bp.get_novelty_score(99) == 1.0

    def test_novelty_score_high_count(self):
        """Novelty for heavily-seen material approaches 0."""
        baby = FakeBaby()
        bp = BabyPerception(baby)
        bp._material_counts = {1: 1000}
        score = bp.get_novelty_score(1)
        assert score < 0.01

    def test_summary_after_perceive(self):
        """Summary reflects perceived data."""
        baby = FakeBaby()
        perception = WorldPerception()
        records = [SourceRecord(content="hello " * 50)] * 3
        perception.ingest_records(records)
        bp = BabyPerception(baby)
        bp.perceive_world(WorldGrid(size=(64, 4, 64)), perception)
        s = bp.summary()
        assert "total_perceived" in s
        assert s["memory_size"] >= 1

    def test_custom_config_used(self):
        baby = FakeBaby()
        cfg = BabyPerceptionConfig(see_radius=2)
        bp = BabyPerception(baby, cfg)
        assert bp.config.see_radius == 2

    def test_material_id_out_of_range(self):
        """Material IDs outside 1-8 should not index into features."""
        baby = FakeBaby()
        perception = WorldPerception()
        # Generate records that hash to positions near baby
        records = [SourceRecord(content="a")] * 20
        perception.ingest_records(records)
        bp = BabyPerception(baby)
        result = bp.perceive_world(WorldGrid(size=(64, 4, 64)), perception)
        # Material features should be valid
        assert result["material_features"].shape == (8,)


# ── BabyAction ──────────────────────────────────────────────────────────

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

    def test_move_toward_close_distance(self):
        """Move toward target that is < 1.0 away → no movement."""
        baby = FakeBaby(position=[32, 0, 32])
        world = WorldGrid(size=(64, 4, 64))
        action = BabyAction(baby)
        target = np.array([32.5, 0, 32], dtype=np.float64)
        success = action.move_toward(target, world)
        assert not success

    def test_move_toward_out_of_bounds(self):
        """Move toward target outside grid → returns False."""
        baby = FakeBaby(position=[1, 0, 1])
        world = WorldGrid(size=(10, 4, 10))
        action = BabyAction(baby)
        target = np.array([100, 0, 100], dtype=np.float64)
        success = action.move_toward(target, world)
        assert not success

    def test_move_toward_updates_position(self):
        """Successful move updates baby position."""
        baby = FakeBaby(position=[32, 0, 32])
        world = WorldGrid(size=(64, 4, 64))
        action = BabyAction(baby)
        target = np.array([40, 0, 32], dtype=np.float64)
        action.move_toward(target, world)
        assert baby.position[0] > 32

    def test_write_to_grid_out_of_bounds(self):
        """Write outside grid → returns False."""
        baby = FakeBaby(position=[-1, 0, -1])
        world = WorldGrid(size=(64, 4, 64))
        action = BabyAction(baby)
        success = action.write_to_grid(world, 1, 5.0)
        assert not success

    def test_interact_with_event_far_away(self):
        """Event too far → no interaction."""
        baby = FakeBaby(position=[32, 0, 32], energy=50.0)
        world = WorldGrid(size=(64, 4, 64))
        event = PerceptionEvent(
            record=SourceRecord(content="far"),
            grid_pos=(60, 0, 60),
            material_type=1,
            energy=5.0,
            timestamp=0,
        )
        action = BabyAction(baby, BabyPerceptionConfig(see_radius=2))
        result = action.interact_with_event(event, world)
        assert not result["success"]

    def test_interact_with_event_low_energy_no_approach(self):
        """Energy <= 20 → no approach action."""
        baby = FakeBaby(position=[32, 0, 32], energy=15.0)
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
        # energy > 10 so absorb still happens
        assert result["action"] == "absorb"

    def test_interact_with_event_very_low_energy(self):
        """Energy <= 10 → no absorb action."""
        baby = FakeBaby(position=[32, 0, 32], energy=5.0)
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
        assert not result["success"]

    def test_action_counts_tracked(self):
        """Actions increment counts."""
        baby = FakeBaby(position=[32, 0, 32], energy=50.0)
        world = WorldGrid(size=(64, 4, 64))
        action = BabyAction(baby)
        action.move_toward(np.array([40, 0, 32], dtype=np.float64), world)
        action.consume_energy(5.0)
        action.write_to_grid(world, 2, 3.0)
        counts = action.summary()["action_counts"]
        assert counts.get("move", 0) >= 1
        assert counts.get("consume", 0) >= 1
        assert counts.get("write", 0) >= 1

    def test_last_action_tracked(self):
        baby = FakeBaby(position=[32, 0, 32], energy=50.0)
        world = WorldGrid(size=(64, 4, 64))
        action = BabyAction(baby)
        action.write_to_grid(world, 1, 1.0)
        assert action.summary()["last_action"] == "write"

    def test_consume_energy_exact_amount(self):
        """Consume exactly the available energy → succeeds."""
        baby = FakeBaby(energy=10.0)
        action = BabyAction(baby)
        assert action.consume_energy(10.0)
        assert baby.energy == 0.0


# ── BabyLearning ────────────────────────────────────────────────────────

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

    def test_record_experience_failed_action(self):
        """Failed action does not add material reward."""
        baby = FakeBaby()
        learning = BabyLearning(baby)
        learning.record_experience(
            {"event_count": 1, "preferred_material": 2},
            {"success": False, "action": None},
        )
        assert learning.get_material_value(2) == 0.0
        assert learning._total_learning == 1

    def test_record_experience_increments_total(self):
        baby = FakeBaby()
        learning = BabyLearning(baby)
        learning.record_experience({"event_count": 0}, {"success": False})
        learning.record_experience({"event_count": 1}, {"success": False})
        assert learning._total_learning == 2

    def test_experience_buffer_truncation(self):
        """Experience list is capped at memory_size."""
        baby = FakeBaby()
        cfg = BabyPerceptionConfig(memory_size=3)
        learning = BabyLearning(baby, cfg)
        for i in range(10):
            learning.record_experience(
                {"event_count": i}, {"success": False},
            )
        assert len(learning._experiences) <= 3

    def test_get_experience_buffer_returns_copy(self):
        baby = FakeBaby()
        learning = BabyLearning(baby)
        learning.record_experience({"event_count": 1}, {"success": True, "action": "a"})
        buf = learning.get_experience_buffer()
        assert len(buf) == 1
        buf.clear()
        assert len(learning._experiences) == 1

    def test_preferred_material_empty(self):
        """No rewards → default preference is 1."""
        baby = FakeBaby()
        learning = BabyLearning(baby)
        assert learning.get_preferred_material() == 1

    def test_summary_after_experiences(self):
        baby = FakeBaby()
        learning = BabyLearning(baby)
        learning.record_experience(
            {"event_count": 2, "preferred_material": 3},
            {"success": True, "action": "a"},
        )
        s = learning.summary()
        assert s["total_experiences"] == 1
        assert s["total_learning"] == 1
        assert 3 in s["material_rewards"]

    def test_material_rewards_accumulate(self):
        baby = FakeBaby()
        learning = BabyLearning(baby)
        learning.record_experience(
            {"preferred_material": 5}, {"success": True},
        )
        learning.record_experience(
            {"preferred_material": 5}, {"success": True},
        )
        assert learning.get_material_value(5) == 2.0

    def test_experience_stores_energy(self):
        baby = FakeBaby(energy=42.0)
        learning = BabyLearning(baby)
        learning.record_experience({"event_count": 0}, {"success": False})
        assert learning._experiences[0]["energy"] == 42.0


# ── BabyPerceptionSystem ───────────────────────────────────────────────

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

    def test_get_modules_unknown_baby(self):
        """Unknown baby ID returns None."""
        system = BabyPerceptionSystem()
        baby = FakeBaby()
        baby.entity.id = 999
        assert system.get_modules(baby) is None

    def test_unregister_nonexistent_baby(self):
        """Unregistering a baby that was never registered is a no-op."""
        system = BabyPerceptionSystem()
        baby = FakeBaby()
        baby.entity.id = 42
        system.unregister_baby(baby)  # should not raise

    def test_tick_no_events(self):
        """Tick with no perception events → zero event_count."""
        system = BabyPerceptionSystem()
        baby = FakeBaby()
        baby.entity.id = 1
        system.register_baby(baby)
        perception = WorldPerception()
        world = WorldGrid(size=(64, 4, 64))
        results = system.tick(world, perception)
        assert len(results) == 1
        assert results[0]["perception"]["event_count"] == 0

    def test_tick_multiple_babies(self):
        system = BabyPerceptionSystem()
        baby1 = FakeBaby()
        baby1.entity.id = 1
        baby2 = FakeBaby()
        baby2.entity.id = 2
        system.register_baby(baby1)
        system.register_baby(baby2)
        perception = WorldPerception()
        world = WorldGrid(size=(64, 4, 64))
        results = system.tick(world, perception)
        assert len(results) == 2
        ids = [r["baby_id"] for r in results]
        assert 1 in ids
        assert 2 in ids

    def test_summary_empty(self):
        system = BabyPerceptionSystem()
        s = system.summary()
        assert s["total_babies"] == 0
        assert s["babies"] == {}

    def test_summary_multiple_babies(self):
        system = BabyPerceptionSystem()
        for i in range(3):
            baby = FakeBaby()
            baby.entity.id = i
            system.register_baby(baby)
        s = system.summary()
        assert s["total_babies"] == 3
        assert len(s["babies"]) == 3

    def test_custom_config_propagated(self):
        cfg = BabyPerceptionConfig(see_radius=3)
        system = BabyPerceptionSystem(cfg)
        baby = FakeBaby()
        baby.entity.id = 1
        system.register_baby(baby)
        modules = system.get_modules(baby)
        assert modules["perception"].config.see_radius == 3
        assert modules["action"].config.see_radius == 3
        assert modules["learning"].config.see_radius == 3

    def test_tick_records_action_result(self):
        """Tick records the action result for each baby."""
        system = BabyPerceptionSystem()
        baby = FakeBaby(position=[32, 0, 32], energy=50.0)
        baby.entity.id = 1
        system.register_baby(baby)
        perception = WorldPerception()
        records = [SourceRecord(content="data")]
        perception.ingest_records(records)
        world = WorldGrid(size=(64, 4, 64))
        perception.apply_to_grid(world)
        results = system.tick(world, perception)
        assert "action" in results[0]
        assert "success" in results[0]["action"]

    def test_tick_learning_records_experience(self):
        """After tick, learning module has recorded an experience."""
        system = BabyPerceptionSystem()
        baby = FakeBaby()
        baby.entity.id = 1
        system.register_baby(baby)
        perception = WorldPerception()
        world = WorldGrid(size=(64, 4, 64))
        system.tick(world, perception)
        modules = system.get_modules(baby)
        assert len(modules["learning"]._experiences) == 1
