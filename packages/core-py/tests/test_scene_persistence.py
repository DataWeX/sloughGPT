"""
Tests for scene persistence — save/load of the world grid, entities, babies.

A scene snapshot must be:
  - lossless: restore reproduces the world arrays and every baby bit-for-bit
  - JSON-safe: ``to_dict()`` output survives json.dumps/loads
  - RNG-neutral: restoring consumes no random numbers, so a resumed run is
    bit-identical to an uninterrupted run
  - collision-free: new babies spawned after a restore never reuse an old id
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from domains.shell.memory import EpisodicMemory
from domains.shell.simulation import (
    Entity,
    EntityType,
    SimBaby,
    SimScene,
    Simulation,
    WorldGrid,
    WorldParams,
)

SMALL = (8, 8, 8)


def _params(grid: tuple[int, int, int] = SMALL, agents: int = 2,
            energy: float = 80.0) -> WorldParams:
    return WorldParams(grid_size=grid, start_agents=agents, start_energy=energy)


def _made_scene(seed: int = 7, ticks: int = 3) -> SimScene:
    """Seed RNG, spawn babies, run ticks, and return the scene."""
    np.random.seed(seed)
    scene = SimScene(_params())
    scene.spawn_babies()
    Simulation(scene, max_ticks=ticks).run()
    return scene


class TestWorldGridPersistence:
    def test_round_trip_preserves_all_arrays(self):
        g = WorldGrid(SMALL)
        g.place_material(1, 2, 3, 4, energy=12.5, temperature=33.5)
        g.signal[g.idx(0, 0, 0)] = 0.75
        r = WorldGrid.from_dict(g.to_dict())
        assert r.size == SMALL
        assert r.material.dtype == g.material.dtype
        assert np.array_equal(r.material, g.material)
        assert np.array_equal(r.energy, g.energy)
        assert np.array_equal(r.temperature, g.temperature)
        assert np.array_equal(r.signal, g.signal)
        assert r.total == g.total

    def test_fresh_grid_round_trip_equals_fresh_grid(self):
        g = WorldGrid(SMALL)
        r = WorldGrid.from_dict(g.to_dict())
        assert np.array_equal(r.material, g.material)
        assert np.array_equal(r.energy, g.energy)
        assert np.array_equal(r.temperature, g.temperature)
        assert np.array_equal(r.signal, g.signal)

    def test_restore_uses_stored_size(self):
        g = WorldGrid((4, 4, 4))
        r = WorldGrid.from_dict(g.to_dict())
        assert r.size == (4, 4, 4)
        assert r.total == 64

    def test_length_mismatch_raises(self):
        data = WorldGrid(SMALL).to_dict()
        data["energy"] = data["energy"][:-1]
        with pytest.raises(ValueError):
            WorldGrid.from_dict(data)


class TestEntityPersistence:
    def test_round_trip_preserves_state(self):
        e = Entity(
            id=7,
            position=np.array([1.5, 2.0, -3.25]),
            energy=42.0,
            entity_type=EntityType.AGENT,
            alive=True,
        )
        r = Entity.from_dict(e.to_dict())
        assert r.id == 7
        assert np.array_equal(r.position, e.position)
        assert r.energy == 42.0
        assert r.entity_type == EntityType.AGENT
        assert r.alive is True

    def test_round_trip_preserves_dead_flag(self):
        e = Entity(id=1, position=np.zeros(3), energy=0.0, alive=False)
        r = Entity.from_dict(e.to_dict())
        assert r.alive is False


class TestPerceptronPersistence:
    def test_round_trip_preserves_weights(self):
        b = SimBaby(position=np.zeros(3), params=_params())
        snap = b.perceptron_cells.to_dict()
        r = b.perceptron_cells.from_dict(snap)
        assert np.array_equal(r.W, b.perceptron_cells.W)
        assert np.array_equal(r.b, b.perceptron_cells.b)

    def test_forward_output_identical_after_restore(self):
        b = SimBaby(position=np.zeros(3), params=_params())
        x = np.array([0.1, 0.2, 0.3, 0.4, 0.0], dtype=np.float32)
        restored = b.perceptron_cells.from_dict(b.perceptron_cells.to_dict())
        assert np.array_equal(restored.forward(x), b.perceptron_cells.forward(x))


class TestMemoryPersistence:
    def test_round_trip_preserves_episodes_and_head(self):
        m = EpisodicMemory(capacity=3)
        for i in range(5):
            m.record(np.array([i] * 4, dtype=np.float32), (1.0, 0.0, 0.5),
                     float(i), tick=i)
        assert m.is_full
        r = EpisodicMemory.from_dict(m.to_dict())
        assert r.capacity == m.capacity
        assert r._head == m._head
        assert len(r) == len(m)
        assert [e.reward for e in r._chronological()] == [
            e.reward for e in m._chronological()
        ]

    def test_eviction_continues_at_same_slot_after_restore(self):
        m = EpisodicMemory(capacity=3)
        for i in range(4):
            m.record(np.zeros(4, dtype=np.float32), (1.0,), float(i), tick=i)
        r = EpisodicMemory.from_dict(m.to_dict())
        m.record(np.zeros(4, dtype=np.float32), (1.0,), 99.0, tick=4)
        r.record(np.zeros(4, dtype=np.float32), (1.0,), 99.0, tick=4)
        assert m.to_dict() == r.to_dict()

    def test_to_dict_is_json_safe(self):
        m = EpisodicMemory(capacity=2)
        m.record(np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32),
                 (1.0, 0.0, 0.5), 3.0, tick=1)
        r = EpisodicMemory.from_dict(json.loads(json.dumps(m.to_dict())))
        assert r.to_dict() == m.to_dict()


class TestBabyPersistence:
    def test_round_trip_preserves_entity_and_weights(self):
        np.random.seed(3)
        baby = SimBaby(position=np.array([1.0, 2.0, 3.0]), params=_params())
        baby.group_id = 4
        baby._total_ticks = 12
        r = SimBaby.from_dict(baby.to_dict(), params=baby.params)
        assert np.array_equal(r.entity.position, baby.entity.position)
        assert r.entity.id == baby.entity.id
        assert r.entity.energy == baby.entity.energy
        assert r.entity.entity_type == EntityType.AGENT
        assert r.tick_count == 12
        assert r.group_id == 4
        assert np.array_equal(r.perceptron_cells.W, baby.perceptron_cells.W)
        assert np.array_equal(r.perceptron_body.W, baby.perceptron_body.W)
        assert np.array_equal(r.perceptron_entity.W, baby.perceptron_entity.W)

    def test_round_trip_preserves_memory(self):
        np.random.seed(3)
        baby = SimBaby(position=np.zeros(3), params=_params())
        baby.memory.record(np.zeros(4, dtype=np.float32), (1.0,), 5.0, tick=1)
        r = SimBaby.from_dict(baby.to_dict(), params=baby.params)
        assert r.memory.to_dict() == baby.memory.to_dict()

    def test_external_entity_instance_is_used(self):
        np.random.seed(3)
        baby = SimBaby(position=np.zeros(3), params=_params())
        entity = Entity.from_dict(baby.entity.to_dict())
        r = SimBaby.from_dict(baby.to_dict(), params=baby.params, entity=entity)
        assert r.entity is entity


class TestScenePersistence:
    def test_round_trip_preserves_scene_state(self):
        scene = _made_scene(seed=11, ticks=5)
        snap = scene.to_dict()
        assert snap["tick"] == 5
        r = SimScene.from_dict(snap)
        assert r.tick == 5
        assert r.params.grid_size == scene.params.grid_size
        assert np.array_equal(r.world.material, scene.world.material)
        assert np.array_equal(r.world.energy, scene.world.energy)
        assert np.array_equal(r.world.temperature, scene.world.temperature)
        assert np.array_equal(r.world.signal, scene.world.signal)
        assert len(r.entities) == len(scene.entities)
        assert len(r.babies) == len(scene.babies)
        for rb, sb in zip(r.babies, scene.babies):
            assert rb.entity.id == sb.entity.id
            assert rb.entity.energy == sb.entity.energy
            assert rb.tick_count == sb.tick_count
            assert np.array_equal(rb.perceptron_cells.W, sb.perceptron_cells.W)
            assert rb.memory.to_dict() == sb.memory.to_dict()

    def test_baby_entity_identity_restored(self):
        scene = _made_scene(seed=5, ticks=2)
        r = SimScene.from_dict(scene.to_dict())
        for baby in r.babies:
            assert any(baby.entity is e for e in r.entities)
        assert r.get_baby(scene.babies[0].entity.id) is r.babies[0]

    def test_non_baby_entities_restored(self):
        scene = SimScene(_params(agents=0))
        extra = Entity(id=901, position=np.array([1.0, 1.0, 1.0]),
                       energy=50.0, entity_type=EntityType.LIGHT)
        scene.entities.append(extra)
        r = SimScene.from_dict(scene.to_dict())
        restored = next(e for e in r.entities if e.id == 901)
        assert restored.entity_type == EntityType.LIGHT
        assert restored.energy == 50.0

    def test_to_dict_is_json_safe_and_round_trips(self):
        scene = _made_scene(seed=2, ticks=4)
        snap = json.loads(json.dumps(scene.to_dict()))
        r = SimScene.from_dict(snap)
        assert r.tick == 4
        assert np.array_equal(r.world.energy, scene.world.energy)

    def test_new_babies_do_not_collide_with_restored_ids(self):
        scene = _made_scene(seed=9, ticks=1)
        max_id = max(b.entity.id for b in scene.babies)
        r = SimScene.from_dict(scene.to_dict())
        r.spawn_babies(count=1)
        new_ids = [b.entity.id for b in r.babies
                   if b.entity.id > max_id]
        assert len(new_ids) == 1
        assert len({b.entity.id for b in r.babies}) == len(r.babies)

    def test_restore_consumes_no_rng(self):
        scene = _made_scene(seed=4, ticks=2)
        snap = scene.to_dict()
        state_before = np.random.get_state()
        SimScene.from_dict(snap)
        state_after = np.random.get_state()
        assert state_before[0] == state_after[0]
        assert np.array_equal(state_before[1], state_after[1])

    def test_resume_matches_continuous_run(self):
        ticks = 6
        np.random.seed(7)
        continuous = SimScene(_params())
        continuous.spawn_babies()
        Simulation(continuous, max_ticks=ticks).run()

        np.random.seed(7)
        resumed = SimScene(_params())
        resumed.spawn_babies()
        Simulation(resumed, max_ticks=3).run()
        restored = SimScene.from_dict(resumed.to_dict())
        Simulation(restored, max_ticks=3).run()

        assert restored.tick == continuous.tick == ticks
        assert np.array_equal(restored.world.material, continuous.world.material)
        assert np.array_equal(restored.world.energy, continuous.world.energy)
        assert np.array_equal(restored.world.temperature, continuous.world.temperature)
        assert np.array_equal(restored.world.signal, continuous.world.signal)

        assert len(restored.babies) == len(continuous.babies)
        for baby, orig in zip(restored.babies, continuous.babies):
            assert baby.entity.energy == orig.entity.energy
            assert np.array_equal(baby.entity.position, orig.entity.position)
            assert np.array_equal(baby.perceptron_cells.W, orig.perceptron_cells.W)
            assert np.array_equal(baby.perceptron_body.W, orig.perceptron_body.W)
            assert np.array_equal(baby.perceptron_entity.W, orig.perceptron_entity.W)
            assert baby.memory.to_dict() == orig.memory.to_dict()

    def test_multiple_save_load_cycles_are_stable(self):
        scene = _made_scene(seed=13, ticks=3)
        r = SimScene.from_dict(scene.to_dict())
        for _ in range(3):
            r = SimScene.from_dict(r.to_dict())
        assert r.tick == 3
        assert np.array_equal(r.world.energy, scene.world.energy)
