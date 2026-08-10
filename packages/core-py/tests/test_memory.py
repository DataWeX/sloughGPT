"""
Tests for the episodic memory ring buffer and its wiring into SimBaby.
"""

from __future__ import annotations

import numpy as np
import pytest

from domains.shell.memory import EpisodicMemory, Episode, WorldEpisode, WorldMemory
from domains.shell.simulation import (
    SimBaby,
    SimScene,
    Simulation,
    WorldGrid,
    WorldParams,
)


def _feat(value: float = 0.5) -> np.ndarray:
    return np.array([value, value, value, value], dtype=np.float32)


class TestEpisodicMemoryBuffer:
    def test_empty_buffer(self):
        m = EpisodicMemory(capacity=8)
        assert len(m) == 0
        assert not m.is_full
        assert m.recall() == []
        assert m.mean_reward() == 0.0

    def test_record_appends(self):
        m = EpisodicMemory(capacity=8)
        m.record(_feat(), (1.0, 0.0, 0.5), 5.0, tick=3)
        assert len(m) == 1
        ep = m.recall()[0]
        assert ep.reward == 5.0
        assert ep.tick == 3
        assert tuple(ep.action) == (1.0, 0.0, 0.5)

    def test_recall_most_recent(self):
        m = EpisodicMemory(capacity=8)
        for i in range(5):
            m.record(_feat(i), (1.0,), float(i), tick=i)
        recent = m.recall(k=2)
        assert [e.reward for e in recent] == [3.0, 4.0]
        assert [e.tick for e in recent] == [3, 4]

    def test_recall_by_reward(self):
        m = EpisodicMemory(capacity=8)
        for reward in (1.0, 5.0, 3.0, -2.0, 4.0):
            m.record(_feat(), (1.0,), reward, tick=0)
        top = m.recall(k=3, by_reward=True)
        assert [e.reward for e in top] == [5.0, 4.0, 3.0]

    def test_ring_buffer_evicts_oldest(self):
        m = EpisodicMemory(capacity=3)
        for i in range(5):
            m.record(_feat(), (1.0,), float(i), tick=i)
        assert len(m) == 3
        assert m.is_full
        recent = m.recall(k=3)
        assert [e.tick for e in recent] == [2, 3, 4]

    def test_chronological_order_after_wrap(self):
        m = EpisodicMemory(capacity=3)
        for i in range(6):
            m.record(_feat(), (1.0,), float(i), tick=i)
        assert [e.tick for e in m.recall(k=3)] == [3, 4, 5]

    def test_mean_reward_over_recent(self):
        m = EpisodicMemory(capacity=8)
        for reward in (1.0, 2.0, 3.0, 4.0):
            m.record(_feat(), (1.0,), reward, tick=0)
        assert m.mean_reward(k=2) == pytest.approx(3.5)
        assert m.mean_reward() == pytest.approx(2.5)  # default k=5 capped to 4 episodes

    def test_capacity_zero_raises(self):
        with pytest.raises(ValueError):
            EpisodicMemory(capacity=0)

    def test_features_copied_not_aliased(self):
        m = EpisodicMemory(capacity=4)
        f = _feat(0.25)
        m.record(f, (1.0,), 1.0, tick=0)
        f[:] = 0.9
        assert float(m.recall()[0].features[0]) == pytest.approx(0.25)

    def test_stats(self):
        m = EpisodicMemory(capacity=8)
        m.record(_feat(), (1.0,), 2.0, tick=10)
        m.record(_feat(), (1.0,), 4.0, tick=12)
        s = m.stats()
        assert s["capacity"] == 8
        assert s["size"] == 2
        assert s["oldest_tick"] == 10
        assert s["newest_tick"] == 12
        assert s["mean_reward"] == pytest.approx(3.0)


class TestSimBabyMemory:
    def test_baby_has_episodic_memory(self):
        baby = SimBaby(initial_energy=100.0)
        assert baby.memory.capacity == WorldParams().memory_capacity
        assert len(baby.memory) == 0

    def test_learn_records_episode(self):
        baby = SimBaby(initial_energy=100.0)
        baby.perceive(WorldGrid((16, 8, 16)))
        baby.react(baby._last_perception, 0.0)
        baby.learn(5.0)
        assert len(baby.memory) == 1
        ep = baby.memory.recall()[0]
        assert ep.reward == pytest.approx(5.0)
        assert len(ep.features) == baby.params.cells_input_dim
        assert len(ep.action) == 3

    def test_recall_memories_returns_dicts(self):
        baby = SimBaby(initial_energy=100.0)
        baby.perceive(WorldGrid((16, 8, 16)))
        baby.react(baby._last_perception, 0.0)
        baby.learn(2.0)
        mems = baby.recall_memories()
        assert len(mems) == 1
        assert mems[0]["reward"] == pytest.approx(2.0)
        assert len(mems[0]["features"]) == baby.params.cells_input_dim

    def test_starving_baby_records_nothing(self):
        baby = SimBaby(initial_energy=5.0)  # starving
        baby.perceive(WorldGrid((16, 8, 16)))
        baby.learn(1.0)
        assert len(baby.memory) == 0

    def test_memory_grows_and_keeps_recent(self):
        baby = SimBaby(initial_energy=100.0)
        for i in range(3):
            baby.perceive(WorldGrid((16, 8, 16)))
            baby.react(baby._last_perception, 0.0)
            baby.learn(float(i + 1))
        assert len(baby.memory) == 3
        rewards = [e.reward for e in baby.memory.recall(k=3)]
        assert rewards == [1.0, 2.0, 3.0]
        assert baby.memory.mean_reward() == pytest.approx(2.0)

    def test_info_exposes_memory_stats(self):
        baby = SimBaby(initial_energy=100.0)
        baby.perceive(WorldGrid((16, 8, 16)))
        baby.react(baby._last_perception, 0.0)
        baby.learn(3.0)
        info = baby.info()
        assert info["memory"]["size"] == 1
        assert info["memory"]["mean_reward"] == pytest.approx(3.0)

    def test_learning_block_none_before_first_learn(self):
        baby = SimBaby(initial_energy=100.0)
        assert baby.info()["learning"] is None

    def test_first_episode_has_zero_baseline_max_surprise(self):
        baby = SimBaby(initial_energy=100.0)
        baby.learn(5.0)
        lr = baby._last_learning
        assert lr["baseline"] == pytest.approx(0.0)
        assert lr["surprise"] == pytest.approx(5.0)
        assert lr["scale"] == pytest.approx(1.5)

    def test_expected_outcome_learns_at_floor_scale(self):
        baby = SimBaby(initial_energy=100.0)
        for _ in range(5):
            baby.memory.record(_feat(), (0.0, 0.0, 0.0), 5.0)  # baseline 5.0
        baby.learn(5.0)
        lr = baby._last_learning
        assert lr["baseline"] == pytest.approx(5.0)
        assert lr["surprise"] == pytest.approx(0.0)
        assert lr["scale"] == pytest.approx(0.5)

    def test_surprise_scales_weight_updates(self):
        def fresh():
            b = SimBaby(initial_energy=100.0,
                        position=np.array([8.0, 4.0, 8.0]))
            b.perceptron_body.W[:] = 0.05  # identical deterministic weights
            return b

        surprising = fresh()   # empty memory → baseline 0, surprise 5
        expected = fresh()     # established +5 baseline → surprise 0
        for _ in range(5):
            expected.memory.record(_feat(), (0.0, 0.0, 0.0), 5.0)

        w_surprised = surprising.perceptron_body.W.copy()
        w_expected = expected.perceptron_body.W.copy()
        surprising.learn(5.0)
        expected.learn(5.0)

        d_surprised = np.abs(surprising.perceptron_body.W - w_surprised).sum()
        d_expected = np.abs(expected.perceptron_body.W - w_expected).sum()
        assert d_surprised > d_expected
        # identical gradient, lr ratio 1.5 / 0.5 = 3x
        assert d_surprised == pytest.approx(3.0 * d_expected, rel=0.05)

    def test_info_exposes_learning_block(self):
        baby = SimBaby(initial_energy=100.0)
        baby.learn(4.0)
        learning = baby.info()["learning"]
        assert learning["baseline"] == pytest.approx(0.0)
        assert learning["surprise"] == pytest.approx(4.0)
        assert learning["scale"] == pytest.approx(1.5)

    def test_simulation_fills_memory_over_ticks(self):
        from domains.shell.simulation import SimScene, Simulation

        params = WorldParams(grid_size=(8, 4, 8), start_agents=1, memory_capacity=16)
        scene = SimScene(params=params)
        scene.spawn_babies(count=1)
        sim = Simulation(scene, max_ticks=10)
        sim.run()
        assert len(scene.babies) == 1
        assert 0 < len(scene.babies[0].memory) <= 16


class TestWorldMemoryReservoir:
    """The world-level long-term reservoir — append-only, never evicts."""

    def test_empty_reservoir(self):
        m = WorldMemory()
        assert len(m) == 0
        assert m.recall() == []
        assert m.mean_reward() == 0.0
        stats = m.stats()
        assert stats["size"] == 0
        assert stats["mean_reward"] == 0.0
        assert stats["groups"] == {}

    def test_record_appends_and_never_evicts(self):
        m = WorldMemory()
        for i in range(300):
            m.record(_feat(float(i)), (0.0, 0.0, 0.0), float(i), tick=i)
        assert len(m) == 300  # no capacity ceiling — deposits are retained

    def test_record_stamps_group_and_donor(self):
        m = WorldMemory()
        m.record(_feat(), (0.0, 0.0, 0.0), 1.0, tick=3, group_id=2, donor_id=7)
        ep = m.recall()[0]
        assert ep.group_id == 2
        assert ep.donor_id == 7
        assert ep.tick == 3

    def test_consolidate_deposits_top_reward_episodes(self):
        donor = EpisodicMemory(capacity=8)
        for i, reward in enumerate((1.0, 5.0, 3.0, 2.0, 4.0)):
            donor.record(np.full(4, float(i), np.float32),
                         (0.5, 0.5, 0.5), reward, tick=i)
        m = WorldMemory()
        deposited = m.consolidate(donor, k=3, group_id=1, donor_id=9)
        assert deposited == 3
        assert len(m) == 3
        assert [e.reward for e in m.recall(k=3)] == [5.0, 4.0, 3.0]
        assert all(e.group_id == 1 and e.donor_id == 9 for e in m._episodes)

    def test_consolidate_zero_deposits_nothing(self):
        donor = EpisodicMemory(capacity=4)
        donor.record(_feat(), (0.0, 0.0, 0.0), 2.0)
        m = WorldMemory()
        assert m.consolidate(donor, k=0) == 0
        assert len(m) == 0

    def test_consolidate_empty_donor_deposits_nothing(self):
        m = WorldMemory()
        assert m.consolidate(EpisodicMemory(capacity=4), k=5) == 0
        assert len(m) == 0

    def test_recall_ranks_by_reward(self):
        m = WorldMemory()
        for reward in (1.0, 5.0, 3.0, 4.0, 2.0):
            m.record(_feat(reward), (0.0, 0.0, 0.0), reward)
        assert [e.reward for e in m.recall(k=2)] == [5.0, 4.0]
        assert m.mean_reward(k=1) == pytest.approx(5.0)  # best single episode

    def test_recall_recent_returns_last_deposits(self):
        m = WorldMemory()
        for reward in (1.0, 2.0, 3.0):
            m.record(_feat(), (0.0, 0.0, 0.0), reward, tick=int(reward))
        assert [e.tick for e in m.recall(k=2, by_reward=False)] == [2, 3]

    def test_recall_filters_by_group(self):
        m = WorldMemory()
        m.record(_feat(), (0.0, 0.0, 0.0), 1.0, group_id=0)
        m.record(_feat(), (0.0, 0.0, 0.0), 9.0, group_id=1)
        m.record(_feat(), (0.0, 0.0, 0.0), 5.0, group_id=1)
        only = m.recall(k=10, group_id=1)
        assert [e.reward for e in only] == [9.0, 5.0]
        assert all(e.group_id == 1 for e in only)

    def test_stats_reports_groups(self):
        m = WorldMemory()
        m.record(_feat(), (0.0, 0.0, 0.0), 2.0, group_id=0)
        m.record(_feat(), (0.0, 0.0, 0.0), 4.0, group_id=1)
        m.record(_feat(), (0.0, 0.0, 0.0), 6.0, group_id=1)
        stats = m.stats()
        assert stats["groups"] == {0: 1, 1: 2}
        assert stats["size"] == 3
        assert stats["mean_reward"] == pytest.approx(4.0)  # top-5 = all episodes

    def test_serialization_roundtrip_is_lossless(self):
        m = WorldMemory()
        for i, reward in enumerate((1.0, 5.0, 3.0)):
            m.record(np.full(4, float(i), np.float32),
                     (0.5, 0.5, 0.5), reward, tick=i,
                     group_id=i % 2, donor_id=100 + i)
        restored = WorldMemory.from_dict(m.to_dict())
        assert len(restored) == 3
        for a, b in zip(m._episodes, restored._episodes):
            assert np.array_equal(a.features, b.features)
            assert a.action == b.action
            assert a.reward == b.reward
            assert a.tick == b.tick
            assert a.group_id == b.group_id
            assert a.donor_id == b.donor_id

    def test_from_dict_empty(self):
        assert len(WorldMemory.from_dict({})) == 0


class TestWorldMemoryScene:
    """SimScene wiring: off by default, reservoir on demand, seed + deposit."""

    def test_scene_reservoir_off_by_default(self):
        from domains.shell.simulation import SimScene

        scene = SimScene(params=WorldParams(grid_size=(8, 4, 8)))
        assert scene.world_memory is None
        assert scene.deposit_memory(SimBaby(initial_energy=100.0)) == 0

    def test_scene_creates_reservoir_when_enabled(self):
        from domains.shell.simulation import SimScene

        params = WorldParams(grid_size=(8, 4, 8), memory_enabled=True)
        scene = SimScene(params=params)
        assert scene.world_memory is not None
        assert len(scene.world_memory) == 0
        assert scene.memory_seeds_given == 0

    def test_deposit_memory_consolidates_baby_best_episodes(self):
        from domains.shell.simulation import SimScene

        params = WorldParams(grid_size=(8, 4, 8), memory_enabled=True,
                             memory_deposit=3)
        scene = SimScene(params=params)
        baby = SimBaby(initial_energy=100.0, params=params)
        for reward in (1.0, 5.0, 3.0, 2.0, 4.0):
            baby.memory.record(_feat(reward), (0.0, 0.0, 0.0), reward, tick=1)
        deposited = scene.deposit_memory(baby)
        assert deposited == 3
        assert len(scene.world_memory) == 3
        assert [e.reward for e in scene.world_memory.recall(k=3)] == \
               [5.0, 4.0, 3.0]

    def test_add_baby_seeds_newborn_from_reservoir(self):
        from domains.shell.simulation import SimScene

        params = WorldParams(grid_size=(8, 4, 8), memory_enabled=True,
                             memory_seed=2)
        scene = SimScene(params=params)
        scene.world_memory.record(_feat(), (0.0, 0.0, 0.0), 1.0, group_id=0)
        scene.world_memory.record(_feat(), (0.0, 0.0, 0.0), 9.0, group_id=0)
        scene.world_memory.record(_feat(), (0.0, 0.0, 0.0), 5.0, group_id=0)
        baby = SimBaby(initial_energy=100.0, params=params)
        scene.add_baby(baby)
        assert len(baby.memory) == 2  # memory_seed best episodes
        assert [e.reward for e in baby.memory.recall(k=2)] == [9.0, 5.0]
        assert scene.memory_seeds_given == 2

    def test_add_baby_seeds_nothing_without_reservoir(self):
        from domains.shell.simulation import SimScene

        params = WorldParams(grid_size=(8, 4, 8), memory_seed=4)
        scene = SimScene(params=params)  # memory_enabled False → no reservoir
        baby = SimBaby(initial_energy=100.0, params=params)
        scene.add_baby(baby)
        assert len(baby.memory) == 0
        assert scene.memory_seeds_given == 0

    def test_scene_serialization_preserves_reservoir(self):
        from domains.shell.simulation import SimScene

        params = WorldParams(grid_size=(8, 4, 8), memory_enabled=True)
        scene = SimScene(params=params)
        scene.world_memory.record(_feat(), (0.0, 0.0, 0.0), 7.0, tick=2,
                                  group_id=1, donor_id=5)
        data = scene.to_dict()
        restored = SimScene.from_dict(data)
        assert restored.world_memory is not None
        assert len(restored.world_memory) == 1
        ep = restored.world_memory.recall()[0]
        assert ep.reward == pytest.approx(7.0)
        assert ep.tick == 2
        assert ep.group_id == 1
        assert ep.donor_id == 5

    def test_dead_baby_deposits_before_removal(self):
        from domains.shell.simulation import SimScene

        params = WorldParams(grid_size=(8, 4, 8), memory_enabled=True,
                             memory_deposit=8)
        scene = SimScene(params=params)
        baby = SimBaby(initial_energy=1.0, params=params)
        for reward in (1.0, 5.0, 3.0, 2.0, 4.0):
            baby.memory.record(_feat(reward), (0.0, 0.0, 0.0), reward, tick=1)
        scene.add_baby(baby)
        baby.entity.energy = 0.0
        baby.entity.alive = False
        sim = Simulation(scene, max_ticks=10)
        sim.step()
        assert len(scene.entities) == 0  # corpse swept from the world
        assert len(scene.world_memory) == 5  # episodes survived the death
        assert [e.reward for e in scene.world_memory.recall(k=3)] == \
               [5.0, 4.0, 3.0]
        ep = scene.world_memory.recall()[0]
        assert ep.donor_id == baby.entity.id

    def test_dead_deposit_respects_deposit_cap(self):
        from domains.shell.simulation import SimScene

        params = WorldParams(grid_size=(8, 4, 8), memory_enabled=True,
                             memory_deposit=2)
        scene = SimScene(params=params)
        baby = SimBaby(initial_energy=1.0, params=params)
        for reward in (1.0, 5.0, 3.0, 2.0, 4.0):
            baby.memory.record(_feat(reward), (0.0, 0.0, 0.0), reward, tick=1)
        scene.add_baby(baby)
        baby.entity.energy = 0.0
        baby.entity.alive = False
        sim = Simulation(scene, max_ticks=10)
        sim.step()
        assert len(scene.world_memory) == 2  # only the best episodes kept
        assert [e.reward for e in scene.world_memory.recall(k=3)] == \
               [5.0, 4.0]
