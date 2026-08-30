"""Tests for domains.shell.memory — Episode, EpisodicMemory, WorldEpisode, WorldMemory."""

import numpy as np
import pytest
from domains.shell.memory import Episode, EpisodicMemory, WorldEpisode, WorldMemory


# ── Episode ───────────────────────────────────────────────────────────

class TestEpisode:
    def test_fields(self):
        ep = Episode(features=np.array([1.0, 2.0]), action=(0.5,), reward=1.0, tick=10)
        assert ep.reward == 1.0
        assert ep.tick == 10
        assert ep.action == (0.5,)

    def test_features_is_array(self):
        ep = Episode(features=np.array([1.0, 2.0]), action=(0.5,), reward=0.5, tick=1)
        assert isinstance(ep.features, np.ndarray)

    def test_action_is_tuple(self):
        ep = Episode(features=np.array([1.0]), action=(0.3, 0.7), reward=1.0, tick=0)
        assert isinstance(ep.action, tuple)
        assert len(ep.action) == 2

    def test_reward_is_float(self):
        ep = Episode(features=np.array([1.0]), action=(0.5,), reward=2.5, tick=0)
        assert isinstance(ep.reward, float)

    def test_tick_is_int(self):
        ep = Episode(features=np.array([1.0]), action=(0.5,), reward=1.0, tick=42)
        assert isinstance(ep.tick, int)

    def test_multi_dim_features(self):
        ep = Episode(features=np.array([[1.0, 2.0], [3.0, 4.0]]), action=(0.5,), reward=1.0, tick=0)
        assert ep.features.shape == (2, 2)

    def test_zero_reward(self):
        ep = Episode(features=np.array([1.0]), action=(0.5,), reward=0.0, tick=0)
        assert ep.reward == 0.0

    def test_negative_reward(self):
        ep = Episode(features=np.array([1.0]), action=(0.5,), reward=-1.5, tick=0)
        assert ep.reward == -1.5


# ── EpisodicMemory ────────────────────────────────────────────────────

class TestEpisodicMemory:
    def test_init(self):
        em = EpisodicMemory(capacity=10)
        assert em.capacity == 10
        assert len(em) == 0

    def test_init_invalid_capacity(self):
        with pytest.raises(ValueError):
            EpisodicMemory(capacity=0)

    def test_record_and_len(self):
        em = EpisodicMemory(capacity=5)
        em.record(np.array([1.0]), action=(0.5,), reward=1.0, tick=1)
        em.record(np.array([2.0]), action=(0.6,), reward=0.8, tick=2)
        assert len(em) == 2

    def test_eviction(self):
        em = EpisodicMemory(capacity=3)
        for i in range(5):
            em.record(np.array([float(i)]), action=(0.5,), reward=float(i), tick=i)
        assert len(em) == 3

    def test_recall_recent(self):
        em = EpisodicMemory(capacity=5)
        for i in range(5):
            em.record(np.array([float(i)]), action=(0.5,), reward=float(i), tick=i)
        episodes = em.recall(k=3, by_reward=False)
        assert len(episodes) == 3
        assert episodes[-1].tick == 4

    def test_recall_by_reward(self):
        em = EpisodicMemory(capacity=5)
        for i in range(5):
            em.record(np.array([float(i)]), action=(0.5,), reward=float(i), tick=i)
        episodes = em.recall(k=3, by_reward=True)
        assert len(episodes) == 3
        rewards = [e.reward for e in episodes]
        assert rewards == sorted(rewards, reverse=True)

    def test_is_full(self):
        em = EpisodicMemory(capacity=2)
        assert not em.is_full
        em.record(np.array([1.0]), action=(0.5,), reward=1.0)
        em.record(np.array([2.0]), action=(0.6,), reward=0.8)
        assert em.is_full

    def test_mean_reward(self):
        em = EpisodicMemory(capacity=5)
        em.record(np.array([1.0]), action=(0.5,), reward=1.0)
        em.record(np.array([2.0]), action=(0.6,), reward=0.5)
        mean = em.mean_reward(k=2)
        assert abs(mean - 0.75) < 0.01

    def test_stats(self):
        em = EpisodicMemory(capacity=5)
        em.record(np.array([1.0]), action=(0.5,), reward=1.0)
        stats = em.stats()
        assert "size" in stats
        assert stats["size"] == 1

    def test_to_dict_and_from_dict(self):
        em = EpisodicMemory(capacity=5)
        em.record(np.array([1.0, 2.0]), action=(0.5,), reward=1.0, tick=1)
        d = em.to_dict()
        em2 = EpisodicMemory.from_dict(d)
        assert em2.capacity == em.capacity
        assert len(em2) == len(em)

    def test_negative_capacity(self):
        with pytest.raises(ValueError):
            EpisodicMemory(capacity=-1)

    def test_capacity_one(self):
        em = EpisodicMemory(capacity=1)
        em.record(np.array([1.0]), action=(0.5,), reward=1.0, tick=0)
        assert em.is_full
        em.record(np.array([2.0]), action=(0.5,), reward=2.0, tick=1)
        assert len(em) == 1
        assert em.recall(k=1)[0].reward == 2.0

    def test_recall_k_larger_than_buffer(self):
        em = EpisodicMemory(capacity=5)
        em.record(np.array([1.0]), action=(0.5,), reward=1.0, tick=0)
        episodes = em.recall(k=100)
        assert len(episodes) == 1

    def test_recall_k_zero_returns_all(self):
        em = EpisodicMemory(capacity=5)
        em.record(np.array([1.0]), action=(0.5,), reward=1.0, tick=0)
        episodes = em.recall(k=0, by_reward=False)
        assert len(episodes) == 1

    def test_recall_k_negative_returns_all(self):
        em = EpisodicMemory(capacity=5)
        em.record(np.array([1.0]), action=(0.5,), reward=1.0, tick=0)
        episodes = em.recall(k=-5, by_reward=False)
        assert len(episodes) == 1

    def test_mean_reward_empty(self):
        em = EpisodicMemory(capacity=5)
        assert em.mean_reward() == 0.0

    def test_mean_reward_single(self):
        em = EpisodicMemory(capacity=5)
        em.record(np.array([1.0]), action=(0.5,), reward=3.0, tick=0)
        assert em.mean_reward(k=1) == 3.0

    def test_stats_empty(self):
        em = EpisodicMemory(capacity=5)
        stats = em.stats()
        assert stats["size"] == 0
        assert stats["capacity"] == 5
        assert stats["mean_reward"] == 0.0

    def test_stats_ticks(self):
        em = EpisodicMemory(capacity=5)
        em.record(np.array([1.0]), action=(0.5,), reward=1.0, tick=10)
        em.record(np.array([2.0]), action=(0.5,), reward=1.0, tick=20)
        stats = em.stats()
        assert stats["oldest_tick"] == 10
        assert stats["newest_tick"] == 20

    def test_to_dict_structure(self):
        em = EpisodicMemory(capacity=3)
        em.record(np.array([1.0, 2.0]), action=(0.5, 0.6), reward=1.0, tick=5)
        d = em.to_dict()
        assert "capacity" in d
        assert "head" in d
        assert "episodes" in d
        assert len(d["episodes"]) == 1

    def test_to_dict_episode_fields(self):
        em = EpisodicMemory(capacity=3)
        em.record(np.array([1.0, 2.0]), action=(0.5,), reward=1.0, tick=5)
        d = em.to_dict()
        ep = d["episodes"][0]
        assert "features" in ep
        assert "action" in ep
        assert "reward" in ep
        assert "tick" in ep

    def test_from_dict_roundtrip_preserves_features(self):
        em = EpisodicMemory(capacity=5)
        em.record(np.array([1.0, 2.0, 3.0]), action=(0.5,), reward=1.0, tick=0)
        d = em.to_dict()
        em2 = EpisodicMemory.from_dict(d)
        np.testing.assert_array_equal(
            em2.recall(k=1)[0].features,
            em.recall(k=1)[0].features
        )

    def test_from_dict_preserves_head(self):
        em = EpisodicMemory(capacity=2)
        em.record(np.array([1.0]), action=(0.5,), reward=1.0, tick=0)
        em.record(np.array([2.0]), action=(0.5,), reward=2.0, tick=1)
        em.record(np.array([3.0]), action=(0.5,), reward=3.0, tick=2)
        d = em.to_dict()
        em2 = EpisodicMemory.from_dict(d)
        assert em2._head == em._head

    def test_eviction_overwrites_oldest(self):
        em = EpisodicMemory(capacity=3)
        em.record(np.array([1.0]), action=(0.5,), reward=1.0, tick=0)
        em.record(np.array([2.0]), action=(0.5,), reward=2.0, tick=1)
        em.record(np.array([3.0]), action=(0.5,), reward=3.0, tick=2)
        em.record(np.array([4.0]), action=(0.5,), reward=4.0, tick=3)
        rewards = [e.reward for e in em.recall(k=3)]
        assert 1.0 not in rewards
        assert 4.0 in rewards

    def test_record_default_tick(self):
        em = EpisodicMemory(capacity=5)
        em.record(np.array([1.0]), action=(0.5,), reward=1.0)
        assert em.recall(k=1)[0].tick == 0

    def test_recall_by_reward_empty(self):
        em = EpisodicMemory(capacity=5)
        episodes = em.recall(k=5, by_reward=True)
        assert episodes == []

    def test_recall_recent_empty(self):
        em = EpisodicMemory(capacity=5)
        episodes = em.recall(k=5, by_reward=False)
        assert episodes == []

    def test_recall_by_reward_all_same_reward(self):
        em = EpisodicMemory(capacity=5)
        for i in range(5):
            em.record(np.array([float(i)]), action=(0.5,), reward=1.0, tick=i)
        episodes = em.recall(k=3, by_reward=True)
        assert len(episodes) == 3

    def test_features_are_float32(self):
        em = EpisodicMemory(capacity=5)
        em.record(np.array([1, 2, 3]), action=(0.5,), reward=1.0, tick=0)
        assert em.recall(k=1)[0].features.dtype == np.float32

    def test_action_converted_to_float_tuple(self):
        em = EpisodicMemory(capacity=5)
        em.record(np.array([1.0]), action=(1, 2, 3), reward=1.0, tick=0)
        ep = em.recall(k=1)[0]
        assert ep.action == (1.0, 2.0, 3.0)


# ── WorldEpisode ──────────────────────────────────────────────────────

class TestWorldEpisode:
    def test_fields(self):
        we = WorldEpisode(
            features=np.array([1.0, 2.0]),
            action=(0.5,),
            reward=1.0,
            tick=10,
            group_id=1,
            donor_id=2,
        )
        assert we.reward == 1.0
        assert we.tick == 10
        assert we.group_id == 1
        assert we.donor_id == 2

    def test_default_ids(self):
        we = WorldEpisode(features=np.array([1.0]), action=(0.5,), reward=1.0, tick=0)
        assert we.group_id == 0
        assert we.donor_id == 0


# ── WorldMemory ───────────────────────────────────────────────────────

class TestWorldMemory:
    def test_init_empty(self):
        wm = WorldMemory()
        assert len(wm) == 0

    def test_record(self):
        wm = WorldMemory()
        wm.record(np.array([1.0]), action=(0.5,), reward=1.0, tick=0)
        assert len(wm) == 1

    def test_record_stamps_group_donor(self):
        wm = WorldMemory()
        wm.record(np.array([1.0]), action=(0.5,), reward=1.0, tick=0, group_id=3, donor_id=7)
        episodes = wm.recall(k=1, by_reward=False)
        assert episodes[0].group_id == 3
        assert episodes[0].donor_id == 7

    def test_consolidate(self):
        em = EpisodicMemory(capacity=5)
        for i in range(5):
            em.record(np.array([float(i)]), action=(0.5,), reward=float(i), tick=i)
        wm = WorldMemory()
        deposited = wm.consolidate(em, k=3, group_id=1, donor_id=2)
        assert deposited == 3
        assert len(wm) == 3

    def test_consolidate_k_zero(self):
        em = EpisodicMemory(capacity=5)
        em.record(np.array([1.0]), action=(0.5,), reward=1.0)
        wm = WorldMemory()
        deposited = wm.consolidate(em, k=0)
        assert deposited == 0
        assert len(wm) == 0

    def test_consolidate_deposits_best(self):
        em = EpisodicMemory(capacity=5)
        for i in range(5):
            em.record(np.array([float(i)]), action=(0.5,), reward=float(i), tick=i)
        wm = WorldMemory()
        wm.consolidate(em, k=2)
        rewards = [e.reward for e in wm.recall(k=2)]
        assert sorted(rewards, reverse=True) == rewards

    def test_recall_by_reward(self):
        wm = WorldMemory()
        wm.record(np.array([1.0]), action=(0.5,), reward=3.0, tick=0)
        wm.record(np.array([2.0]), action=(0.5,), reward=1.0, tick=1)
        wm.record(np.array([3.0]), action=(0.5,), reward=2.0, tick=2)
        episodes = wm.recall(k=2, by_reward=True)
        assert episodes[0].reward >= episodes[1].reward

    def test_recall_recent(self):
        wm = WorldMemory()
        wm.record(np.array([1.0]), action=(0.5,), reward=1.0, tick=0)
        wm.record(np.array([2.0]), action=(0.5,), reward=2.0, tick=1)
        wm.record(np.array([3.0]), action=(0.5,), reward=3.0, tick=2)
        episodes = wm.recall(k=2, by_reward=False)
        assert len(episodes) == 2
        assert episodes[-1].tick == 2

    def test_recall_filter_group_id(self):
        wm = WorldMemory()
        wm.record(np.array([1.0]), action=(0.5,), reward=1.0, tick=0, group_id=1)
        wm.record(np.array([2.0]), action=(0.5,), reward=2.0, tick=1, group_id=2)
        wm.record(np.array([3.0]), action=(0.5,), reward=3.0, tick=2, group_id=1)
        episodes = wm.recall(k=10, by_reward=True, group_id=1)
        assert len(episodes) == 2

    def test_recall_empty(self):
        wm = WorldMemory()
        episodes = wm.recall(k=5)
        assert episodes == []

    def test_mean_reward(self):
        wm = WorldMemory()
        wm.record(np.array([1.0]), action=(0.5,), reward=1.0, tick=0)
        wm.record(np.array([2.0]), action=(0.5,), reward=3.0, tick=1)
        mean = wm.mean_reward(k=2)
        assert abs(mean - 2.0) < 0.01

    def test_mean_reward_empty(self):
        wm = WorldMemory()
        assert wm.mean_reward() == 0.0

    def test_stats(self):
        wm = WorldMemory()
        wm.record(np.array([1.0]), action=(0.5,), reward=1.0, tick=10, group_id=1)
        wm.record(np.array([2.0]), action=(0.5,), reward=2.0, tick=20, group_id=1)
        stats = wm.stats()
        assert stats["size"] == 2
        assert stats["oldest_tick"] == 10
        assert stats["newest_tick"] == 20
        assert stats["groups"] == {1: 2}

    def test_stats_empty(self):
        wm = WorldMemory()
        stats = wm.stats()
        assert stats["size"] == 0
        assert stats["mean_reward"] == 0.0

    def test_to_dict(self):
        wm = WorldMemory()
        wm.record(np.array([1.0, 2.0]), action=(0.5,), reward=1.0, tick=5, group_id=1, donor_id=2)
        d = wm.to_dict()
        assert "episodes" in d
        assert len(d["episodes"]) == 1

    def test_from_dict(self):
        wm = WorldMemory()
        wm.record(np.array([1.0]), action=(0.5,), reward=1.0, tick=0, group_id=1, donor_id=2)
        d = wm.to_dict()
        wm2 = WorldMemory.from_dict(d)
        assert len(wm2) == 1
        assert wm2.recall(k=1)[0].group_id == 1

    def test_from_dict_empty(self):
        wm = WorldMemory.from_dict({"episodes": []})
        assert len(wm) == 0

    def test_never_evicts(self):
        wm = WorldMemory()
        for i in range(100):
            wm.record(np.array([float(i)]), action=(0.5,), reward=float(i), tick=i)
        assert len(wm) == 100

    def test_consolidate_returns_min_k_len(self):
        em = EpisodicMemory(capacity=2)
        em.record(np.array([1.0]), action=(0.5,), reward=1.0, tick=0)
        wm = WorldMemory()
        deposited = wm.consolidate(em, k=10)
        assert deposited == 1

    def test_recall_negative_k(self):
        wm = WorldMemory()
        wm.record(np.array([1.0]), action=(0.5,), reward=1.0, tick=0)
        episodes = wm.recall(k=-5)
        assert episodes == []

    def test_features_are_float32(self):
        wm = WorldMemory()
        wm.record(np.array([1, 2, 3]), action=(0.5,), reward=1.0, tick=0)
        assert wm.recall(k=1)[0].features.dtype == np.float32

    def test_action_converted_to_float_tuple(self):
        wm = WorldMemory()
        wm.record(np.array([1.0]), action=(1, 2, 3), reward=1.0, tick=0)
        ep = wm.recall(k=1, by_reward=False)[0]
        assert ep.action == (1.0, 2.0, 3.0)

    def test_to_dict_episode_fields(self):
        wm = WorldMemory()
        wm.record(np.array([1.0]), action=(0.5,), reward=1.0, tick=0, group_id=1, donor_id=2)
        d = wm.to_dict()
        ep = d["episodes"][0]
        assert "features" in ep
        assert "action" in ep
        assert "reward" in ep
        assert "tick" in ep
        assert "group_id" in ep
        assert "donor_id" in ep

    def test_init_with_episodes(self):
        we = WorldEpisode(features=np.array([1.0]), action=(0.5,), reward=1.0, tick=0)
        wm = WorldMemory(episodes=[we])
        assert len(wm) == 1

    def test_recall_default_by_reward(self):
        wm = WorldMemory()
        wm.record(np.array([1.0]), action=(0.5,), reward=1.0, tick=0)
        wm.record(np.array([2.0]), action=(0.5,), reward=5.0, tick=1)
        episodes = wm.recall(k=2)
        assert episodes[0].reward == 5.0
