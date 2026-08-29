"""Tests for domains.shell.memory — Episode, EpisodicMemory."""

import numpy as np
from domains.shell.memory import Episode, EpisodicMemory


class TestEpisode:
    def test_fields(self):
        ep = Episode(features=np.array([1.0, 2.0]), action=(0.5,), reward=1.0, tick=10)
        assert ep.reward == 1.0
        assert ep.tick == 10
        assert ep.action == (0.5,)


class TestEpisodicMemory:
    def test_init(self):
        em = EpisodicMemory(capacity=10)
        assert em.capacity == 10
        assert len(em) == 0

    def test_init_invalid_capacity(self):
        try:
            EpisodicMemory(capacity=0)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

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
