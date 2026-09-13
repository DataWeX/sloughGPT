"""
Memory for world-realm agents.

EpisodicMemory is a per-baby ring buffer of lived experiences. Each episode
records the perception features that preceded an action, the action itself
(the gates the cells perceptron emitted), and the reward that followed (the
energy delta). Only the most recent `capacity` episodes are kept; the oldest
are evicted. Retrieval returns the most recent episodes or the highest-reward
ones, which the baby's learning loop uses as a baseline for "how well have I
been doing lately" (surprise-based learning).

WorldMemory is the world-level long-term reservoir. It is append-only and
never evicts: episodes are deposited by babies (at death and at generation
boundaries) and survive across generations and lineages, seeding newborns
so lived experience outlives any single agent — the counterpart to the
inherited memotype at the collective level.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Episode:
    """A single lived experience: features -> action -> reward."""

    features: np.ndarray
    action: tuple[float, ...]
    reward: float
    tick: int


class EpisodicMemory:
    """
    Fixed-capacity ring buffer of episodes.

    ``record()`` appends an episode; once full the oldest is overwritten.
    ``recall()`` returns the k most recent episodes, or the k highest-reward
    ones. Chronological order is preserved on retrieval.

    Args:
        capacity: maximum number of episodes retained.

    Raises:
        ValueError: if capacity is less than 1.
    """

    def __init__(self, capacity: int = 64):
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self.capacity = int(capacity)
        self._episodes: list[Episode] = []
        self._head = 0

    def record(
        self,
        features: np.ndarray,
        action: tuple[float, ...],
        reward: float,
        tick: int = 0,
    ) -> None:
        """
        Record one episode, evicting the oldest when the buffer is full.

        Args:
            features: perception feature vector seen before acting.
            action: action signature (cells-perceptron gates).
            reward: energy delta that followed the action.
            tick: world tick when the episode happened.
        """
        episode = Episode(
            features=np.asarray(features, dtype=np.float32).copy(),
            action=tuple(float(a) for a in action),
            reward=float(reward),
            tick=int(tick),
        )
        if len(self._episodes) < self.capacity:
            self._episodes.append(episode)
        else:
            self._episodes[self._head] = episode
            self._head = (self._head + 1) % self.capacity

    def _chronological(self) -> list[Episode]:
        """Episodes in insertion order regardless of buffer wrap state."""
        if len(self._episodes) < self.capacity or self._head == 0:
            return list(self._episodes)
        return self._episodes[self._head:] + self._episodes[:self._head]

    def recall(self, k: int = 5, by_reward: bool = False) -> list[Episode]:
        """
        Retrieve k episodes.

        Args:
            k: number of episodes to return (capped by buffer size).
            by_reward: when True, return the k highest-reward episodes;
                otherwise return the k most recent.

        Returns:
            List of episodes in chronological order.
        """
        episodes = self._chronological()
        if k <= 0:
            return []
        if not by_reward:
            return episodes[-k:]
        ranked = sorted(episodes, key=lambda e: e.reward, reverse=True)
        return ranked[:k]

    def mean_reward(self, k: int = 5) -> float:
        """
        Average reward over the k most recent episodes.

        Args:
            k: how many recent episodes to average.

        Returns:
            Mean reward (0.0 when the buffer is empty).
        """
        recent = self.recall(k)
        if not recent:
            return 0.0
        return float(np.mean([e.reward for e in recent]))

    def __len__(self) -> int:
        """Number of episodes currently held."""
        return len(self._episodes)

    @property
    def is_full(self) -> bool:
        """True once the buffer has wrapped at least once."""
        return len(self._episodes) == self.capacity

    def stats(self) -> dict[str, Any]:
        """
        Observable summary of the buffer.

        Returns:
            Dict with capacity, size, oldest/newest tick, and mean reward.
        """
        episodes = self._chronological()
        oldest_tick = episodes[0].tick if episodes else 0
        newest_tick = episodes[-1].tick if episodes else 0
        return {
            "capacity": self.capacity,
            "size": len(episodes),
            "oldest_tick": oldest_tick,
            "newest_tick": newest_tick,
            "mean_reward": self.mean_reward(),
        }

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the buffer to a JSON-safe dict.

        The ring-buffer head position is stored so restore is exact — eviction
        resumes from the same slot it would have used in the live process.

        Returns:
            Dict with capacity, head, and the episodes (in storage order).
        """
        return {
            "capacity": self.capacity,
            "head": self._head,
            "episodes": [
                {
                    "features": e.features.tolist(),
                    "action": list(e.action),
                    "reward": e.reward,
                    "tick": e.tick,
                }
                for e in self._episodes
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EpisodicMemory":
        """
        Rebuild a buffer from :meth:`to_dict` output.

        Args:
            data: serialized memory dict.

        Returns:
            A new EpisodicMemory with the exact stored episodes and head.
        """
        memory = cls(int(data["capacity"]))
        memory._episodes = [
            Episode(
                features=np.asarray(ep["features"], dtype=np.float32).copy(),
                action=tuple(float(a) for a in ep["action"]),
                reward=float(ep["reward"]),
                tick=int(ep["tick"]),
            )
            for ep in data["episodes"]
        ]
        memory._head = int(data["head"])
        return memory


@dataclass
class WorldEpisode:
    """A world-level episode: experience a baby deposited into the reservoir."""

    features: np.ndarray
    action: tuple[float, ...]
    reward: float
    tick: int
    group_id: int = 0
    donor_id: int = 0


class WorldMemory:
    """
    World-level long-term memory — an append-only reservoir that never evicts.

    Where EpisodicMemory is a fixed-capacity ring buffer that dies with its
    baby, WorldMemory is the world's collective record: babies deposit their
    best episodes (at death, and at generation boundaries via the evolution
    engine) and newborns are seeded from the reservoir's best episodes. The
    reservoir holds every deposit — the growth ceiling is the deposit cadence
    itself, not a fixed capacity — so experience survives death and crosses
    lineages rather than moving only parent->child through the memotype.

    Args:
        episodes: optional initial contents (used by :meth:`from_dict`).
    """

    def __init__(self, episodes: list[WorldEpisode] | None = None):
        self._episodes: list[WorldEpisode] = list(episodes or [])

    def record(
        self,
        features: np.ndarray,
        action: tuple[float, ...],
        reward: float,
        tick: int = 0,
        group_id: int = 0,
        donor_id: int = 0,
    ) -> None:
        """
        Append one episode to the reservoir (never evicted).

        Args:
            features: perception feature vector seen before acting.
            action: action signature (cells-perceptron gates).
            reward: energy delta that followed the action.
            tick: world tick when the episode happened.
            group_id: donor's tribe id.
            donor_id: donor baby's entity id.
        """
        self._episodes.append(WorldEpisode(
            features=np.asarray(features, dtype=np.float32).copy(),
            action=tuple(float(a) for a in action),
            reward=float(reward),
            tick=int(tick),
            group_id=int(group_id),
            donor_id=int(donor_id),
        ))

    def consolidate(
        self,
        memory: EpisodicMemory,
        k: int,
        group_id: int = 0,
        donor_id: int = 0,
    ) -> int:
        """
        Deposit up to k highest-reward episodes from an episodic memory.

        Args:
            memory: the donor baby's episodic ring buffer.
            k: max number of episodes to deposit (0 deposits nothing).
            group_id: donor's tribe id (stamped on each deposit).
            donor_id: donor baby's entity id.

        Returns:
            Number of episodes actually deposited.
        """
        if k <= 0:
            return 0
        for e in memory.recall(k, by_reward=True):
            self.record(e.features, e.action, e.reward, e.tick,
                        group_id=group_id, donor_id=donor_id)
        return min(k, len(memory))

    def recall(
        self,
        k: int = 5,
        by_reward: bool = True,
        group_id: int | None = None,
    ) -> list[WorldEpisode]:
        """
        Retrieve episodes from the reservoir.

        Args:
            k: number of episodes to return (capped by reservoir size).
            by_reward: when True return the k highest-reward episodes;
                otherwise the k most recently deposited.
            group_id: when given, only episodes deposited by this tribe.

        Returns:
            List of episodes (ranked by reward when ``by_reward``).
        """
        if k < 0:
            k = 0
        if group_id is not None:
            episodes = [e for e in self._episodes if e.group_id == group_id]
        else:
            episodes = self._episodes
        if not episodes:
            return []
        if by_reward:
            ranked = sorted(episodes, key=lambda e: e.reward, reverse=True)
            return ranked[:k]
        return episodes[-k:]

    def mean_reward(self, k: int = 5) -> float:
        """
        Average reward over the k highest-reward reservoir episodes.

        Args:
            k: how many episodes to average.

        Returns:
            Mean reward (0.0 when the reservoir is empty).
        """
        best = self.recall(k)
        if not best:
            return 0.0
        return float(np.mean([e.reward for e in best]))

    def __len__(self) -> int:
        """Number of episodes currently held."""
        return len(self._episodes)

    def stats(self) -> dict[str, Any]:
        """
        Observable summary of the reservoir.

        Returns:
            Dict with size, oldest/newest tick, mean reward, and per-tribe
            episode counts.
        """
        groups: dict[int, int] = {}
        for e in self._episodes:
            groups[e.group_id] = groups.get(e.group_id, 0) + 1
        return {
            "size": len(self._episodes),
            "oldest_tick": self._episodes[0].tick if self._episodes else 0,
            "newest_tick": self._episodes[-1].tick if self._episodes else 0,
            "mean_reward": self.mean_reward(),
            "groups": groups,
        }

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the reservoir to a JSON-safe dict.

        Returns:
            Dict with the full episode list (deposits are never dropped, so
            a serialized reservoir round-trips losslessly).
        """
        return {
            "episodes": [
                {
                    "features": e.features.tolist(),
                    "action": list(e.action),
                    "reward": e.reward,
                    "tick": e.tick,
                    "group_id": e.group_id,
                    "donor_id": e.donor_id,
                }
                for e in self._episodes
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorldMemory":
        """
        Rebuild a reservoir from :meth:`to_dict` output.

        Args:
            data: serialized reservoir dict.

        Returns:
            A new WorldMemory with the exact stored episodes.
        """
        episodes = [
            WorldEpisode(
                features=np.asarray(ep["features"], dtype=np.float32).copy(),
                action=tuple(float(a) for a in ep["action"]),
                reward=float(ep["reward"]),
                tick=int(ep["tick"]),
                group_id=int(ep.get("group_id", 0)),
                donor_id=int(ep.get("donor_id", 0)),
            )
            for ep in data.get("episodes", [])
        ]
        return cls(episodes)
