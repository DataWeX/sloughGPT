"""
Episodic memory for world-realm agents — a ring buffer of lived experiences.

Each episode records the perception features that preceded an action, the
action itself (the gates the cells perceptron emitted), and the reward that
followed (the energy delta). Only the most recent `capacity` episodes are
kept; the oldest are evicted. Retrieval returns the most recent episodes or
the highest-reward ones, which the baby's learning loop uses as a baseline
for "how well have I been doing lately" (surprise-based learning).
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
        if k < 0:
            k = 0
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
