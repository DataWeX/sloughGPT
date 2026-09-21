"""Learning system — experience replay buffer and feedback loop."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from arken.ai.models import Action, Step, Trajectory


@dataclass
class Experience:
    """One recorded (state, action, reward) tuple."""

    task: str
    screenshot_b64: str = ""
    accessibility_tree: dict[str, Any] | None = None
    action: Action | None = None
    reward: float = 0.0
    next_screenshot_b64: str = ""
    done: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "action": self.action.to_dict() if self.action else None,
            "reward": self.reward,
            "done": self.done,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


class ExperienceBuffer:
    """Stores agent experiences, in memory with JSON persistence."""

    def __init__(self, max_size: int = 10000):
        self._experiences: list[Experience] = []
        self._max_size = max_size

    @property
    def size(self) -> int:
        return len(self._experiences)

    def add(self, experience: Experience) -> None:
        self._experiences.append(experience)
        if len(self._experiences) > self._max_size:
            self._experiences = self._experiences[-self._max_size :]

    def get_by_task(self, task: str, limit: int = 100) -> list[Experience]:
        return [e for e in self._experiences if e.task == task][-limit:]

    def get_high_reward(self, min_reward: float = 0.5, limit: int = 100) -> list[Experience]:
        return [e for e in self._experiences if e.reward >= min_reward][-limit:]

    def get_recent(self, limit: int = 100) -> list[Experience]:
        return self._experiences[-limit:]

    def get_trajectories(self, task: str | None = None) -> list[Trajectory]:
        """Reconstruct trajectories from sequential experiences."""
        exps = self._experiences
        if task:
            exps = [e for e in exps if e.task == task]

        trajectories: list[Trajectory] = []
        current: list[Experience] = []

        for exp in exps:
            current.append(exp)
            if exp.done:
                steps = [
                    Step(
                        step_number=i,
                        screenshot_b64=e.screenshot_b64,
                        action=e.action,
                        reward=e.reward,
                        done=e.done,
                    )
                    for i, e in enumerate(current)
                ]
                traj = Trajectory(
                    task=task or current[0].task if current else "",
                    steps=steps,
                    success=current[-1].reward > 0 if current else False,
                )
                trajectories.append(traj)
                current = []

        return trajectories

    def successful_trajectories(self, task: str | None = None) -> list[Trajectory]:
        return [t for t in self.get_trajectories(task) if t.success]

    def stats(self) -> dict[str, Any]:
        if not self._experiences:
            return {"total": 0}
        rewards = [e.reward for e in self._experiences]
        tasks = set(e.task for e in self._experiences)
        return {
            "total": len(self._experiences),
            "tasks": len(tasks),
            "avg_reward": sum(rewards) / len(rewards),
            "max_reward": max(rewards),
            "min_reward": min(rewards),
            "success_rate": sum(1 for r in rewards if r > 0) / len(rewards),
        }

    def save(self, path: str) -> None:
        data = [e.to_dict() for e in self._experiences]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def load(self, path: str) -> None:
        with open(path) as f:
            data = json.load(f)
        self._experiences.clear()
        for item in data:
            action = Action.from_dict(item["action"]) if item.get("action") else None
            self._experiences.append(
                Experience(
                    task=item["task"],
                    action=action,
                    reward=item.get("reward", 0),
                    done=item.get("done", False),
                    metadata=item.get("metadata", {}),
                    timestamp=item.get("timestamp", 0),
                )
            )

    def clear(self) -> None:
        self._experiences.clear()


class FeedbackLoop:
    """Human feedback: ratings, corrections, failure patterns."""

    def __init__(self):
        self._feedback: list[dict[str, Any]] = []
        self._patterns: dict[str, int] = {}

    def record_feedback(self, step: Step, rating: float, comment: str = "") -> None:
        entry = {
            "step": step.to_dict(),
            "rating": rating,
            "comment": comment,
            "timestamp": time.time(),
        }
        self._feedback.append(entry)
        if rating < 0 and step.action:
            key = f"{step.action.action_type.value}:{step.observation[:50]}"
            self._patterns[key] = self._patterns.get(key, 0) + 1

    def record_correction(self, step: Step, corrected_action: Action) -> None:
        self._feedback.append(
            {
                "type": "correction",
                "original": step.to_dict() if step else None,
                "corrected_action": corrected_action.to_dict(),
                "timestamp": time.time(),
            }
        )

    def failure_patterns(self, min_count: int = 2) -> list[tuple[str, int]]:
        patterns = [(k, v) for k, v in self._patterns.items() if v >= min_count]
        patterns.sort(key=lambda x: x[1], reverse=True)
        return patterns

    def improvement_suggestions(self) -> list[str]:
        suggestions = []
        for pattern, count in self.failure_patterns():
            action_type, observation = pattern.split(":", 1)
            suggestions.append(
                f"Pattern '{action_type}' failed {count}x with observation '{observation}'. "
                f"Consider adding a rule or adjusting the model's behavior for this case."
            )
        return suggestions

    def stats(self) -> dict[str, Any]:
        if not self._feedback:
            return {"total": 0}
        ratings = [f["rating"] for f in self._feedback if "rating" in f]
        return {
            "total": len(self._feedback),
            "avg_rating": sum(ratings) / len(ratings) if ratings else 0.0,
            "corrections": sum(1 for f in self._feedback if f.get("type") == "correction"),
            "unique_patterns": len(self._patterns),
        }

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._feedback, f, indent=2, default=str)

    def load(self, path: str) -> None:
        with open(path) as f:
            self._feedback = json.load(f)
