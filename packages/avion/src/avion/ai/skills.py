"""Skill library — keep trajectories that worked, replay them later.

A skill is a named, described sequence of actions distilled from a
successful trajectory. Skills persist as JSON files and replay through
the agent loop via EchoModel.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from avion.ai.models import Action, Trajectory


@dataclass
class Skill:
    """Reusable action sequence with a name and description."""

    name: str
    description: str = ""
    actions: list[Action] = field(default_factory=list)
    source_task: str = ""
    uses: int = 0
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "actions": [a.to_dict() for a in self.actions],
            "source_task": self.source_task,
            "uses": self.uses,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Skill:
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            actions=[Action.from_dict(a) for a in data.get("actions", [])],
            source_task=data.get("source_task", ""),
            uses=data.get("uses", 0),
            created_at=data.get("created_at", 0),
        )

    def to_echo_model(self):
        """Replay this skill through an Agent. Marks one use."""
        from avion.ai.models import EchoModel
        self.uses += 1
        return EchoModel(list(self.actions))


class SkillLibrary:
    """Named skills backed by a directory of JSON files.

    Usage::

        lib = SkillLibrary("skills/")
        lib.add_from_trajectory(traj, name="open_chat",
                                description="go to chat and start")
        skill = lib.find("chat")[0]
        agent = Agent(model=skill.to_echo_model())
    """

    def __init__(self, directory: str = "skills"):
        self.directory = directory
        self._skills: dict[str, Skill] = {}

    def __len__(self) -> int:
        return len(self._skills)

    @property
    def names(self) -> list[str]:
        return sorted(self._skills)

    def add(self, skill: Skill) -> None:
        self._skills[skill.name] = skill

    def add_from_trajectory(
        self,
        trajectory: Trajectory,
        name: str,
        description: str = "",
        skip_waits: bool = True,
    ) -> Skill:
        """Distill a successful trajectory into a skill.

        Only keeps steps with actions (drops observations); WAIT steps
        are skipped by default since recorded pauses rarely replay well.
        """
        from avion.ai.models import ActionType
        actions = [
            s.action for s in trajectory.steps
            if s.action is not None
            and not (skip_waits and s.action.action_type == ActionType.WAIT)
        ]
        skill = Skill(name=name, description=description,
                      actions=actions, source_task=trajectory.task)
        self.add(skill)
        return skill

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def find(self, keywords: str) -> list[Skill]:
        """Substring match over name + description + source task."""
        words = keywords.lower().split()
        scored = []
        for skill in self._skills.values():
            hay = f"{skill.name} {skill.description} {skill.source_task}".lower()
            score = sum(1 for w in words if w in hay)
            if score:
                scored.append((score, skill))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [s for _, s in scored]

    def delete(self, name: str) -> bool:
        if name in self._skills:
            del self._skills[name]
            path = Path(self.directory) / f"{name}.json"
            if path.exists():
                path.unlink()
            return True
        return False

    def save(self, directory: str | None = None) -> int:
        """Write every skill as <name>.json. Returns count saved."""
        target = Path(directory or self.directory)
        target.mkdir(parents=True, exist_ok=True)
        for skill in self._skills.values():
            with open(target / f"{skill.name}.json", "w") as f:
                json.dump(skill.to_dict(), f, indent=2, default=str)
        return len(self._skills)

    def load(self, directory: str | None = None) -> int:
        """Load every <name>.json. Returns count loaded."""
        target = Path(directory or self.directory)
        count = 0
        if not target.is_dir():
            return 0
        for path in sorted(target.glob("*.json")):
            try:
                with open(path) as f:
                    skill = Skill.from_dict(json.load(f))
                self._skills[skill.name] = skill
                count += 1
            except Exception:
                continue
        return count

    def clear(self) -> None:
        self._skills.clear()
