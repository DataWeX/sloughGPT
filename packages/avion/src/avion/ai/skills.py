"""Skill library — keep trajectories that worked, replay them later.

A skill is a named, described sequence of actions distilled from a
successful trajectory. Skills persist as JSON files and replay through
the agent loop via EchoModel.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from avion.ai.models import Action, Trajectory

#: Bump when the on-disk skill JSON layout changes. Files written before
#: versioning existed load as v0 (their missing fields take defaults);
#: files from a NEWER build are skipped loudly rather than half-read.
SCHEMA_VERSION = 1

_log = logging.getLogger(__name__)

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def skill_filename(name: str) -> str:
    """Filesystem-safe filename for a skill name (never a path traversal)."""
    safe = _SAFE_NAME.sub("_", name).strip("._")
    return f"{safe or 'skill'}.json"


def trajectory_hash(actions: list[Action], task: str) -> str:
    """Stable hash of what a skill actually replays: its task + action list.

    Used to distill the same trajectory twice without writing it twice.
    """
    payload = json.dumps(
        {"task": task, "actions": [a.to_dict() for a in actions]},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class Skill:
    """Reusable action sequence with a name and description.

    Replay stats: ``uses`` counts replays started (``to_echo_model``);
    ``successes``/``failures`` are recorded outcomes (``record_replay``),
    which is what ``success_rate`` reports. Unproven skills rate ``None``.
    """

    name: str
    description: str = ""
    actions: list[Action] = field(default_factory=list)
    source_task: str = ""
    uses: int = 0
    successes: int = 0
    failures: int = 0
    source_hash: str = ""
    created_at: float = field(default_factory=time.time)

    @property
    def success_rate(self) -> float | None:
        """Replay success rate, or None when no outcome has been recorded."""
        total = self.successes + self.failures
        return self.successes / total if total else None

    def record_replay(self, success: bool) -> None:
        """Record the outcome of a replay so ``find`` can rank by it."""
        if success:
            self.successes += 1
        else:
            self.failures += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "name": self.name,
            "description": self.description,
            "actions": [a.to_dict() for a in self.actions],
            "source_task": self.source_task,
            "uses": self.uses,
            "successes": self.successes,
            "failures": self.failures,
            "source_hash": self.source_hash,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Skill:
        version = data.get("schema_version", 0)
        if not isinstance(version, int) or isinstance(version, bool):
            raise ValueError(f"skill schema_version must be an int, got {version!r}")
        if version > SCHEMA_VERSION:
            raise ValueError(
                f"skill schema v{version} is newer than this build supports "
                f"(v{SCHEMA_VERSION}) — upgrade avion to load it"
            )
        # v0 predates versioning; every newer field falls back to its default.
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            actions=[Action.from_dict(a) for a in data.get("actions", [])],
            source_task=data.get("source_task", ""),
            uses=data.get("uses", 0),
            successes=data.get("successes", 0),
            failures=data.get("failures", 0),
            source_hash=data.get("source_hash", ""),
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
        #: files ``load()`` skipped, and why — never silently dropped.
        self.load_errors: list[dict[str, str]] = []

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

        Deduped by trajectory hash: distilling the same task with the
        same resulting actions twice returns the existing skill instead
        of writing a second copy.
        """
        from avion.ai.models import ActionType

        actions = [
            s.action
            for s in trajectory.steps
            if s.action is not None and not (skip_waits and s.action.action_type == ActionType.WAIT)
        ]
        digest = trajectory_hash(actions, trajectory.task)
        for existing in self._skills.values():
            if existing.source_hash and existing.source_hash == digest:
                return existing
        skill = Skill(
            name=name,
            description=description,
            actions=actions,
            source_task=trajectory.task,
            source_hash=digest,
        )
        self.add(skill)
        return skill

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    @staticmethod
    def _success_key(skill: Skill) -> float:
        """Rank key for an unproven skill: neutral, so proven beats failed."""
        rate = skill.success_rate
        return 0.5 if rate is None else rate

    def find(self, keywords: str) -> list[Skill]:
        """Substring match over name + description + source task.

        Ranked by keyword score, then by replay success rate — relevance
        decides *whether* a skill matches, proven reliability only breaks
        ties, so a perfect textual match never loses to an untested one.
        """
        words = keywords.lower().split()
        scored = []
        for skill in self._skills.values():
            hay = f"{skill.name} {skill.description} {skill.source_task}".lower()
            score = sum(1 for w in words if w in hay)
            if score:
                scored.append((score, self._success_key(skill), skill))
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        return [s for _, _, s in scored]

    def delete(self, name: str) -> bool:
        if name in self._skills:
            del self._skills[name]
            path = Path(self.directory) / skill_filename(name)
            if path.exists():
                path.unlink()
            return True
        return False

    def save(self, directory: str | None = None) -> int:
        """Write every skill as <name>.json. Returns count saved.

        Atomic per file: write a sibling ``.tmp``, fsync, then rename, so
        an interrupted run can never leave a half-written skill behind.
        """
        target = Path(directory or self.directory)
        target.mkdir(parents=True, exist_ok=True)
        for skill in self._skills.values():
            path = target / skill_filename(skill.name)
            tmp = path.with_name(path.name + ".tmp")
            with open(tmp, "w") as f:
                json.dump(skill.to_dict(), f, indent=2, default=str)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        return len(self._skills)

    def load(self, directory: str | None = None) -> int:
        """Load every <name>.json. Returns count loaded.

        Unreadable or too-new files are skipped **loudly**: each lands in
        ``load_errors`` and is logged, never dropped without a trace.
        """
        target = Path(directory or self.directory)
        self.load_errors = []
        count = 0
        if not target.is_dir():
            return 0
        for path in sorted(target.glob("*.json")):
            try:
                with open(path) as f:
                    skill = Skill.from_dict(json.load(f))
            except Exception as e:
                reason = str(e) or type(e).__name__
                self.load_errors.append({"file": path.name, "reason": reason})
                _log.warning("skipping skill file %s: %s", path.name, reason)
                continue
            self._skills[skill.name] = skill
            count += 1
        return count

    def clear(self) -> None:
        self._skills.clear()
