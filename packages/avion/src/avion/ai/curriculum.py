"""Auto-curriculum — propose what the agent should practice next.

Rule-based and deterministic: retry what fails (top failure pattern),
try what is untried (task list minus skilled tasks), otherwise extend
what works (harder variant of a mastered task).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaskProposal:
    """The next task to attempt."""

    task: str
    reason: str = ""
    goal: list[dict[str, str]] = field(default_factory=list)
    source: str = ""  # retry-failure | untried | extend

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "reason": self.reason,
            "goal": self.goal,
            "source": self.source,
        }


class Curriculum:
    """Suggests the next task from skills, failures, and a task list.

    Usage::

        proposal = Curriculum().suggest(
            known_tasks=["open chat", "search datasets"],
            skills=library,
            feedback=agent.feedback_loop,
        )
    """

    def suggest(
        self,
        known_tasks: list[str] | None = None,
        skills=None,
        feedback=None,
        experience=None,
        max_suggestions: int = 1,
    ) -> list[TaskProposal]:
        proposals: list[TaskProposal] = []

        # 1. Retry what fails most.
        if feedback is not None:
            for pattern, count in feedback.failure_patterns():
                action = pattern.split(":", 1)[0]
                proposals.append(TaskProposal(
                    task=f"practice {action} until it succeeds",
                    reason=f"failed {count}x ({pattern[:80]})",
                    source="retry-failure",
                ))
                if len(proposals) >= max_suggestions:
                    return proposals

        # 2. Try what is untried.
        if known_tasks is not None:
            covered = set()
            if skills is not None:
                for name in skills.names:
                    skill = skills.get(name)
                    if skill is not None:
                        covered.add(skill.source_task)
                        covered.add(name)
            for task in known_tasks:
                if task not in covered:
                    proposals.append(TaskProposal(
                        task=task,
                        reason="not yet learned",
                        goal=[{"kind": "text_absent", "value": ""}],
                        source="untried",
                    ))
                    if len(proposals) >= max_suggestions:
                        return proposals

        # 3. Extend what works.
        if skills is not None and len(skills):
            best: Any = None
            for name in skills.names:
                skill = skills.get(name)
                if skill is not None and (best is None or skill.uses > best.uses):
                    best = skill
            if best is not None:
                proposals.append(TaskProposal(
                    task=f"{best.source_task or best.name} with an extra step",
                    reason=f"mastered '{best.name}' ({best.uses} uses)",
                    source="extend",
                ))

        # 4. Fallback: explore.
        if not proposals:
            proposals.append(TaskProposal(
                task="explore the current page and describe it",
                reason="nothing to retry, try, or extend",
                source="explore",
            ))
        return proposals[:max_suggestions] if max_suggestions else proposals
