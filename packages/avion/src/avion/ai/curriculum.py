"""Auto-curriculum — propose what the agent should practice next, then drive it.

Rule-based and deterministic suggestions: retry what fails (top failure
pattern), try what is untried (task list minus skilled tasks), otherwise
extend what works (harder variant of a mastered task). ``ExerciseRunner``
turns those proposals into a batch: it rides the agent's terminal DONE/FAIL
events, so one live agent works through the queue without a new run per
exercise.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from avion.ai.agent import AgentEvent


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
                proposals.append(
                    TaskProposal(
                        task=f"practice {action} until it succeeds",
                        reason=f"failed {count}x ({pattern[:80]})",
                        source="retry-failure",
                    )
                )
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
                    proposals.append(
                        TaskProposal(
                            task=task,
                            reason="not yet learned",
                            goal=[{"kind": "text_absent", "value": ""}],
                            source="untried",
                        )
                    )
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
                proposals.append(
                    TaskProposal(
                        task=f"{best.source_task or best.name} with an extra step",
                        reason=f"mastered '{best.name}' ({best.uses} uses)",
                        source="extend",
                    )
                )

        # 4. Fallback: explore.
        if not proposals:
            proposals.append(
                TaskProposal(
                    task="explore the current page and describe it",
                    reason="nothing to retry, try, or extend",
                    source="explore",
                )
            )
        return proposals[:max_suggestions] if max_suggestions else proposals


@dataclass
class ExerciseRecord:
    """How one exercise ended, read off the terminal event that closed it."""

    task: str
    outcome: str  # done | unverified | fail | unfinished
    reason: str = ""
    source: str = ""
    steps_from: int = 0
    steps_to: int = 0

    @property
    def steps_taken(self) -> int:
        return self.steps_to - self.steps_from

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "outcome": self.outcome,
            "reason": self.reason,
            "source": self.source,
            "steps_from": self.steps_from,
            "steps_to": self.steps_to,
        }


class ExerciseRunner:
    """Drives a queue of proposals through one live agent.

    ``Curriculum`` only ever suggested tasks and nothing executed them. This
    attaches to ``Agent.on_event``: every terminal DONE/FAIL is recorded as
    the result of the exercise in flight and, while proposals remain, vetoed
    so the same agent — backend, skills and experience buffer intact — picks
    up the next task without re-entering ``run()``.

        runner = ExerciseRunner(
            curriculum.suggest(known_tasks=tasks, max_suggestions=len(tasks))
        )
        agent.on_event(runner.handle)
        proposal = runner.start()
        result = await agent.run(proposal.task, context={"goal": proposal.goal})
        runner.finish(result.steps_taken, result.stop_reason)

    ``suggest`` defaults to one proposal, which leaves the runner nothing to
    hand over — ask for the batch size you intend to run.

    All exercises share one trajectory and one transcript file; records mark
    step boundaries (``steps_from``/``steps_to``) rather than owning slices of
    it, and ``AgentResult.task`` reports only the last exercise worked on.
    """

    def __init__(self, proposals: list[TaskProposal] | None = None) -> None:
        self._queue = deque(proposals or [])
        self._current: TaskProposal | None = None
        self._steps_at_start = 0
        self.records: list[ExerciseRecord] = []

    def start(self) -> TaskProposal | None:
        """Pop the first exercise, for the task/goal to pass to ``run()``."""
        self._begin(self._queue.popleft() if self._queue else None, at=0)
        return self._current

    def _begin(self, proposal: TaskProposal | None, at: int) -> None:
        self._current = proposal
        self._steps_at_start = at

    def handle(self, event: AgentEvent) -> None:
        """``on_event`` handler: record the finished exercise, queue the next.

        Without a next proposal it declines to veto, so the run ends exactly
        as it would with no runner attached.
        """
        if self._current is not None:
            outcome = event.kind
            if outcome == "done" and not event.trajectory.success:
                # The goal check failed: the agent claimed done, the page
                # disagreed. Record what the page said.
                outcome = "unverified"
            self.records.append(
                ExerciseRecord(
                    task=self._current.task,
                    outcome=outcome,
                    reason=event.reason,
                    source=self._current.source,
                    steps_from=self._steps_at_start,
                    steps_to=len(event.trajectory.steps),
                )
            )
        if not self._queue:
            self._begin(None, at=len(event.trajectory.steps))
            return
        proposal = self._queue.popleft()
        steps_now = len(event.trajectory.steps)
        event.continue_run = True
        event.task = proposal.task
        if proposal.goal:
            event.goal = proposal.goal
        self._begin(proposal, at=steps_now)

    @property
    def pending(self) -> list[TaskProposal]:
        """Exercises still queued — a stalled batch says what went undone."""
        return list(self._queue)

    def finish(self, steps_total: int, stop_reason: str) -> None:
        """Record the exercise still in flight when the run ended.

        A run can stop without any terminal action — max_steps, deadline, or
        the no-progress guard — and then ``handle`` never fires. Without this
        the batch would silently report fewer exercises than it attempted.
        No-op once the queue has drained, because the last exercise was
        already recorded by its own DONE/FAIL.
        """
        if self._current is None:
            return
        self.records.append(
            ExerciseRecord(
                task=self._current.task,
                outcome="unfinished",
                reason=stop_reason,
                source=self._current.source,
                steps_from=self._steps_at_start,
                steps_to=steps_total,
            )
        )
        self._begin(None, at=steps_total)

    def summary(self) -> dict[str, int]:
        """Outcome counts, so a batch of exercises reads at a glance."""
        counts: dict[str, int] = {}
        for record in self.records:
            counts[record.outcome] = counts.get(record.outcome, 0) + 1
        return counts
