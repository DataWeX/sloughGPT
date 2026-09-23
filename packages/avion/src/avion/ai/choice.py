"""Choice agent — computer use for small models.

JSON ReAct needs instruction-following a tiny sampler lacks. The choice
loop shrinks each decision to one number: Avion grounds the page into a
numbered candidate list (clicks, fills, navigation, DONE/FAIL), the model
replies with any text containing its pick, Avion executes it.

Grounding stays in code; the model only expresses preference.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from avion.ai.slo import strip_echo


@dataclass
class Choice:
    """One numbered option. run() executes it against a session."""

    label: str
    run: Callable[[Any], Awaitable[str]] = field(repr=False)
    kind: str = "action"  # click | fill | goto | done | fail | wait

    async def execute(self, session) -> str:
        return await self.run(session)


def propose_candidates(
    accessibility_tree: dict[str, Any] | None,
    task: str = "",
    limit: int = 8,
) -> list[Choice]:
    """Ground a page into clickable/fillable choices (then DONE/FAIL).

    Walks a generic a11y tree ({role, name, children}); buttons and links
    become click choices, textboxes become fill-with-task choices.
    Always ends with wait, done, fail so the model can pause or stop.
    """

    choices: list[Choice] = []

    def _click(name: str):
        async def run(session) -> str:
            await session.click_text(name)
            return f"clicked {name!r}"
        return run

    def _fill(name: str, text: str):
        async def run(session) -> str:
            from avion.core.element import ElementLocator
            await session.fill(ElementLocator.label(name), text)
            return f"typed into {name!r}"
        return run

    async def _noop(session) -> str:
        return "waited"

    async def _done(session) -> str:
        return "done"

    async def _fail(session) -> str:
        raise _GiveUp("model gave up")

    def walk(node: Any):
        if len(choices) >= limit or not isinstance(node, dict):
            return
        role = str(node.get("role", "")).lower()
        name = str(node.get("name", "") or node.get("text", "")).strip()
        if name and role in ("button", "link", "tab", "menuitem", "option"):
            choices.append(Choice(f"click {name!r}", _click(name), kind="click"))
        elif name and role in ("textbox", "searchbox", "input", "combobox") and task:
            choices.append(Choice(
                f"type {task!r} into {name!r}", _fill(name, task), kind="fill"))
        for child in node.get("children", []) or []:
            walk(child)

    walk(accessibility_tree or {})
    choices.append(Choice("wait and look again", _noop, kind="wait"))
    choices.append(Choice("task is done", _done, kind="done"))
    choices.append(Choice("task cannot be done", _fail, kind="fail"))
    return choices


class _GiveUp(Exception):
    pass


def parse_pick(text: str, count: int) -> int | None:
    """First in-range integer in model output, else None (→ re-observe)."""
    for match in re.findall(r"-?\d+", text or ""):
        pick = int(match)
        if 1 <= pick <= count:
            return pick
    return None


def build_choice_prompt(task: str, choices: list[Choice]) -> str:
    """Compact numbered prompt — fits small contexts."""
    lines = [f"Task: {task}", "Pick one number:", ""]
    lines += [f"{i}. {c.label}" for i, c in enumerate(choices, 1)]
    lines.append("Number:")
    return "\n".join(lines)


@dataclass
class ChoiceResult:
    """Outcome of a choice-agent run."""

    task: str
    success: bool = False
    steps_taken: int = 0
    stop_reason: str = ""  # done|fail|unverified|max_steps|give_up
    error: str = ""
    total_duration_ms: float = 0.0
    transcript: list[dict[str, Any]] = field(default_factory=list)


class ChoiceAgent:
    """Runs the pick-a-number loop against an Avion session.

    Usage::

        agent = ChoiceAgent(session, generate=tiny_sampler)
        result = await agent.run("open chat", goal=[...])
    """

    def __init__(
        self,
        session,
        generate: Callable[[str], Awaitable[str]],
        max_steps: int = 20,
        candidate_limit: int = 8,
    ):
        self._session = session
        self._generate = generate
        self._max_steps = max_steps
        self._candidate_limit = candidate_limit

    async def run(
        self,
        task: str,
        goal: list[dict[str, str]] | None = None,
        context: dict[str, Any] | None = None,
    ) -> ChoiceResult:
        from avion.ai.verifier import Verifier

        start = time.monotonic()
        transcript: list[dict[str, Any]] = []
        stop_reason = "max_steps"
        success = False
        error = ""

        for _ in range(self._max_steps):
            try:
                tree = await self._session._backend.get_accessibility_tree()
            except Exception:
                tree = None
            choices = propose_candidates(tree, task=task, limit=self._candidate_limit)
            prompt = build_choice_prompt(task, choices)
            try:
                text = strip_echo(await self._generate(prompt), prompt)
            except Exception as e:
                error = f"inference error: {e}"
                stop_reason = "fail"
                break
            pick = parse_pick(text, len(choices))
            if pick is None:
                transcript.append({"pick": None, "output": text[:120]})
                continue
            choice = choices[pick - 1]
            try:
                observation = await choice.execute(self._session)
            except _GiveUp:
                stop_reason = "give_up"
                error = "model gave up"
                break
            except Exception as e:
                observation = f"error: {e}"
            transcript.append({"pick": pick, "label": choice.label,
                               "observation": observation})
            if choice.kind == "done":
                if not goal:
                    success = True
                    stop_reason = "done"
                    break
                try:
                    url = await self._session._backend.get_url()
                except Exception:
                    url = ""
                try:
                    text_now = str(await self._session._backend.evaluate(
                        "document.body.innerText") or "")
                except Exception:
                    text_now = ""
                verify = await Verifier().verify(goal, url=url, text=text_now)
                if verify.passed:
                    success = True
                    stop_reason = "done"
                else:
                    error = "; ".join(verify.failed)
                    stop_reason = "unverified"
                break
            if choice.kind == "fail":
                stop_reason = "fail"
                error = "model picked fail"
                break

        return ChoiceResult(
            task=task,
            success=success,
            steps_taken=len(transcript),
            stop_reason=stop_reason,
            error=error,
            total_duration_ms=(time.monotonic() - start) * 1000,
            transcript=transcript,
        )
