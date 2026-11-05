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


_FILL_VERBS = r"type|enter|write|input|fill|search"
_FILL_VERB = re.compile(rf"\b(?:{_FILL_VERBS})\b", re.IGNORECASE)
_QUOTED = re.compile(r"[\"'`]([^\"'`]+)[\"'`]")
_INTO = re.compile(
    rf"\b(?:{_FILL_VERBS})\s+(?:the\s+)?(.+?)\s+(?:into|in|to)\s+(?:the\s+)?\w+",
    re.IGNORECASE,
)


def payload_from_task(task: str) -> str:
    """The text a fill choice should type.

    A task states its content either as a quote ("type 'hello' into the field")
    or as the object of type/enter/write into X ("Type hello into the note
    field"). A task that names an intent without content ("write a note") has no
    payload to extract, so its whole text is what gets typed.
    """
    text = (task or "").strip()
    quoted = _QUOTED.search(text)
    if quoted:
        return quoted.group(1).strip()
    into = _INTO.search(text)
    if into:
        return into.group(1).strip(" .,!")
    return text


def resolve_field(name: str, role: str):
    """A locator that finds a field by any name the accessibility tree used.

    The tree's "name" is the accessible name, which browsers derive from
    <label for>, aria-label, OR placeholder — while ElementLocator.label maps to
    the label channel alone. A placeholder-named input is therefore visible in
    the observation and unaddressable by label, so one lookup has to try every
    channel the name could have come from.
    """
    from avion.core.element import ElementLocator

    return ElementLocator.chain(
        ElementLocator.label(name),
        ElementLocator.placeholder(name),
        ElementLocator.role(role, name),
    )


def propose_candidates(
    accessibility_tree: dict[str, Any] | None,
    task: str = "",
    limit: int = 8,
) -> list[Choice]:
    """Ground a page into clickable/fillable choices (then DONE/FAIL).

    Walks a generic a11y tree ({role, name, children}); buttons and links
    become click choices, and a field becomes a fill choice carrying the payload
    read out of the task (see payload_from_task) only when the task asks for
    text to be entered.
    Always ends with wait, done, fail so the model can pause or stop.

    Presumes a backend that can resolve a name to an element: the closures
    below call session.click_text/fill, which go through find_element. On a
    desktop backend the tree is real but those calls fail, so a visible choice
    is not a reachable one.
    """

    choices: list[Choice] = []

    def _click(name: str):

        async def run(session) -> str:
            await session.click_text(name)
            return f"clicked {name!r}"

        return run

    def _fill(name: str, field_role: str, text: str):
        async def run(session) -> str:
            await session.fill(resolve_field(name, field_role), text)
            return f"typed {text!r} into {name!r}"

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
        elif (
            name
            and role in ("textbox", "searchbox", "input", "combobox")
            and task
            and _FILL_VERB.search(task)
        ):
            # A field earns a fill card only when the task asks for text to be
            # entered. Without that test the card reads "type 'Click the Clear
            # list button' into 'new note'" — wrong, and first in the list,
            # which is where a small sampler looks.
            payload = payload_from_task(task)
            choices.append(
                Choice(f"type {payload!r} into {name!r}", _fill(name, role, payload), kind="fill")
            )
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


def build_choice_prompt(
    task: str,
    choices: list[Choice],
    history: list[str] | None = None,
    unavailable: set[str] | None = None,
) -> str:
    """Compact numbered prompt — fits small contexts.

    history is the picks that failed. Successes are left out deliberately:
    echoed, a tiny sampler copies the last line instead of moving on, which is
    the loop this agent is meant to avoid.

    unavailable labels are shown but annotated. They are not removed, because
    removal renumbers every card after them and the model answers with a number
    it read from the previous list.
    """
    lines = [f"Task: {task}"]
    if history:
        lines += ["", "Already tried:"]
        lines += [f"- {h}" for h in history[-3:]]
    lines += ["", "Pick one number:", ""]
    for i, c in enumerate(choices, 1):
        note = " (tried, it did not work)" if unavailable and c.label in unavailable else ""
        lines.append(f"{i}. {c.label}{note}")
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

    async def _verify_goal(self, goal: list[dict[str, str]]) -> tuple[bool, str]:
        """Did the page reach its goal, and if not, which check said so."""
        from avion.ai.verifier import Verifier

        try:
            url = await self._session._backend.get_url()
        except Exception:
            url = ""
        try:
            text_now = str(await self._session._backend.evaluate("document.body.innerText") or "")
        except Exception:
            text_now = ""
        verify = await Verifier().verify(goal, url=url, text=text_now)
        return verify.passed, "; ".join(verify.failed)

    async def run(
        self,
        task: str,
        goal: list[dict[str, str]] | None = None,
        context: dict[str, Any] | None = None,
    ) -> ChoiceResult:
        start = time.monotonic()
        transcript: list[dict[str, Any]] = []
        stop_reason = "max_steps"
        success = False
        error = ""
        history: list[str] = []
        errors: dict[str, int] = {}
        dead: set[str] = set()
        repeat_of: str | None = None

        for _ in range(self._max_steps):
            try:
                tree = await self._session._backend.get_accessibility_tree()
            except Exception:
                tree = None
            choices = propose_candidates(tree, task=task, limit=self._candidate_limit)
            unavailable = dead | ({repeat_of} if repeat_of else set())
            prompt = build_choice_prompt(task, choices, history, unavailable)
            try:
                text = strip_echo(await self._generate(prompt), prompt)
            except Exception as e:
                error = f"inference error: {e}"
                stop_reason = "fail"
                break
            pick = parse_pick(text, len(choices))
            if pick is None:
                transcript.append({"pick": None, "output": text[:120]})
                history.append("no number was picked, so nothing happened")
                continue
            choice = choices[pick - 1]
            if choice.label in unavailable:
                # Withheld until a different number is picked, so escaping a
                # card costs a pick instead of buying a free repeat.
                history.append(f"{choice.label} -> did not work, pick a different number")
                transcript.append({"pick": pick, "label": choice.label, "observation": "refused"})
                continue
            repeat_of = None
            try:
                observation = await choice.execute(self._session)
            except _GiveUp:
                stop_reason = "give_up"
                error = "model gave up"
                break
            except Exception as e:
                observation = f"error: {e}"
                errors[choice.label] = errors.get(choice.label, 0) + 1
                if errors[choice.label] >= 2:
                    dead.add(choice.label)
            if observation.startswith("error:"):
                history.append(f"{choice.label} -> {observation}")
            transcript.append({"pick": pick, "label": choice.label, "observation": observation})
            if choice.kind == "done":
                if not goal:
                    success = True
                    stop_reason = "done"
                    break
                passed, why = await self._verify_goal(goal)
                if passed:
                    success = True
                    stop_reason = "done"
                else:
                    error = why
                    stop_reason = "unverified"
                break
            if choice.kind == "fail":
                stop_reason = "fail"
                error = "model picked fail"
                break
            if (
                goal
                and choice.kind in ("click", "fill", "goto")
                and not observation.startswith("error:")
            ):
                # The goal is checked after every step that changed the page,
                # not only when the model volunteers done: a tiny sampler keeps
                # acting on a page that has already met its goal, and a run that
                # ends by running out of steps reports that as failure.
                passed, why = await self._verify_goal(goal)
                if passed:
                    success = True
                    stop_reason = "done"
                    break
            if choice.kind != "wait":
                # Withholding the card just picked for one step is what turns a
                # repeating sampler into a sequence: a successful pick that did
                # not finish the task is one repeating cannot improve. "wait" is
                # exempt, since re-looking is the reason to pick it twice.
                repeat_of = choice.label

        return ChoiceResult(
            task=task,
            success=success,
            steps_taken=len(transcript),
            stop_reason=stop_reason,
            error=error,
            total_duration_ms=(time.monotonic() - start) * 1000,
            transcript=transcript,
        )
