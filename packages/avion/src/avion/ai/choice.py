"""Choice agent — computer use for small models.

JSON ReAct needs instruction-following a tiny sampler lacks. The choice
loop shrinks each decision to one letter: Avion grounds the page into a
lettered list of options (clicks, fills, navigation, DONE/FAIL), the model
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
    """One lettered option. run() executes it against a session."""

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


def propose_options(
    accessibility_tree: dict[str, Any] | None,
    task: str = "",
    limit: int = 8,
) -> list[Choice]:
    """Ground a page into clickable/fillable options (then DONE/FAIL).

    Walks a generic a11y tree ({role, name, children}); buttons and links
    become click choices, and a field becomes a fill choice carrying the payload
    read out of the task (see payload_from_task) only when the task asks for
    text to be entered.
    Always ends with wait, done, fail, and limit counts those three too: the
    list is what gets lettered A- and parsed back, so a card past position 8
    would have no readable letter.

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
        # limit is the size of the *list*, so the three stop cards are reserved
        # out of it rather than added on top: a page busy enough to fill limit
        # would otherwise push done and fail to I, J, K — letters parse_pick
        # does not read, and a task the model can never declare finished.
        if len(choices) >= limit - len(STOP_LABELS) or not isinstance(node, dict):
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
    for label, run, kind in zip(STOP_LABELS, (_noop, _done, _fail), STOP_KINDS):
        # Same words parse_pick matches on (STOP_LABELS); a card appended in
        # prose here and by list position there would drift the moment one of
        # them is reworded.
        choices.append(Choice(label, run, kind=kind))
    return choices


class _GiveUp(Exception):
    pass


# The cards propose_options appends to every page, in this order. Their
# labels are fixed words, so a model that copies one is naming a state of the
# run, not restating a page-dependent card — unlike the "1. type 'hello' into
# '" echo, whose digits are prompt punctuation. Hence the stop cards are matched
# by text, and parse_pick finds one at count-3..count.
STOP_LABELS = ("wait and look again", "task is done", "task cannot be done")
STOP_KINDS = ("wait", "done", "fail")


_LIST_MARKER = re.compile(r"^[ \t]*(-?\d+)[.)][ \t]", re.MULTILINE)
# "B. click 'Add note'" and a bare "B". A letter is only read as an answer when
# it opens a line and is either punctuated (B. / B)) or alone on its line, which
# keeps the article in "type a note" from being mistaken for card A.
_LINE_MARKER = re.compile(r"^[ \t]*([A-Ha-h])(?:[.)][ \t]|[.)]?[ \t]*$)", re.MULTILINE)
# A marker ("B. ", "2) ") at the start of a line, and only one, so what
# follows can be read as the content of a pick rather than as more list.
_LINE_PREFIX = re.compile(r"^[ \t]*-?(?:\d+|[A-Ha-h])[.)]?[ \t]+")


def _after_marker_line(text: str) -> list[str]:
    """The part of each line that follows a list marker, for marker-opening lines."""
    out = []
    for line in text.splitlines():
        match = _LINE_PREFIX.match(line)
        if match:
            out.append(line[match.end() :])
    return out


def parse_pick(text: str, count: int) -> int | None:
    """First in-range pick that is an answer, not a copied option list.

    Asked for a number, a 0.5B continues the list instead of replying to it:
    "1. type 'hello' into 'new note'". Those digits are the prompt's own
    punctuation, and reading them as a choice pins every step to card 1. A real
    number arrives bare, so a number that opens a line and is followed by ". " or
    ") " is a restatement and gets skipped.

    The options are lettered now because lettering is what broke the copy: the
    same sampler answers "B. click 'Add note'" against a lettered list. There the
    punctuated form IS the answer, so a leading letter is read as a pick rather
    than discarded, and only the first one counts.

    A letter, though, is the model's weakest signal — it may aim at the letter
    or at the text beside it. So text wins when it names one of the stop cards:
    "B. task is done" is done (5), not card B.
    """
    body = text or ""
    first_stop = count - len(STOP_LABELS)
    for content in _after_marker_line(body):
        if content.strip() in STOP_LABELS and 0 <= first_stop < 8:
            return first_stop + STOP_LABELS.index(content.strip()) + 1
    for match in _LINE_MARKER.finditer(body):
        pick = ord(match.group(1).upper()) - ord("A") + 1
        if 1 <= pick <= count:
            return pick
    markers = {m.start(1) for m in _LIST_MARKER.finditer(body)}
    for match in re.finditer(r"-?\d+", body):
        if match.start() in markers:
            continue
        pick = int(match.group())
        if 1 <= pick <= count:
            return pick
    return None


def build_choice_prompt(
    task: str,
    choices: list[Choice],
    history: list[str] | None = None,
    unavailable: set[str] | None = None,
) -> str:
    """Compact lettered prompt — fits small contexts.

    Letters, not digits: asked to pick a number, the 0.5B restarts the list and
    copies it ("1. type 'hello' into '"), which is the first card whatever the
    task is. The same sampler lettered as "A. B. C." answers with a card it chose,
    "B. click 'Add note'". option_limit tops out at 8, so A-H is enough.

    history is the picks that failed. Successes are left out deliberately:
    echoed, a tiny sampler copies the last line instead of moving on, which is
    the loop this agent is meant to avoid.

    unavailable labels are shown but annotated. They are not removed, because
    removal renumbers every card after them and the model answers with a letter
    it read from the previous list.
    """
    lines = [f"Task: {task}"]
    if history:
        lines += ["", "Already tried:"]
        lines += [f"- {h}" for h in history[-3:]]
    lines += ["", "Pick one letter:", ""]
    for i, c in enumerate(choices, 1):
        note = " (tried, it did not work)" if unavailable and c.label in unavailable else ""
        lines.append(f"{chr(ord('A') + i - 1)}. {c.label}{note}")
    lines.append("Letter:")
    return "\n".join(lines)


def build_next_action_prompt(
    task: str,
    choices: list[Choice],
    history: list[str] | None = None,
) -> str:
    """The task, the page as a sentence, and a question — the other framing.

    No lettered list, so nothing to continue and no letter to aim at: the model
    says what happens next and parse_action works out which option that was.
    Recorded against the live 0.5B, the lettered menu names click 'Add note' on
    the fill-first task 15/15 (and names it when it is card A, too, so that is a
    preference, not a position bias), while this shape stated the fill — which is
    the right first step — 4/4.

    Caveat kept in the open: that 4/4 is the task's own sentence, so it is as
    consistent with the model reciting the plan as with it reading the page. The
    second step is what tells the two apart, and the batch is where the page
    really changes.

    The options are inlined rather than listed because a list, lettered or not,
    is what the sampler restarts.
    """
    page = ", ".join(c.label for c in choices if c.kind in ("click", "fill"))
    lines = [f"Task: {task}"]
    if page:
        lines.append(f"The page offers: {page}.")
    if history:
        lines += ["", "Already tried:"]
        lines += [f"- {h}" for h in history[-3:]]
    lines += ["", "What does it do next? Answer in a few words.", "", "Next:"]
    return "\n".join(lines)


_WORD_SIGS = {
    "wait": ("wait", "look again", "check"),
    "done": ("done", "finished", "complete", "nothing left"),
    "fail": ("cannot", "can't", "impossible", "no way", "fail", "stuck"),
}
_QUOTED_IN_LABEL = re.compile(r"'([^']+)'")


def _phrases(text: str) -> tuple[str, ...]:
    """Each word of a label phrase, tolerant of words in between.

    'Clear list' has to answer to "Clear the list" and 'Add note' to "the Add
    note button", so the words are matched with a short gap rather than as a
    contiguous substring. The gap is bounded to keep an unrelated pair of words
    in a long answer from reading as one action.
    """
    words = [re.escape(w.lower()) for w in text.split() if w]
    if not words:
        return ()
    gap = r"\s+(?:\w+\s+){0,2}"
    return (gap.join(words),) if len(words) > 1 else (words[0],)


def signatures(choice: Choice) -> tuple[str, ...]:
    """Patterns that identify this option in running text.

    Deliberately no bare verbs: 'enter' is a fill verb, and "press Enter" is not
    one — a name, or the payload, has to appear. A reply that names nothing is
    no pick, and the loop re-observes instead of acting on a guess.
    """
    if choice.kind in _WORD_SIGS:
        return tuple(p for word in _WORD_SIGS[choice.kind] for p in _phrases(word))
    quoted = _QUOTED_IN_LABEL.findall(choice.label)
    if not quoted:
        return _phrases(choice.label)
    return tuple(p for name in quoted for p in _phrases(name))


def parse_action(text: str, choices: list[Choice], skip: set[str] | None = None) -> int | None:
    """The option named earliest in the reply — the first thing it says to do.

    Order, not priority: "Type hello into the note field, then click Add note"
    mentions both cards, and the run needs the one that comes first, because that
    is the step the page is waiting for. Matching by priority would answer the
    same words with the same card at every step.

    skip holds the labels the run has already done and withheld. Excluding them
    from the *match*, rather than refusing the pick afterwards, is what lets a
    restated plan advance: "type hello, then click Add note" names the fill first
    and the click second, so once the fill is spent the second half of the same
    sentence is the answer. Measured against the live 0.5B, without this the
    framing named the fill correctly and then recited it until the step budget
    ran out.
    """
    low = (text or "").lower()
    best: tuple[int, int] | None = None
    for i, choice in enumerate(choices, 1):
        if skip and choice.label in skip:
            continue
        hit = -1
        for pattern in signatures(choice):
            match = re.search(pattern, low)
            if match and (hit < 0 or match.start() < hit):
                hit = match.start()
        if hit >= 0 and (best is None or hit < best[0]):
            best = (hit, i)
    return best[1] if best else None


ALREADY_SATISFIED = (
    "goal was already satisfied before the first action, so no step can earn it; "
    "the exercise needs a start state where the goal is false"
)


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
    """Runs the choose-the-next-step loop against an Avion session.

    Usage::

        agent = ChoiceAgent(session, generate=tiny_sampler)
        result = await agent.run("open chat", goal=[...])

    framing="menu" asks for a letter off a list and reads the letter;
    framing="next" asks what happens next and reads the words. See
    build_choice_prompt and build_next_action_prompt for what each costs.
    """

    def __init__(
        self,
        session,
        generate: Callable[[str], Awaitable[str]],
        max_steps: int = 20,
        option_limit: int = 8,
        framing: str = "menu",
    ):
        self._session = session
        self._generate = generate
        self._max_steps = max_steps
        self._option_limit = option_limit
        self._framing = framing

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
        # A goal the page already met before the first action is not this run's
        # achievement: the post-step check below would credit any action that
        # left the page alone, so clicking a no-op button reads as success.
        # Require a false -> true transition, and say so when it is impossible.
        if goal:
            pre_passed, _ = await self._verify_goal(goal)
            goal_pre_held = pre_passed
        else:
            goal_pre_held = False

        for _ in range(self._max_steps):
            try:
                tree = await self._session._backend.get_accessibility_tree()
            except Exception:
                tree = None
            choices = propose_options(tree, task=task, limit=self._option_limit)
            unavailable = dead | ({repeat_of} if repeat_of else set())
            if self._framing == "next":
                prompt = build_next_action_prompt(task, choices, history)
            else:
                prompt = build_choice_prompt(task, choices, history, unavailable)
            try:
                text = strip_echo(await self._generate(prompt), prompt)
            except Exception as e:
                error = f"inference error: {e}"
                stop_reason = "fail"
                break
            pick = (
                parse_action(text, choices, skip=unavailable)
                if self._framing == "next"
                else parse_pick(text, len(choices))
            )
            if pick is None:
                transcript.append({"pick": None, "output": text[:120]})
                history.append(
                    "nothing was named, so nothing happened"
                    if self._framing == "next"
                    else "no letter was picked, so nothing happened"
                )
                continue
            choice = choices[pick - 1]
            if choice.label in unavailable:
                # Withheld until a different letter is picked, so escaping a
                # card costs a pick instead of buying a free repeat.
                history.append(
                    f"{choice.label} -> did not work, name a different action"
                    if self._framing == "next"
                    else f"{choice.label} -> did not work, pick a different letter"
                )
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
                if passed and not goal_pre_held:
                    success = True
                    stop_reason = "done"
                else:
                    error = ALREADY_SATISFIED if passed else why
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
                if passed and goal_pre_held:
                    # Nothing left to earn and no action can earn it: the
                    # exercise is malformed, not the policy. Stop saying so at
                    # the first step instead of burning the whole step budget.
                    error = ALREADY_SATISFIED
                    stop_reason = "unverified"
                    break
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
