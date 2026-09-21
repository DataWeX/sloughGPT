"""Event replay — re-run recorded events against a live session.

Recorded CLICK/FILL events carry locator descriptions ("css=input",
"text(contains)='Start'"); replay parses them back into locators and
drives an Arken session. NAVIGATE events re-goto the recorded path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from arken.core.element import ElementLocator
from arken.events.models import Event, EventType


def parse_locator(description: str) -> ElementLocator | None:
    """Parse a locator describe() string back into a locator."""
    if description.startswith("css="):
        return ElementLocator.css(description[4:])
    m = re.fullmatch(r"text\((?:exact|contains)\)='(.*)'", description)
    if m:
        exact = description.startswith("text(exact)")
        return ElementLocator.text(m.group(1).replace("\\'", "'"), exact=exact)
    if description.startswith("role="):
        return ElementLocator.role(description[5:])
    if description.startswith("testId="):
        raw = description[7:].strip("'")
        return ElementLocator.test_id(raw)
    if description.startswith("label="):
        return ElementLocator.label(description[6:].strip("'"))
    return None


@dataclass
class ReplayConfig:
    """How to replay."""

    stop_on_error: bool = True
    dry_run: bool = False  # resolve locators, skip actions


@dataclass
class ReplayStep:
    """Outcome for one replayed event."""

    event_seq: int
    action: str
    success: bool = True
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_seq": self.event_seq,
            "action": self.action,
            "success": self.success,
            "error": self.error,
        }


@dataclass
class ReplayResult:
    """Outcome of a full replay."""

    steps: list[ReplayStep] = field(default_factory=list)
    success: bool = True

    @property
    def passed(self) -> int:
        return sum(1 for s in self.steps if s.success)

    @property
    def failed(self) -> int:
        return sum(1 for s in self.steps if not s.success)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "passed": self.passed,
            "failed": self.failed,
            "steps": [s.to_dict() for s in self.steps],
        }


class EventReplay:
    """Replays events through a session with goto/find/click/fill.

    Usage::

        replay = EventReplay(session)
        result = await replay.replay(recorder.events)
    """

    def __init__(self, session, config: ReplayConfig | None = None):
        self._session = session
        self._config = config or ReplayConfig()

    async def replay(self, events: list[Event]) -> ReplayResult:
        result = ReplayResult()
        for event in events:
            step = await self._replay_one(event)
            if step is None:
                continue  # event carries no action (e.g. session markers)
            result.steps.append(step)
            if not step.success:
                result.success = False
                if self._config.stop_on_error:
                    break
        return result

    async def _replay_one(self, event: Event) -> ReplayStep | None:
        try:
            if event.type == EventType.NAVIGATE:
                return await self._do(
                    event, f"goto {event.name}", lambda: self._session.goto(event.name)
                )
            if event.type in (EventType.CLICK, EventType.FILL):
                locator = parse_locator(event.name)
                if locator is None:
                    return ReplayStep(
                        event.seq, f"{event.type.value} {event.name}", False, "unparseable locator"
                    )
                if event.type == EventType.CLICK:
                    el = await self._session.find(locator)
                    if not self._config.dry_run:
                        await self._session.click(el)
                else:
                    value = str((event.data or {}).get("value", ""))
                    if not self._config.dry_run:
                        await self._session.fill(locator, value)
                return ReplayStep(event.seq, f"{event.type.value} {event.name}")
        except Exception as e:
            return ReplayStep(event.seq, event.type.value, False, str(e))
        return None

    async def _do(self, event: Event, action: str, run) -> ReplayStep:
        try:
            if not self._config.dry_run:
                await run()
            return ReplayStep(event.seq, action)
        except Exception as e:
            return ReplayStep(event.seq, action, False, str(e))
