"""complete — the flow-completion verb: drive a recorded flow to done.

The three halves of a UI flow now meet in one place:

1. **Record** — every Arken session already records its NAVIGATE/CLICK/FILL
   events (the EventRecorder rides along); a human or script performs the
   flow once.
2. **Map** — ``flow.FlowGraph`` turns the recorded stream into a step
   graph (the model half, for humans and viewers).
3. **Complete** — this module: hand avion the recorded events and a
   browser it OWNS (``launch_chromium`` + a persistent profile) and the
   flow is re-driven step by step to its end, with a per-step report and
   the real final URL.

The seam that makes it one line: ``Arken.start(backend=...)`` accepts a
pre-built backend, so the session runs on OUR CDP connection to OUR
browser — the same browser whose profile already holds the login, which
is why a recorded flow can start mid-session (e.g. straight at the
dashboard) and still complete.

Lifecycle separation: completing a flow opens and closes the CDP
*connection* (session.stop()); the browser *process* belongs to
``Launched`` — stop it when the whole workday is done, and the profile
makes the next launch pick up exactly where this one left off.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

from avion.core.session import Arken
from avion.events.models import Event
from avion.events.replay import EventReplay, ReplayConfig, ReplayResult

if TYPE_CHECKING:
    from avion.launch import Launched


@dataclass
class FlowOutcome:
    """What completing the flow produced — the completion report."""

    result: ReplayResult
    final_url: str = ""
    steps_total: int = 0

    @property
    def completed(self) -> bool:
        """Every replayed step succeeded — the flow ran to its end."""
        return self.result.success

    @property
    def steps_done(self) -> int:
        return self.result.passed

    @property
    def failed_steps(self) -> list[dict[str, Any]]:
        return [s.to_dict() for s in self.result.steps if not s.success]

    def to_dict(self) -> dict[str, Any]:
        return {
            "completed": self.completed,
            "final_url": self.final_url,
            "steps_done": self.steps_done,
            "steps_total": self.steps_total,
            "failed_steps": self.failed_steps,
            "steps": [s.to_dict() for s in self.result.steps],
        }


async def complete_flow(
    events: Sequence[Event],
    launched: Launched | Any,
    *,
    base_url: str = "",
    config: ReplayConfig | None = None,
) -> FlowOutcome:
    """Complete a recorded flow on a browser avion owns (or is attached to).

    ``launched`` is a :class:`avion.launch.Launched` (avion owns the
    process) or a bare backend already connected to some browser (the
    CLI's attach model, ``avion run --cdp``) — whichever carries the
    connection is used. Every recorded NAVIGATE/CLICK/FILL is re-driven
    through a fresh Arken session; session markers and other noise are
    skipped by replay itself (they count toward ``steps_total`` as
    replayed steps only); recorded ``ELEMENT_FOUND`` evidence replays as
    barriers — the settle points that keep a form submit from racing the
    next navigation. ``config`` carries the policy: ``stop_on_error``
    (replay's default: stop at the first failed step) and ``dry_run``
    (resolve every locator, touch nothing — a dress rehearsal).

    The final URL is read from the live page (``get_url``), so it reflects
    click-driven navigations a form submission caused — not merely the
    last explicit goto.
    """
    backend = getattr(launched, "backend", launched)  # Launched or bare backend
    session = Arken(base_url=base_url)
    await session.start(backend=backend)
    try:
        result = await EventReplay(session, config).replay(list(events))
        try:
            final_url = str(await backend.get_url() or "")
        except Exception:  # noqa: BLE001 — report the flow, never mask it with a read error
            final_url = ""
        return FlowOutcome(
            result=result,
            final_url=final_url,
            steps_total=len(result.steps),
        )
    finally:
        await session.stop()
