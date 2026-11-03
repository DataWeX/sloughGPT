"""Process-notification contract — the projector over training process state.

Card 20261102_014254. A notification is a projection of a process event
carrying the action-space the process currently accepts — not a string
with buttons glued on. Action validity is a function of process state, so
the surface is resolved from the CURRENT state at projection time and the
server re-validates any dispatched action fail-closed (see
``training/router.py`` recover endpoint).

Boot-time events cannot ride the SSE streams (no client is connected
yet): they derive from a REST snapshot of the durable JobStore
(``boot_snapshot``), and live transitions ride the existing per-request
SSE streams. Both paths project through this module, so the event_id —
the dedupe key — is stable across snapshot and stream for the same
underlying process event.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("slo")

CONTRACT_VERSION = 1
PROJECTOR_KIND = "training"

# ── Vocabularies (card 20261102_110421) — the closed sets a projected
# event may draw from. The TS projection (contracts.gen.ts) and the
# dock's exhaustive tone map key off these; a new value must be declared
# here first and the parity tests re-run against it.
ACTION_INTENTS = ("recover", "cancel", "navigate", "dismiss")
# Values the `state` field carries (job states that notify).
STATES = ("interrupted", "failed", "running")

# The process's real command surface, per state. Entries are structured
# (id/label/intent) so consumers render actions instead of plain text and
# dispatch goes through the normal controller layer. An action listed here
# exists in the system — dead buttons are never projected.
_ACTIONS_BY_STATE: dict[str, list[dict[str, str]]] = {
    "interrupted": [
        {"id": "recover_now", "label": "Recover now", "intent": "recover"},
        {"id": "view_logs", "label": "View logs", "intent": "navigate"},
        {"id": "dismiss", "label": "Dismiss", "intent": "dismiss"},
    ],
    "failed": [
        {"id": "recover_now", "label": "Recover now", "intent": "recover"},
        {"id": "view_logs", "label": "View logs", "intent": "navigate"},
        {"id": "dismiss", "label": "Dismiss", "intent": "dismiss"},
    ],
    "running": [
        {"id": "cancel", "label": "Cancel", "intent": "cancel"},
        {"id": "view_logs", "label": "View logs", "intent": "navigate"},
    ],
    # An auto-resume the policy declined: automation is legible, and the
    # manual fallback (recover) still exists — the human can act.
    "skipped": [
        {"id": "recover_now", "label": "Recover now", "intent": "recover"},
        {"id": "dismiss", "label": "Dismiss", "intent": "dismiss"},
    ],
}

_TITLES: dict[str, str] = {
    "interrupted": "Training interrupted",
    "failed": "Training failed",
    "running": "Training running",
    "auto_train_started": "Training auto-resumed",
    "auto_resume_skipped": "Auto-resume skipped",
}

# Derived so the table cannot drift from the vocabulary it feeds: the
# `event` field's values are exactly the titled events.
EVENTS: tuple[str, ...] = tuple(_TITLES)

# ── The one declaration of the envelope (card 20261102_110421) ────────
# field -> (ts type, required). This is THE source for the event shape:
# scripts/gen_ts_contracts.py projects it into contracts.gen.ts
# (ProcessAction/ProcessEvent, all three frontend consumers) so the TS
# half is generated, never hand-written; make_event rejects anything not
# declared here at the single construction point. TS types may reference
# the emitted aliases (ProcessAction[], ProcessState, ProcessEventName)
# or the token PROJECTOR_KIND (resolved to its literal at generation).
EVENT_SCHEMA: dict[str, tuple[str, bool]] = {
    "event_id": ("string", True),
    "kind": ("PROJECTOR_KIND", True),
    "process_id": ("string", True),
    "event": ("ProcessEventName", True),
    "state": ("ProcessState", True),
    "actions": ("ProcessAction[]", True),
    "process_link": ("string", True),
    "title": ("string", True),
    "contract_version": ("number", True),
    "name": ("string", False),
    "reason": ("string", False),
}

_REQUIRED_FIELDS = frozenset(f for f, (_t, required) in EVENT_SCHEMA.items() if required)
_OPTIONAL_FIELDS = frozenset(f for f, (_t, required) in EVENT_SCHEMA.items() if not required)

# The action sub-object's declaration — same discipline as EVENT_SCHEMA:
# scripts/gen_ts_contracts.py renders ProcessAction from this, and the
# parity tests assert every table action matches it exactly.
ACTION_SCHEMA: dict[str, tuple[str, bool]] = {
    "id": ("string", True),
    "label": ("string", True),
    "intent": ("ProcessActionIntent", True),
}


def action_surface(state: str) -> list[dict[str, str]]:
    """The structured command surface for a process state (fresh copies)."""
    return [dict(a) for a in _ACTIONS_BY_STATE.get(state, [])]


def make_event(
    process_id: str,
    event: str,
    state: str,
    *,
    surface: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Build a full contract event. The event_id is deterministic — the
    same process + event always maps to the same id, so the dock dedupes
    a REST snapshot against a later live transition without double-render.

    ``surface`` selects the action set (defaults to ``state``); an event
    may project a different surface than its state name when the
    notification is about a transition (e.g. auto_train_started carries
    the running surface).

    Contract (card 20261102_110421): the emitted keys are exactly
    EVENT_SCHEMA's required fields, extras must be declared optional
    fields, and event/state/surface must be in the declared vocabularies.
    A producer that invents anything raises HERE — the single construction
    point — instead of shipping a field the schema (and its generated TS
    half) has never heard of.
    """
    unknown = set(extra) - _OPTIONAL_FIELDS
    if unknown:
        raise ValueError(f"undeclared event fields: {sorted(unknown)} — add them to EVENT_SCHEMA")
    if state not in STATES:
        raise ValueError(f"unknown state: {state!r} — add it to STATES")
    if event not in EVENTS:
        raise ValueError(f"unknown event: {event!r} — add a title to _TITLES")
    effective_surface = surface or state
    if effective_surface not in _ACTIONS_BY_STATE:
        raise ValueError(
            f"unknown action surface: {effective_surface!r} — add it to _ACTIONS_BY_STATE"
        )
    return {
        "event_id": f"{PROJECTOR_KIND}:{process_id}:{event}",
        "kind": PROJECTOR_KIND,
        "process_id": process_id,
        "event": event,
        "state": state,
        "actions": action_surface(surface or state),
        "process_link": f"/training/job/{process_id}",
        "title": _TITLES.get(event, _TITLES.get(state, "Training update")),
        "contract_version": CONTRACT_VERSION,
        **extra,
    }


def project_job_event(row: dict[str, Any]) -> dict[str, Any] | None:
    """Project a durable JobStore row into a contract event.

    Returns None for states that are not notifications (completed,
    pending, purged) — the dock never sees settled jobs.
    """
    if not row:
        return None
    status = row.get("status") or ""
    if status not in ("interrupted", "failed", "running"):
        return None
    process_id = row.get("id") or ""
    if not process_id:
        return None
    return make_event(process_id, status, status, name=row.get("name") or "")


def action_ids(event: dict[str, Any]) -> list[str]:
    return [a["id"] for a in event.get("actions", [])]


def boot_snapshot(store: Any) -> list[dict[str, Any]]:
    """Project the durable recoverable set into contract events.

    This is the boot-time source: SSE cannot deliver events that happened
    while no client was connected, so the dock's initial state is a REST
    snapshot of durable state. Store failure degrades to an empty list —
    a notification outage must never take the boot path down.
    """
    try:
        rows = store.get_recoverable_jobs()
    except Exception as e:
        logger.warning("Process-event snapshot failed: %s", e, extra={"tag": "TRAIN"})
        return []
    events: list[dict[str, Any]] = []
    for row in rows:
        ev = project_job_event(row)
        if ev is not None:
            events.append(ev)
    return events
