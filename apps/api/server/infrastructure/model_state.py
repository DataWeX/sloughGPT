"""Explicit model lifecycle state for the health surfaces.

Card 20261104_053224: ``/health`` inferred "loading" from an uptime window
(``uptime < 90 s``), which cannot distinguish a server whose weights are
landing on the preload thread from one that will never load anything — and
it let a registry-only process report ready with no serving handle.

Written by the lazy preload path (``_try_lazy_guard_autoload`` →
``_start_parent_preload`` → ``_preload_parent_weights``), read by the health
controller:

- ``absent``        — nothing installed, nothing scheduled (default)
- ``materialising`` — a provider is installed; weights are landing
- ``loaded``        — weights are resident (published to ``server_state``)
- ``degraded``      — the scheduled preload failed; materialise on first request

A plain module value behind a lock, deliberately not a ``ServerState`` field:
it records *this process's boot path*, and both writers (startup) and readers
(health) must be able to touch it without depending on singleton init order.
"""

from __future__ import annotations

import threading

ABSENT = "absent"
MATERIALIZING = "materialising"
LOADED = "loaded"
DEGRADED = "degraded"

_VALID = frozenset({ABSENT, MATERIALIZING, LOADED, DEGRADED})

_lock = threading.Lock()
_state = ABSENT


def set_model_state(state: str) -> None:
    """Record the lifecycle state.

    Raises:
        ValueError: on an unknown state — a typo in a writer must fail loudly
            in tests, not silently produce a state health can never render.
    """
    global _state
    if state not in _VALID:
        raise ValueError(f"unknown model state: {state!r}")
    with _lock:
        _state = state


def get_model_state() -> str:
    """Current lifecycle state (one of the four constants above)."""
    with _lock:
        return _state


def reset_model_state() -> None:
    """Return to ``ABSENT`` — a fresh boot, and the test cleanup hook."""
    set_model_state(ABSENT)
