"""record_experience — the ONE substrate-recording entrypoint for
execution paths outside chat (agent runs, journey executions).

Chat records through its own post-gen pipeline (the inference router's
``_run_post_gen_tasks``); every other runner funnels through this single
function so the substrate has exactly one recording seam, not a copy per
runner. Callers pass the run's goal/prompt and its outcome; recording is
best-effort by design — a substrate failure must never take down the run
that produced the experience (mirrors chat's ``_consciousness_post_process``
swallow-and-log contract).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Cap the response text handed to consciousness: runs can stream megabytes
# of tool output through `response`; the episode only needs the outcome.
_MAX_EXPERIENCE_CHARS = 2000


def record_experience(
    prompt: str, response: str, *, source: str = "agent"
) -> dict[str, Any] | None:
    """Record one agent/journey experience as a consciousness episode.

    Returns the consciousness result, or ``None`` when consciousness is
    unavailable or recording fails. Never raises.
    """
    try:
        from domain.core import get_consciousness

        engine = get_consciousness()
        return engine.process(
            (prompt or "")[:_MAX_EXPERIENCE_CHARS],
            (response or "")[:_MAX_EXPERIENCE_CHARS],
        )
    except Exception as exc:  # noqa: BLE001 — recording is best-effort by contract
        logger.debug("record_experience(source=%s) failed: %s", source, exc)
        return None
