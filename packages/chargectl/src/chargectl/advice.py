"""advice — pure charging advice derived from a :class:`~chargectl.status.ChargeStatus`.

The 80% longevity rule: lithium cells age fastest held near 100% and near 0%. Keep the
usable band at 20–80% and the pack keeps more of its cycle life.
"""

from __future__ import annotations

from typing import Any

from .status import ChargeStatus

OPTIMAL_LIMIT = 80
LOW_THRESHOLD = 20
HIGH_THRESHOLD = 90


def optimize_hint(status: ChargeStatus, optimal_limit: int = OPTIMAL_LIMIT) -> dict[str, Any]:
    """Return ``{"limit", "action", "reason"}`` for the given status.

    ``action`` is one of ``unplug`` | ``cap_at_80`` | ``plug_in`` | ``maintain``.
    """
    level = status.level
    charging = status.is_charging

    if level >= HIGH_THRESHOLD and charging:
        return {
            "limit": optimal_limit,
            "action": "unplug",
            "reason": (
                f"Battery at {level}% — unplug to preserve longevity ({optimal_limit}% optimal)."
            ),
        }
    if level >= optimal_limit and charging:
        return {
            "limit": optimal_limit,
            "action": "cap_at_80",
            "reason": (
                f"At {level}% and still charging — cap long-term charge to {optimal_limit}%."
            ),
        }
    if level < LOW_THRESHOLD and not charging:
        return {
            "limit": 100,
            "action": "plug_in",
            "reason": f"Battery low ({level}%) — plug in to avoid deep discharge.",
        }
    return {
        "limit": optimal_limit,
        "action": "maintain",
        "reason": (
            f"Battery {level}% — {'charging' if charging else 'on battery'} "
            f"(optimal range {LOW_THRESHOLD}–{optimal_limit}%)."
        ),
    }
