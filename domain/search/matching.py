"""Shared matching helper — one substring rule for stores without a
specialized index.

Returns a relevance score in ``(0, 1]`` or ``None`` for no match:

- title match        -> 1.0
- detail-only match  -> 0.6

Stores with real indexes (vector, FTS) are free to ignore this and
score on their own — the aggregator only requires ordering to be
meaningful.
"""

from __future__ import annotations


def substring_score(q: str, title: str, detail: str = "") -> float | None:
    needle = q.casefold()
    if not needle:
        return None
    if needle in (title or "").casefold():
        return 1.0
    if needle in (detail or "").casefold():
        return 0.6
    return None
