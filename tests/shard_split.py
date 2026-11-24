"""Owned CI test sharding — round-robin split of the collected test list.

Card 20261107_112103 (ci(xdist)): the test-core-py suite is the PR queue's
long pole; `-n auto` inside ONE job tops out at the runner's 2 CPUs (measured
1.45x vs serial), so the real lever is running N shards on N runners —
public repo, free minutes. This module is that split, owned rather than
imported: the repo's build-from-scratch rule (AGENTS.md, execution layer)
applies to CI plumbing too, and a hand-rolled round-robin is ~20 lines with
exactly one behaviour, versus a floating unpinned plugin in CI's install line.

Contract
--------
``SLO_TEST_SHARD="k/n"`` (1-based, e.g. ``2/4``) keeps every n-th collected
test starting at offset k-1. Shards are therefore **disjoint and complete**:
their union is the full list, in collection order — deterministic because CI
installs no random-order plugin. Round-robin (not contiguous chunks) spreads
sibling test files across shards, so one pathological file cannot strand a
quarter of the suite on one runner.

The variable is read at collection time (see ``tests/conftest.py`` hook).
**Unset = do nothing** — local runs, testmon closures and every non-CI suite
collect exactly as before. Invalid specs raise ``ValueError`` and the hook
turns it into a pytest usage error: a misconfigured CI must fail loud and
instant, never silently run the whole suite in one shard.
"""

from __future__ import annotations

import os
import re

_SPEC = re.compile(r"^\s*(\d+)\s*/\s*(\d+)\s*$")


def parse(spec: str) -> tuple[int, int]:
    """Parse ``SLO_TEST_SHARD`` into ``(group, total)``; 1-based group.

    Raises ValueError on anything but ``k/n`` with ``1 <= k <= n``.
    """
    m = _SPEC.match(spec or "")
    if not m:
        raise ValueError(f"SLO_TEST_SHARD={spec!r} must look like 'k/n' (e.g. '2/4')")
    group, total = int(m.group(1)), int(m.group(2))
    if total < 1:
        raise ValueError(f"SLO_TEST_SHARD={spec!r}: total n must be >= 1")
    if group < 1 or group > total:
        raise ValueError(f"SLO_TEST_SHARD={spec!r}: group k must satisfy 1 <= k <= n")
    return group, total


def apply(items: list, spec: str) -> list:
    """Return this shard's slice of *items* (round-robin, disjoint, complete)."""
    group, total = parse(spec)
    return items[group - 1 :: total]


def enabled() -> str | None:
    """The raw spec if set (empty string counts as unset), else None."""
    return os.environ.get("SLO_TEST_SHARD") or None
