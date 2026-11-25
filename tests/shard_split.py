"""Owned CI test sharding — file-grouped split of the collected test list.

Card 20261107_112103 (ci(xdist)): the test-core-py suite is the PR queue's
long pole; `-n auto` inside ONE job tops out at the runner's 2 CPUs (measured
1.45x vs serial), so the real lever is running N shards on N runners —
public repo, free minutes. This module is that split, owned rather than
imported: the repo's build-from-scratch rule (AGENTS.md, execution layer)
applies to CI plumbing too, and a hand-rolled file-grouped split is ~20
lines with exactly one behaviour, versus a floating unpinned plugin in CI's
install line.

Contract
--------
``SLO_TEST_SHARD="k/n"`` (1-based, e.g. ``2/4``) keeps every file's tests
in ONE shard: collected items are grouped by file (nodeid prefix) and the
files are LPT bin-packed across the n shards (largest first, into the
currently smallest shard, ties broken by name). Shards are therefore
**disjoint and complete** — their union is the full list — and a file
never straddles two shards.

Why file groups (run 45361359978, 2026-11-24): a plain round-robin splits
adjacent tests of one file across runners, which deterministically broke
every test pair sharing state — a module-global handoff
(``test_infra_thread_teardown::test_a/b``) and a disk chain
(``test_consciousness_router::TestPersonas`` save → load → delete). A
pytest file is the unit of shared state (module globals, ordered
scenarios), so the split respects that boundary. LPT keeps the balance
round-robin had (measured 10,472 x 4 exact over 41,888 items) and the
name tie-break makes placement independent of filesystem collection order
— a local slice run predicts CI placement exactly.

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


def _file_key(item) -> str:
    """The file an item belongs to: its nodeid before the first ``::``."""
    return item.nodeid.split("::", 1)[0]


def apply(items: list, spec: str) -> list:
    """Return this shard's slice of *items* (file-grouped LPT; disjoint, complete)."""
    group, total = parse(spec)
    files: dict[str, list] = {}
    for item in items:
        files.setdefault(_file_key(item), []).append(item)
    shards: list[list] = [[] for _ in range(total)]
    sizes = [0] * total
    # Largest file first, ties by name; each file lands in the currently
    # smallest shard (ties by index). Deterministic AND independent of the
    # order the filesystem handed us the files.
    for name in sorted(files, key=lambda n: (-len(files[n]), n)):
        k = min(range(total), key=lambda i: (sizes[i], i))
        shards[k].extend(files[name])
        sizes[k] += len(files[name])
    return shards[group - 1]


def enabled() -> str | None:
    """The raw spec if set (empty string counts as unset), else None."""
    return os.environ.get("SLO_TEST_SHARD") or None
