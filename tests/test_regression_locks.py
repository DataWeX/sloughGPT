"""Regression-lock registry (card cf308a33) — pin clobber-prone fixes as assertions.

PROBLEM
-------
A "converge X to main — was N lines stale" style commit (single-parent, NOT a
merge) wholesale-overwrites a tree from a stale snapshot and silently reverts
fixes that landed in the interim. The fixes that SURVIVE are exactly the ones
WITH a locking test; the one that got re-clobbered (avion logger) had none, so
nothing went red in CI.

THE GUARD
---------
The test file IS the guard: it lives in a path the converging branch typically
does not touch, so it survives and fails loudly. These are deliberately
IMPORT-FREE, SOURCE-LEVEL checks (avion is not importable from the root pytest
env) — they read the source text, matching the idiom already established by
``tests/cli/test_logs_unification.py`` and ``tests/test_embedding_config.py``.

Each registry row cites the card id + fix sha + the clobbering sha so the
provenance is self-documenting. Adding a lock is ONE ROW, not one file.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ── registry ─────────────────────────────────────────────────────────────────
# (lock_id, relative path, must_contain, must_not_contain, cite)
REGISTRY: list[tuple[str, str, list[str], list[str], str]] = [
    (
        "avion-structured-session-log",
        "packages/avion/src/avion/core/session.py",
        # The structured-field form restored by 443ad24e0: fields passed as
        # kwargs to the StructuredLogger, never splatted into a %s format string.
        ['self._logger.info("Arken session started", backend=self._backend.name)'],
        # The clobbered (positional %s) form that 074475d26 reintroduced.
        ['"Arken session started", "%s"', '"Arken session started", %', "% backend"],
        "avion structured session log — fix 443ad24e0, clobbered by 074475d26 (card cf308a33)",
    ),
    (
        "embedding-config-openai-alias",
        "domain/infrastructure/_internal/config.py",
        # The fields a 81616a6f7 sweep deleted and 61831dc4d restored.
        ["api_key", "base_url"],
        [],
        "EmbeddingConfig fields — fix 61831dc4d, deleted by 81616a6f7 (card ff9ef505)",
    ),
    (
        "train-dashboard-single-path",
        "apps/cli/src/groups/train.py",
        # monitor.py was deleted but the import left behind caused a live
        # ModuleNotFoundError; train must route through logs.run_dashboard.
        ["from commands.logs import run_dashboard", "run_dashboard("],
        ["from commands import monitor", "commands.monitor", "import monitor"],
        "train -> run_dashboard — fix cc7a5e3ee, import left by 0a29d5c6e (card 8182563a)",
    ),
]


def test_registry_is_non_empty():
    assert REGISTRY, "Regression-lock registry must not be empty — it is the guard"


def test_registry_rows_are_well_formed():
    for row in REGISTRY:
        lock_id, rel, must_contain, _must_not, cite = row
        assert lock_id and rel and cite, f"Malformed registry row: {row!r}"
        assert must_contain, (
            f"Lock '{lock_id}' must assert at least the fixed form is present, "
            "otherwise it cannot detect a clobber"
        )


def test_avion_structured_session_log_locked():
    """Lock the fix that got clobbered silently (443ad24e0 -> 074475d26)."""
    lock_id, rel, must_contain, must_not, cite = REGISTRY[0]
    text = (ROOT / rel).read_text()
    for needle in must_contain:
        assert needle in text, f"[{lock_id}] {cite}: expected fixed form {needle!r}"
    for needle in must_not:
        assert needle not in text, (
            f"[{lock_id}] {cite}: clobbered positional-%s form {needle!r} reappeared"
        )


def test_every_registry_entry_passes():
    """Drive the whole table so a future sweep trips the central registry even
    if it DELETES the individual locking test file."""
    failures: list[str] = []
    for lock_id, rel, must_contain, must_not, cite in REGISTRY:
        path = ROOT / rel
        if not path.exists():
            failures.append(f"[{lock_id}] {cite}: source {rel} is missing")
            continue
        text = path.read_text()
        for needle in must_contain:
            if needle not in text:
                failures.append(f"[{lock_id}] {cite}: missing fixed form {needle!r} in {rel}")
        for needle in must_not:
            if needle in text:
                failures.append(f"[{lock_id}] {cite}: clobbered form {needle!r} present in {rel}")
    assert not failures, "Regression locks tripped:\n" + "\n".join(failures)
