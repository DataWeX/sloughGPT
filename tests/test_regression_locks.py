"""Regression-lock registry (card cf308a33) — pin clobber-prone fixes as assertions.

PROBLEM
-------
A "converge X to main — was N lines stale" style commit (single-parent, NOT a
merge) wholesale-overwrites a tree from a stale snapshot and silently reverts
fixes that landed in the interim. The fixes that SURVIVE are exactly the ones
WITH a locking test; the one that got re-clobbered (avion logger) had none, so
nothing went red in CI.

A fourth instance, same class: the unguarded ``while True`` in ``_cmd_logs -f``
(card 5311698a, guard 64c8c73f6) froze chunk C3 of the full suite at 41 % —
the fix lives *in a test file*, and a stale-base overwrite of that file reverts
the guard along with the code it protects.

THE GUARD
---------
The test file IS the guard: it lives in a path the converging branch typically
does not touch, so it survives and fails loudly. These are deliberately
IMPORT-FREE, SOURCE-LEVEL checks (avion is not importable from the root pytest
env) — they read the source text, matching the idiom already established by
``tests/cli/test_logs_unification.py`` and ``tests/test_embedding_config.py``.

WHY THE REGISTRY SITS HERE, CENTRALLY
-------------------------------------
The locks guard source subtrees that convergence commits rewrite.  A lock
placed *inside* the guarded subtree is reverted by the same overwrite that
reverts the code — that is exactly why the avion fix had no defense: any
protection living under ``packages/avion/`` would have travelled with the
stale snapshot.  Keeping the registry under ``tests/`` — outside every
subtree it names — means the overwrite takes the source and leaves the
witness behind.  (``tests/test_timestamp_guard.py`` is the established
precedent for this shape.)

ADDING A LOCK
-------------
Each registry row cites the card id + fix sha + the clobbering sha so the
provenance is self-documenting. Adding a lock is ONE ROW, not one file.
Probe the *exact* fixed form: loose probes are defeated by neighbours — a
sweep that deletes ``api_key: str = ""`` leaves ``wandb_api_key`` and the
field's own docstring behind, so a bare ``"api_key"`` probe stays green on
the very deletion it claims to catch.  If a lock trips because the change is
intentional, update that row in the SAME commit that makes the change and
say why; never delete a row to go green.
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
        # kwargs to the StructuredLogger, never splatted into a format string.
        ['self._logger.info("Arken session started", backend=self._backend.name)'],
        # BOTH historical bad forms interpolate into the message — the f-string
        # that 074475d26 actually restored, and the positional %s that
        # ef58a0c9c wrote first (a latent TypeError against info(message,
        # **fields)). One substring matches either. Verified: neither of the
        # original three probes ('"...", %s', '...", %', '% backend') matches
        # the real f-string clobber form, so they were dead code.
        ["Arken session started (backend="],
        "avion structured session log — fix 443ad24e0 (re-landing b94798cbc), "
        "clobbered by 074475d26, originally regressed by ef58a0c9c (card cf308a33)",
    ),
    (
        "embedding-config-openai-alias",
        "domain/infrastructure/_internal/config.py",
        # The fields a 81616a6f7 sweep deleted and 61831dc4d restored — probed
        # as full declarations. Bare "api_key"/"base_url" probes are defeated
        # by wandb_api_key, api_keys_enabled and the field's own docstring,
        # which all survive the sweep (proven: the loose probes stay green on
        # the exact deletion they claim to catch).
        ['api_key: str = ""', 'base_url: str = ""', "def _sync_openai_alias"],
        [],
        "EmbeddingConfig fields — fix 61831dc4d, deleted by 81616a6f7 "
        "(card ff9ef505; deep lock tests/test_embedding_config.py, 11 tests)",
    ),
    (
        "train-dashboard-single-path",
        "apps/cli/src/groups/train.py",
        # monitor.py was deleted but the import left behind caused a live
        # ModuleNotFoundError; train must route through logs.run_dashboard.
        # The pass_context anchor is load-bearing: train_monitor reads ctx.obj,
        # and the stranded 6bb3a97ea dropped the decorator while keeping the
        # body — a TypeError the moment the command runs.
        [
            "from commands.logs import run_dashboard",
            "run_dashboard(",
            "@click.pass_context\n    def train_monitor(",
        ],
        ["from commands import monitor", "commands.monitor", "import monitor"],
        "train -> run_dashboard — fix cc7a5e3ee, import left by 0a29d5c6e (card 8182563a)",
    ),
    (
        "cli-logs-shared-dashboard-helper",
        "apps/cli/src/commands/logs.py",
        # run_dashboard() is the single implementation both 'logs --dashboard'
        # and 'train monitor' call; the fold that created it is cc7a5e3ee.
        ["def run_dashboard("],
        [],
        "logs run_dashboard — fix cc7a5e3ee (card 8182563a)",
    ),
    (
        "shell-logs-follow-hang-guard",
        "tests/core-py/test_shell_repl_more.py",
        # The citation comment ships inside the guarded test function, so it
        # vanishes with it under a stale-base overwrite of this test file.
        ["card 5311698a: killed chunk C3", "signal.alarm"],
        [],
        "logs-follow SIGALRM guard — fix 64c8c73f6 (card 5311698a)",
    ),
]


def test_registry_is_non_empty():
    assert REGISTRY, "Regression-lock registry must not be empty — it is the guard"


def test_registry_rows_are_well_formed():
    seen: set[str] = set()
    for row in REGISTRY:
        lock_id, rel, must_contain, must_not, cite = row
        assert lock_id and rel and cite, f"Malformed registry row: {row!r}"
        assert lock_id not in seen, f"duplicate lock id {lock_id!r} — rows must be unique"
        seen.add(lock_id)
        assert must_contain or must_not, (
            f"Lock '{lock_id}' pins nothing — a row with no patterns always passes"
        )


def test_avion_structured_session_log_locked():
    """Lock the fix that got clobbered silently (443ad24e0 -> 074475d26)."""
    lock_id, rel, must_contain, must_not, cite = REGISTRY[0]
    text = (ROOT / rel).read_text()
    for needle in must_contain:
        assert needle in text, f"[{lock_id}] {cite}: expected fixed form {needle!r}"
    for needle in must_not:
        assert needle not in text, (
            f"[{lock_id}] {cite}: regressed interpolating form {needle!r} reappeared"
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
