"""Phase-split tests for scripts/run_affected.sh (card 20261111_125558 Part B).

The script's two phases are exercised end-to-end with SLO_PYTHON pointed at a
stub interpreter: the stub records every ``-m pytest`` invocation (argv +
phase env) and returns per-phase exit codes, so no real test run happens.
What is asserted is the CONTRACT:

* phase 1 (foreground) ignores every tracked slow path and its exit code is
  the script's exit code;
* phase 2 re-enters the script with RUN_AFFECTED_PHASE=slow over exactly
  those paths (partition by construction), in the background, and records
  its own EXIT= marker + status file;
* no slow list → today's behavior, byte-for-byte single invocation;
* an empty slow list at phase-2 entry must not collect pytest's testpaths.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_affected.sh"

# Real, always-existing repo files used as fake "slow paths" — the stub never
# runs pytest, so the paths only need to pass the script's existence check
# (no fixture files are created inside the checkout).
SLOW_A = "scripts/seed_testmon.py"
SLOW_B = "tests/test_run_affected_seed.py"


@pytest.fixture()
def stub_python(tmp_path: Path) -> Path:
    """A fake interpreter: logs each -m pytest invocation, exits per phase."""
    stub = tmp_path / "fake-python"
    stub.write_text(
        """#!/bin/sh
# argv logger for run_affected phase tests (SLO_PYTHON target)
if [ "${1:-}" = "-m" ]; then
  printf 'PHASE=%s ARGS=%s\\n' "${RUN_AFFECTED_PHASE:-phase1}" "$*" >> "$STUB_LOG"
  if [ "${RUN_AFFECTED_PHASE:-}" = "slow" ]; then
    exit "${STUB_RC_SLOW:-0}"
  fi
  exit "${STUB_RC:-0}"
fi
# seed-helper call (`python - <root>` with a heredoc): pretend success
exit 0
"""
    )
    stub.chmod(0o755)
    return stub


def _env(stub: Path, log: Path, **extra: str) -> dict[str, str]:
    env = {**os.environ, "SLO_PYTHON": str(stub), "STUB_LOG": str(log)}
    env.update(extra)
    # The script must see a slow list we control; RUN_ALL tests opt in via extra.
    return env


def _run(
    args: list[str], env: dict[str, str], slow_list: Path | None
) -> subprocess.CompletedProcess:
    if slow_list is not None:
        env["RUN_AFFECTED_SLOW_LIST"] = str(slow_list)
    return subprocess.run(
        [str(SCRIPT), *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO_ROOT,
        timeout=60,
    )


def _slow_list(tmp_path: Path, entries: list[str]) -> Path:
    p = tmp_path / "slow.txt"
    p.write_text("# comment line\n\n" + "\n".join(entries) + "\n")
    return p


def _argv(line: str) -> list[str]:
    """Split a stub log line (`PHASE=x ARGS=...`) into pytest's argv tokens."""
    return line.split("ARGS=", 1)[1].split()


def _wait_for_lines(log: Path, n: int, timeout: float = 15.0) -> list[str]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if log.exists():
            lines = log.read_text().splitlines()
            if len(lines) >= n:
                return lines
        time.sleep(0.05)
    raise AssertionError(
        f"expected {n} stub lines in {log}, got: {log.read_text() if log.exists() else '(no log)'}"
    )


def _cleanup_phases() -> None:
    for name in ("slow-phase.log", "slow-phase.status"):
        (REPO_ROOT / ".pytest_cache" / name).unlink(missing_ok=True)


def test_phase1_ignores_slow_paths_phase2_runs_them_exit_codes_split(
    stub_python: Path, tmp_path: Path
) -> None:
    """The partition + exit-code contract, end to end."""
    _cleanup_phases()
    log = tmp_path / "stub.log"
    sl = _slow_list(tmp_path, [SLOW_A, SLOW_B])
    env = _env(stub_python, log, STUB_RC="3", STUB_RC_SLOW="9")

    try:
        res = _run([], env, sl)
        # The script's exit code is PHASE 1's, never phase 2's (9).
        assert res.returncode == 3, res.stderr

        lines = _wait_for_lines(log, 2)
        phase1, phase2 = lines[0], lines[1]

        # Phase 1: default positional `tests`, each slow path ignored, not positional.
        assert phase1.startswith("PHASE=phase1 ")
        assert " tests " in phase1 or phase1.rstrip().endswith(" tests")
        assert f"--ignore={SLOW_A}" in phase1
        assert f"--ignore={SLOW_B}" in phase1
        assert "--testmon-forceselect" in phase1
        assert f" {SLOW_A}" not in phase1.replace(f"--ignore={SLOW_A}", "")

        # Phase 2: slow paths positional, NO `tests`, same selection mode.
        assert phase2.startswith("PHASE=slow ")
        assert f" {SLOW_A}" in phase2 and f" {SLOW_B}" in phase2
        assert "--ignore=" not in phase2  # the complement, not the exclusion
        assert " tests" not in phase2.replace(f" {SLOW_A}", "").replace(f" {SLOW_B}", "")
        assert "--testmon-forceselect" in phase2

        # Phase 2 recorded its own result: EXIT=9 in log + status file.
        deadline = time.monotonic() + 15
        status = REPO_ROOT / ".pytest_cache" / "slow-phase.status"
        while time.monotonic() < deadline and (
            not status.exists() or status.read_text().strip() != "EXIT=9"
        ):
            time.sleep(0.05)
        assert status.read_text().strip() == "EXIT=9"
        slow_log = (REPO_ROOT / ".pytest_cache" / "slow-phase.log").read_text()
        assert "=== slow phase started" in slow_log
        assert "EXIT=9" in slow_log
    finally:
        _cleanup_phases()


def test_missing_slow_list_keeps_single_invocation(stub_python: Path, tmp_path: Path) -> None:
    """No list → no --ignore, no phase 2: today's behavior exactly."""
    _cleanup_phases()
    log = tmp_path / "stub.log"
    env = _env(stub_python, log)

    res = _run([], env, tmp_path / "does-not-exist.txt")

    assert res.returncode == 0
    lines = _wait_for_lines(log, 1)
    assert len(lines) == 1, f"expected exactly one invocation, got {lines}"
    assert "--ignore=" not in lines[0]
    assert "running in background" not in res.stdout
    assert not (REPO_ROOT / ".pytest_cache" / "slow-phase.status").exists()


def test_user_flags_pass_through_to_both_phases(stub_python: Path, tmp_path: Path) -> None:
    """Flags-only invocations partition exactly: user flags ride both phases."""
    _cleanup_phases()
    log = tmp_path / "stub.log"
    sl = _slow_list(tmp_path, [SLOW_A])
    env = _env(stub_python, log)

    try:
        res = _run(["--ignore=tests/server/test_phoneme_router.py"], env, sl)
        assert res.returncode == 0

        lines = _wait_for_lines(log, 2)
        assert "--ignore=tests/server/test_phoneme_router.py" in lines[0]
        assert "--ignore=tests/server/test_phoneme_router.py" in lines[1]
        assert lines[1].startswith("PHASE=slow ")
        # Default `tests` positional only appears when no args were given.
        assert " tests " not in lines[1]
    finally:
        _cleanup_phases()


def test_run_all_splits_with_noselect(stub_python: Path, tmp_path: Path) -> None:
    """RUN_ALL=1 also splits: both phases noseleect; union = full set."""
    _cleanup_phases()
    log = tmp_path / "stub.log"
    sl = _slow_list(tmp_path, [SLOW_A])
    env = _env(stub_python, log, RUN_ALL="1")

    try:
        res = _run([], env, sl)
        assert res.returncode == 0
        lines = _wait_for_lines(log, 2)
        assert "--testmon-noselect" in lines[0]
        assert "--testmon-noselect" in lines[1]
        assert lines[1].startswith("PHASE=slow ")
        assert f" {SLOW_A}" in lines[1]
    finally:
        _cleanup_phases()


def test_phase2_entry_with_empty_list_runs_nothing(stub_python: Path, tmp_path: Path) -> None:
    """An empty list at phase-2 entry must not collect pytest's testpaths."""
    log = tmp_path / "stub.log"
    env = _env(stub_python, log, RUN_AFFECTED_PHASE="slow")
    env["RUN_AFFECTED_SLOW_LIST"] = str(tmp_path / "empty.txt")
    (tmp_path / "empty.txt").write_text("# only comments\n")

    res = subprocess.run(
        [str(SCRIPT)], capture_output=True, text=True, env=env, cwd=REPO_ROOT, timeout=60
    )

    assert res.returncode == 0
    assert "slow list is empty" in res.stderr
    assert not log.exists(), "pytest must not be invoked at all"


def test_positional_scope_partitions_exactly(stub_python: Path, tmp_path: Path) -> None:
    """A scoped invocation must not re-collect its fast closure in phase 2.

    ``run_affected.sh tests/core-py`` (the documented scope usage) used to pass
    ``tests/core-py`` through to phase 2 as well, so the background run
    re-ran every affected fast test of the scope AND the slow paths outside
    it. Phase 2's positionals are exactly the in-scope slow paths, and the
    scope argument itself never reaches phase 2.
    """
    _cleanup_phases()
    log = tmp_path / "stub.log"
    sl = _slow_list(
        tmp_path,
        ["tests/core-py/test_vm_life_program.py", "tests/test_journeys_vertical.py"],
    )
    env = _env(stub_python, log)

    try:
        res = _run(["tests/core-py"], env, sl)
        assert res.returncode == 0, res.stderr

        lines = _wait_for_lines(log, 2)
        a1, a2 = _argv(lines[0]), _argv(lines[1])

        # Phase 1 keeps the scope argument and defers the in-scope slow path.
        assert "tests/core-py" in a1
        assert "--ignore=tests/core-py/test_vm_life_program.py" in a1
        # Out-of-scope slow paths are neither collected nor deferred: the scope
        # argument does not reach them, so ignoring them would be a no-op and
        # running them in phase 2 would leave the user's scope.
        assert "--ignore=tests/test_journeys_vertical.py" not in a1
        assert "--testmon-forceselect" in a1

        # Phase 2: scope argument stripped, ONLY the in-scope slow path.
        assert "tests/core-py" not in a2
        assert "tests/core-py/test_vm_life_program.py" in a2
        assert "tests/test_journeys_vertical.py" not in a2
        assert not [t for t in a2 if t.startswith("--ignore=")]
        assert "--testmon-forceselect" in a2
    finally:
        _cleanup_phases()


def test_explicit_slow_path_arg_runs_exactly_once(stub_python: Path, tmp_path: Path) -> None:
    """A slow path passed explicitly runs in phase 1 — never a second time.

    ``--ignore`` does not apply to an explicitly-passed file (measured:
    pytest still collects it), so ignoring it AND deferring it would run it
    twice, while deferring it alone could drop it. Explicit args are
    therefore excluded from both sides of the partition.
    """
    _cleanup_phases()
    log = tmp_path / "stub.log"
    sl = _slow_list(tmp_path, [SLOW_A, SLOW_B])
    env = _env(stub_python, log)

    try:
        res = _run([SLOW_B], env, sl)
        assert res.returncode == 0, res.stderr

        lines = _wait_for_lines(log, 1)
        assert len(lines) == 1, f"expected a single phase, got: {lines}"
        a1 = _argv(lines[0])
        assert SLOW_B in a1  # the explicit arg runs in phase 1
        assert f"--ignore={SLOW_B}" not in a1  # never ignored (would be a no-op/dup)
        # Out of scope: the scope is the explicit file, so the other tracked
        # path is collected by neither phase — ignoring it would be a no-op.
        assert f"--ignore={SLOW_A}" not in a1
        assert SLOW_A not in a1
    finally:
        _cleanup_phases()


def test_slow_paths_txt_entries_exist_and_stay_in_testpaths() -> None:
    """A dead entry defers nothing — the shipped list must name real files.

    The list shipped with 4 of 6 entries pointing at ``tests/`` while the
    files live in ``tests/core-py/``; the script's missing-entry warning
    skipped them, so the slow tail silently ran in the fast phase. Entries
    must also live under pytest.ini's testpaths, or the default (no-args)
    run would collect them in neither phase.
    """
    entries = [
        line.strip()
        for line in (REPO_ROOT / "scripts" / "slow_paths.txt").read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert entries, "the shipped slow list must not be empty"
    missing = [e for e in entries if not (REPO_ROOT / e).exists()]
    assert not missing, f"slow_paths.txt entries do not exist: {missing}"
    testpaths = ("tests/", "apps/cli/tests/", "packages/grib/tests/", "packages/planner/tests/")
    outside = [e for e in entries if not e.startswith(testpaths)]
    assert not outside, f"slow_paths.txt entries are outside pytest testpaths: {outside}"


def test_stale_slow_phase_failure_warns_on_next_run(stub_python: Path, tmp_path: Path) -> None:
    """A nonzero EXIT from the last completed slow phase is surfaced, not lost."""
    _cleanup_phases()
    log = tmp_path / "stub.log"
    env = _env(stub_python, log)
    status = REPO_ROOT / ".pytest_cache" / "slow-phase.status"
    status.parent.mkdir(parents=True, exist_ok=True)
    status.write_text("EXIT=7")

    try:
        res = _run([], env, tmp_path / "does-not-exist.txt")
        assert "last slow phase FAILED (EXIT=7)" in res.stderr
        _wait_for_lines(log, 1)
    finally:
        _cleanup_phases()
