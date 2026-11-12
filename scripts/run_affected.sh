#!/bin/sh
# Change-scoped test runner (AGENTS.md Test Rule).
#
#   scripts/run_affected.sh [pytest args...]   run only the tests affected by
#                                              files changed since the last
#                                              recorded run
#   RUN_ALL=1 scripts/run_affected.sh [args]   run everything and record
#                                              (keeps the dependency DB fresh)
#
# Selection is mechanical, not filename-guessing: pytest-testmon records
# which source files each test actually reads during a run; the next run
# selects only tests whose files changed since that recorded state. Shared
# utils, conftest edits, and transitive imports are therefore picked up
# without anyone maintaining a mapping.
#
# Conservative by design: no .testmondata yet (first run, fresh clone) or an
# unreadable one means EVERYTHING runs — unknown state is a full run, never a
# guess. A selective run is an optimization of the local loop only; the full
# suite in CI on merge to main remains the backstop (Test Rule clause 3).
#
# Auto-seed (card 20261111_125558 Part A, env-guard 20261112_122610):
# selective mode snapshots the freshest ENVIRONMENT-MATCHING recording from
# another worktree of this repo (scripts/seed_testmon.py — sqlite backup API,
# safe against a live writer; TESTMON_SEED_DB overrides the source). A
# mismatched environment would be cascade-purged on the destination's first
# run, so it is never copied — no match degrades to the full record. A dest
# holding a valid DB with 0 recorded tests (the post-wipe state) re-seeds;
# healthy and invalid dests are trusted as-is. No seed source, a corrupt
# source, or any error → the full record run, exactly as before.
#
# Slow phase (card 20261111_125558 Part B): phase 1 (foreground) adds
# --ignore for exactly the tracked slow paths IN SCOPE of your invocation,
# and ITS exit code is this script's exit code — fast feedback in minutes.
# Phase 2 then re-enters this script with RUN_AFFECTED_PHASE=slow over
# exactly those paths (same selection mode; ignored here = positional there,
# a partition by construction), setsid'd into the background, appending to
# .pytest_cache/slow-phase.log with an EXIT= marker. The last completed slow
# exit code lives in .pytest_cache/slow-phase.status and a nonzero one is
# warned about on the next invocation — no silent reds. Same tests, same
# mode, merely ordered; nothing is deselected.
#
# Scope: the path arguments YOU passed decide what is deferred. A slow path
# passed explicitly runs in phase 1 only (pytest collects explicit files no
# matter what --ignore says — ignoring one would be a no-op or a duplicate),
# and your scope argument itself is stripped from phase 2 so the fast
# closure is never collected a second time; flags ride both phases. No path
# arguments (default or flags-only) = the whole tracked list is deferred.
# Caveat: a flag value that names an existing path would read as scope.
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Both phases need the repo packages on the path (seed helper + pytest).
export PYTHONPATH="$ROOT/packages/downcraft:$ROOT/packages/core-py:$ROOT/apps/api/server${PYTHONPATH:+:$PYTHONPATH}"

SLOW_LIST="${RUN_AFFECTED_SLOW_LIST:-$ROOT/scripts/slow_paths.txt}"

# ── slow list: repo-relative paths, one per line, comments on their own
#    line, no spaces (POSIX sh word-splitting). Missing entries are skipped
#    with a warning so a deleted test file cannot wedge every run. ──
SLOW_ALL=""
if [ -f "$SLOW_LIST" ]; then
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in ''|\#*) continue ;; esac
    if [ ! -e "$ROOT/$line" ]; then
      echo "run_affected: slow list entry missing, skipped: $line" >&2
      continue
    fi
    SLOW_ALL="$SLOW_ALL $line"
  done < "$SLOW_LIST"
fi

# ── selection mode (identical for both phases) ──
if [ "${RUN_ALL:-0}" = "1" ]; then
  # --testmon-noselect = "don't deselect anything, still collect + write"
  # (pytest-testmon 2.2 renamed this; the old --testmon-no-changes does not
  # exist there, so RUN_ALL=1 died with "unrecognized arguments" and the
  # documented full-run-and-re-record path was dead).
  TESTMON_FLAG="--testmon-noselect"             # run all, ignore changes, re-record
else
  # forceselect, not plain --testmon: pytest.ini bakes `-m "not slow"` into
  # addopts, and testmon silently deactivates SELECTION whenever -m/-k/--lf
  # is present (configure.py::_get_noselect_reasons). forceselect overrides
  # that deactivation — selection stays on, the marker filter still applies
  # on top (affected AND not slow).
  TESTMON_FLAG="--testmon-forceselect"
fi

PY="$ROOT/scripts/python"

# ── phase 2 entry: exactly the deferred slow paths, then record and exit ──
if [ "${RUN_AFFECTED_PHASE:-}" = "slow" ]; then
  # The launcher passes its scope-filtered set; a direct invocation falls
  # back to the whole list parsed above.
  SLOW_POSITIONAL="${RUN_AFFECTED_SLOW_PATHS:-$SLOW_ALL}"
  if [ -z "$SLOW_POSITIONAL" ]; then
    # An empty list here would otherwise collect pytest's whole testpaths —
    # the opposite of the partition. Nothing to do, cleanly.
    echo "run_affected: slow phase requested but the slow list is empty — nothing to run" >&2
    exit 0
  fi
  SLOW_LOG="$ROOT/.pytest_cache/slow-phase.log"
  SLOW_STATUS="$ROOT/.pytest_cache/slow-phase.status"
  mkdir -p "$ROOT/.pytest_cache"
  {
    echo "=== slow phase started $(date -Is) ==="
    if [ "$#" -eq 0 ]; then
      # Default invocation must NOT inherit `tests` — phase 2's positional
      # set is exactly the slow list (that is the partition).
      set -- -p no:randomly
    fi
    "$PY" -m pytest "$@" $SLOW_POSITIONAL "$TESTMON_FLAG"
    rc=$?
    echo "EXIT=$rc"
  } >> "$SLOW_LOG" 2>&1
  echo "EXIT=$rc" > "$SLOW_STATUS"
  exit "$rc"
fi

# ── seed + seed health (cards 20261111_125558 + 20261112_122610) ──
# Runs on EVERY selective invocation: healthy/invalid dests are trusted
# silently; a missing or cascade-wiped (0-row) dest re-seeds from an
# environment-matching worktree recording. RUN_ALL re-records atomically —
# no seed, no health check.
if [ "${RUN_ALL:-0}" != "1" ]; then
  "$PY" - "$ROOT" <<'PY' || echo "run_affected: seed failed — full record run" >&2
import sys
from pathlib import Path

root = Path(sys.argv[1])
dest = root / ".testmondata"
sys.path.insert(0, str(root / "scripts"))
from seed_testmon import dest_state, seed_testmon_db

state = dest_state(dest)
if state == "empty":
    # Valid schema, 0 recorded tests = the cascade wipe already purged it
    # (env drift on a previous first run). Nothing to select from — and a
    # stale -wal/-shm would corrupt whatever is written next.
    print("run_affected: .testmondata holds 0 recorded tests (cascade-wiped) — re-seeding")
    for p in (dest, Path(f"{dest}-wal"), Path(f"{dest}-shm")):
        p.unlink(missing_ok=True)
    state = "missing"
if state == "missing":
    if seed_testmon_db(dest, repo=root):
        print("run_affected: seeded .testmondata from an environment-matching worktree recording (fresh-worktree penalty skipped)")
    else:
        print("run_affected: no seed source — full record run (unknown state, Test Rule 3)")
# healthy / invalid → silent: an existing recording is trusted as-is.
PY
fi

# Scope = the path arguments YOU passed — captured BEFORE the default
# injection, normalized (./ prefix, trailing slash) so `./tests/` and
# `tests/` scope identically.
USER_PATHS=""
for a in "$@"; do
  case "$a" in ./*) a="${a#./}" ;; esac
  a="${a%/}"
  if [ -e "$a" ] || [ -e "$ROOT/$a" ]; then
    USER_PATHS="$USER_PATHS $a"
  fi
done

if [ "$#" -eq 0 ]; then
  # 2026-10-25 test-tree consolidation: tests live at the unified tests/ root
  # (packages/core-py/tests is gone — the old default errored on bare invoke).
  # Collect everything; testmon forceselect then narrows to the changed closure.
  set -- -p no:randomly tests
fi

# The last COMPLETED slow phase's exit code — a red must not vanish silently.
SLOW_STATUS="$ROOT/.pytest_cache/slow-phase.status"
if [ -f "$SLOW_STATUS" ] && grep -q '^EXIT=[^0]' "$SLOW_STATUS" 2>/dev/null; then
  echo "run_affected: WARNING — last slow phase FAILED ($(cat "$SLOW_STATUS")): see .pytest_cache/slow-phase.log" >&2
fi

# ── the deferred set: what phase 1 ignores is EXACTLY what phase 2 runs. ──
# A slow path passed explicitly is excluded from BOTH sides (pytest collects
# explicit files regardless of --ignore: ignoring one would be a no-op at
# best, a duplicate run at worst). Without path arguments the whole tracked
# list defers; with a scope, only the slow paths under it — everything else
# is outside your scope in either phase. Symmetry is the invariant: an
# ignored path is always re-run in phase 2, so nothing is ever dropped.
SLOW_DEFERRED=""
for p in $SLOW_ALL; do
  explicit=0
  for u in $USER_PATHS; do
    if [ "$p" = "$u" ]; then explicit=1; break; fi
  done
  if [ "$explicit" = "1" ]; then continue; fi
  if [ -z "$USER_PATHS" ]; then
    SLOW_DEFERRED="$SLOW_DEFERRED $p"
    continue
  fi
  for u in $USER_PATHS; do
    case "$p" in "$u"/*) SLOW_DEFERRED="$SLOW_DEFERRED $p"; break ;; esac
  done
done

SLOW_IGNORES=""
for p in $SLOW_DEFERRED; do
  SLOW_IGNORES="$SLOW_IGNORES --ignore=$p"
done

"$PY" -m pytest "$@" $SLOW_IGNORES "$TESTMON_FLAG"
PHASE1_RC=$?

# ── launch the deferred remainder in the background ──
if [ -n "$SLOW_DEFERRED" ]; then
  # Drop every path argument before launching: phase 2's positionals are
  # the deferred slow paths, so re-passing the scope would re-collect the
  # fast closure we just ran. Flags (and their values) ride along untouched.
  n=$#
  i=0
  while [ "$i" -lt "$n" ]; do
    a="$1"
    shift
    case "$a" in ./*) na="${a#./}" ;; *) na="$a" ;; esac
    na="${na%/}"
    if [ -e "$na" ] || [ -e "$ROOT/$na" ]; then
      : # scope positional — replaced by the deferred paths
    else
      set -- "$@" "$a"
    fi
    i=$((i + 1))
  done
  SLOW_LOG="$ROOT/.pytest_cache/slow-phase.log"
  mkdir -p "$ROOT/.pytest_cache"
  echo "run_affected: slow phase ($SLOW_DEFERRED) running in background — log: $SLOW_LOG"
  RUN_AFFECTED_PHASE=slow \
    RUN_AFFECTED_SLOW_PATHS="$SLOW_DEFERRED" \
    RUN_AFFECTED_SLOW_LIST="$SLOW_LIST" \
    setsid nohup "$ROOT/scripts/run_affected.sh" "$@" >> "$SLOW_LOG" 2>&1 &
fi

exit "$PHASE1_RC"
