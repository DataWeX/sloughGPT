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
# Cost note: recording traces file reads (~slower per test); selection pays
# it back by skipping unaffected tests. Delete .testmondata to force a full
# re-record.
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ "$#" -eq 0 ]; then
  set -- -p no:randomly packages/core-py/tests
fi

if [ "${RUN_ALL:-0}" = "1" ]; then
  TESTMON_FLAG="--testmon-no-changes"           # run all, ignore changes, re-record
else
  # forceselect, not plain --testmon: pytest.ini bakes `-m "not slow"` into
  # addopts, and testmon silently deactivates SELECTION whenever -m/-k/--lf
  # is present (configure.py::_get_noselect_reasons). forceselect overrides
  # that deactivation — selection stays on, the marker filter still applies
  # on top (affected AND not slow).
  TESTMON_FLAG="--testmon-forceselect"
fi

PYTHONPATH="$ROOT/packages/downcraft:$ROOT/packages/core-py:$ROOT/apps/api/server${PYTHONPATH:+:$PYTHONPATH}" \
  exec "$ROOT/scripts/python" -m pytest "$@" "$TESTMON_FLAG"
