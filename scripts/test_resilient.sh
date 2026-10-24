#!/usr/bin/env bash
# Chunked, resumable pytest runner — card ca167dbd (test suite resilience).
#
# Why: full-suite runs (~2h) die to power-offs, restarts and fork-deadlocks.
# This runner splits the collected file list into chunks, runs them one by
# one, and records a per-chunk summary ONLY when the chunk runs to completion.
# Kill it mid-run and rerun the same command: completed chunks are skipped
# (their summaries exist), the interrupted chunk re-runs, the remainder runs.
#
# Promotes the sweep prototypes ~/.cache/sweep-logs/{watchdog,babysitter}.sh:
#   - watchdog's "relaunch remainder" role  -> chunk summaries + resume
#   - babysitter's idle-child killer        -> --babysit
#
# Usage:
#   scripts/test_resilient.sh [-c CHUNK_FILES] [-s STATE_DIR] [--babysit]
#                             [-- <args>]
#   <args>: existing files/dirs scope the file list (collect only); anything
#           else (flags like -k/-m/-o) is passed to every chunk run.
# Defaults: chunk = 25 files, state = .resilient-state/ at the repo root.
# Exit code: 0 all chunks completed with no failures, 1 any chunk red,
#             2 setup error or an incomplete run (a chunk died without a summary).
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CHUNK=25
STATE="$REPO_ROOT/.resilient-state"
BABYSIT=0
SCOPE=()
RUNFLAGS=()

while [ $# -gt 0 ]; do
    case "$1" in
        -c) CHUNK="$2"; shift 2 ;;
        -s) STATE="$2"; shift 2 ;;
        --babysit) BABYSIT=1; shift ;;
        --)
            shift
            for a in "$@"; do
                # existing path (repo-relative or caller-relative) = scope;
                # everything else (incl. flag values) = per-chunk run flag
                if [ -e "$a" ] || [ -e "$REPO_ROOT/$a" ]; then SCOPE+=("$a"); else RUNFLAGS+=("$a"); fi
            done
            break ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

case "$CHUNK" in ''|*[!0-9]*) echo "chunk size must be numeric" >&2; exit 2 ;; esac

PY="${SLO_PYTHON:-/home/mana/miniconda3/envs/sloughgpt/bin/python}"
cd "$REPO_ROOT" || exit 2
export PYTHONNOUSERSITE=1
# canonical gate PYTHONPATH (AGENT_SYNC a76143735); keep any caller extras after
_canon="$PWD:$PWD/packages/downcraft:$PWD/packages/core-py:$PWD/apps/api/server:$PWD/packages/mogdb/src:$PWD/apps/cli/src:$PWD/packages/sdk-py:$PWD/packages/app-planner/src"
export PYTHONPATH="$_canon${PYTHONPATH:+:$PYTHONPATH}"

# Idle fork-deadlocked child killer (from babysitter.sh): kills children of a
# deadlocked parent (wchan=do_wait) that are >=90s old and used <10s CPU.
babysit() {
    local PARENT=$1 LOG="$STATE/babysitter.log"
    sec() { echo "$1" | awk -F: '{if(NF==3) print $1*3600+$2*60+$3; else print $1*60+$2}'; }
    while kill -0 "$PARENT" 2>/dev/null; do
        if [ "$(cat "/proc/$PARENT/wchan" 2>/dev/null)" = "do_wait" ]; then
            for c in $(ps --ppid "$PARENT" -o pid= 2>/dev/null); do
                local et cpu
                et=$(ps -o etimes= -p "$c" 2>/dev/null | tr -d ' ')
                cpu=$(ps -o time= -p "$c" 2>/dev/null | tr -d ' ')
                [ -z "$et" ] && continue
                if [ "$et" -ge 90 ] && [ "$(sec "$cpu")" -lt 10 ]; then
                    echo "$(date +%T) killing idle child $c etime=$et cpu=$cpu" >> "$LOG"
                    kill -9 "$c" 2>/dev/null
                fi
            done
        fi
        sleep 20
    done
}

mkdir -p "$STATE" || exit 2

LIST="$STATE/files.txt"
echo "collecting file list ..." >&2
# -qq (not -q): pytest 9 renders --collect-only -q as a <Dir>/<File> tree;
# -qq yields bare nodeids, which the awk below parses to file paths.
"$PY" -m pytest --collect-only -qq "${SCOPE[@]+"${SCOPE[@]}"}" "${RUNFLAGS[@]+"${RUNFLAGS[@]}"}" 2>/dev/null \
    | awk -F'::' '/^(packages|tests|apps)\// {print $1}' \
    | sort -u > "$LIST" || true
NFILES=$(wc -l < "$LIST")
if [ "$NFILES" -eq 0 ]; then
    echo "no test files collected" >&2
    exit 2
fi
echo "$NFILES files, chunk=$CHUNK, state=$STATE" >&2

TOTAL=$(( (NFILES + CHUNK - 1) / CHUNK ))
DONE=0; SKIPPED=0; RED=0; DIED=0
mapfile -t FILES < "$LIST"

for ((i = 0; i < TOTAL; i++)); do
    start=$((i * CHUNK)); end=$((start + CHUNK))
    [ "$end" -gt "$NFILES" ] && end="$NFILES"
    cid="chunk_$(printf '%04d' "$i")"
    label="$cid ($((end - start)) files)"
    summary="$STATE/$cid.summary"
    if [ -f "$summary" ]; then
        echo "SKIP  $label — summary present" >&2
        SKIPPED=$((SKIPPED + 1))
        continue
    fi
    # file-level salvage from a previous death of this chunk: run only files
    # with no .done marker (written by the DIED branch below)
    remaining=()
    for ((j = start; j < end; j++)); do
        [ -f "$STATE/$(echo "${FILES[j]}" | tr '/' '_').done" ] && continue
        remaining+=("${FILES[j]}")
    done
    if [ "${#remaining[@]}" -eq 0 ]; then
        echo "SKIP  $label — all files salvaged from a prior killed run" >&2
        SKIPPED=$((SKIPPED + 1))
        { echo "rc=0"; echo "resumed: all files marked done by salvage"; } > "$summary"
        DONE=$((DONE + 1))
        continue
    fi
    if [ "${#remaining[@]}" -lt $((end - start)) ]; then
        echo "RUN   $label (${#remaining[@]}/$((end - start)) files pending after salvage)" >&2
    else
        echo "RUN   $label" >&2
    fi
    out="$STATE/$cid.log"
    if [ "$BABYSIT" -eq 1 ]; then
        "$PY" -m pytest "${remaining[@]}" "${RUNFLAGS[@]+"${RUNFLAGS[@]}"}" -q -rf > "$out" 2>&1 &
        pytest_pid=$!
        babysit "$pytest_pid" &
        baby_pid=$!
        wait "$pytest_pid"; rc=$?
        kill "$baby_pid" 2>/dev/null
        wait "$baby_pid" 2>/dev/null
    else
        "$PY" -m pytest "${remaining[@]}" "${RUNFLAGS[@]+"${RUNFLAGS[@]}"}" -q -rf > "$out" 2>&1
        rc=$?
    fi
    if [ "$rc" -eq 0 ] || [ "$rc" -eq 1 ] || [ "$rc" -eq 5 ]; then
        # completed (green / red / all-deselected) — record so a rerun
        # never repeats it
        {
            echo "rc=$rc"
            grep -E "=+ .*(passed|failed|error).* =+" "$out" | tail -1
        } > "$summary"
        DONE=$((DONE + 1))
        if [ "$rc" -ne 0 ] && [ "$rc" -ne 5 ]; then
            RED=$((RED + 1))
            echo "RED   $label — see $out" >&2
        fi
    else
        # killed / crashed before a summary. Salvage file-level progress from
        # the log (addopts -v prints one nodeid line per completed test, in
        # execution order): every file strictly before the last visible one
        # completed -> mark it so a rerun resumes from the last completed
        # file, not the chunk start. The file containing the last line may
        # be partial -> it re-runs (safe direction: never skip un-run files).
        last=$(grep -oE "^(packages|tests|apps)/[^:]+\.py" "$out" 2>/dev/null | tail -1)
        marked=0
        if [ -n "$last" ]; then
            for ((j = start; j < end; j++)); do
                [ "${FILES[j]}" = "$last" ] && break
                touch "$STATE/$(echo "${FILES[j]}" | tr '/' '_').done"
                marked=$((marked + 1))
            done
        fi
        echo "DIED  $label rc=$rc — resumed from '$last' ($marked files salvaged)" >&2
        DIED=$((DIED + 1))
    fi
done

echo "=== resilient summary: $DONE run, $SKIPPED skipped (already done), $RED red, $DIED died, $TOTAL total chunks ===" >&2
grep -l "^rc=1" "$STATE"/chunk_*.summary 2>/dev/null | head -20 >&2
if [ "$RED" -gt 0 ]; then exit 1; fi
if [ "$DIED" -gt 0 ]; then exit 2; fi
exit 0
