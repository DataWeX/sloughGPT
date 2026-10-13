#!/bin/bash
# scripts/test_resilient.sh — chunked, resumable pytest runner.
#
# Promotes the two sweep-era helpers (~/.cache/sweep-logs/watchdog.sh +
# babysitter.sh) into the repo, per card ca167dbd:
#   (b) chunked runner: split the file list, resume from the last completed
#       file, skip any chunk whose log already carries a final summary.
#
# Acceptance:
#   1. killing the run mid-way then rerunning completes the remainder
#      automatically (no flag soup, no manual file lists),
#   2. wedged tests die at the per-test ceiling instead of hanging the run
#      (pair with `timeout = 120` in pytest.ini),
#   3. chunked file-list runs stay the norm — a full `pytest tests/` exceeds
#      10 minutes, which is exactly why bare runs die at reboots.
#
# Usage:
#   scripts/test_resilient.sh [files...]        # chunked run (resumable)
#   scripts/test_resilient.sh --all             # full tests/ tree (default)
#   scripts/test_resilient.sh --list            # show plan/progress, run nothing
#   scripts/test_resilient.sh --reset           # forget state, start over
#   scripts/test_resilient.sh --babysitter ...  # also kill idle-deadlocked
#                                               #   children (opt-in; the sweep
#                                               #   needed it, but kill -9 of a
#                                               #   live test is not free)
#
#   Extra args after `--` are appended to every pytest invocation.
#
# State lives in $TEST_RESILIENT_DIR (default ~/.cache/test-resilient):
#   plan.txt          the ordered file list — pinned on first run so a resume
#                     sees the SAME order even if new files appeared since
#   chunk-003.list    files assigned to chunk 3
#   chunk-003.log     its pytest output (a final summary line = completed)
#   chunk-003.rc      its exit code (recorded once, never overwritten)
#   run.log           progress lines
#
# Exit code: 0 iff every completed chunk exited 0. Re-running after a kill
# re-enters only the unfinished chunks.

set -uo pipefail

DIR="${TEST_RESILIENT_DIR:-$HOME/.cache/test-resilient}"
CHUNK="${TEST_RESILIENT_CHUNK:-8}"
PY=/home/mana/miniconda3/envs/sloughgpt/bin/python
# AGENTS-pinned env: never user-site (orphaned `tests` package shadows the
# repo's), PYTHONPATH carries the three shared roots.
export PYTHONNOUSERSITE=1

MODE="run"
BABYSITTER=0
FILES=()
PYTEST_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --all) MODE="run"; ALL=1; shift ;;
        --list) MODE="list"; shift ;;
        --reset) MODE="reset"; shift ;;
        --babysitter) BABYSITTER=1; shift ;;
        --) shift; PYTEST_ARGS=("$@"); break ;;
        -*) PYTEST_ARGS+=("$1"); shift ;;
        *) FILES+=("$1"); shift ;;
    esac
done

cd "$(git rev-parse --show-toplevel)" || exit 1
export PYTHONPATH="$PWD/packages/downcraft:$PWD/packages/core-py:$PWD/apps/api/server${PYTHONPATH:+:$PYTHONPATH}"

log() { echo "$(date +%T) $*" >> "$DIR/run.log"; echo "$*"; }

# Idle-child killer (promoted from sweep-logs/babysitter.sh): while the pytest
# parent is parked in do_wait, any child idle >=90s with <10s CPU is a
# fork-deadlock casualty — kill -9 it so the join can finish.
babysit() {
    local P=$1
    local sec; sec() { echo "$1" | awk -F: '{if(NF==3) print $1*3600+$2*60+$3; else if(NF==2) print $1*60+$2; else print $1}'; }
    while kill -0 "$P" 2>/dev/null; do
        if [[ "$(cat /proc/$P/wchan 2>/dev/null)" == "do_wait" ]]; then
            for c in $(ps --ppid "$P" -o pid= 2>/dev/null); do
                local et cpu
                et=$(ps -o etimes= -p "$c" 2>/dev/null | tr -d ' ')
                cpu=$(ps -o time= -p "$c" 2>/dev/null | tr -d ' ')
                [[ -z "$et" || -z "$cpu" ]] && continue
                if [[ "$et" -ge 90 ]] && [[ "$(sec "$cpu")" -lt 10 ]]; then
                    log "babysitter: killing idle child $c (etime=${et}s cpu=${cpu})"
                    kill -9 "$c" 2>/dev/null
                fi
            done
        fi
        sleep 20
    done
}

# A chunk log counts as complete only when it carries pytest's final summary.
# (The watchdog used `[0-9]+ (passed|error)`; widened for skipped/deselected/
#  xfail-only and "no tests ran" summaries, which are also terminal.)
has_summary() {
    grep -qE "=+ .*(passed|failed|error|skipped|deselected|xfailed|xpassed|no tests ran).* =+" "$1" 2>/dev/null
}

if [[ "$MODE" == "reset" ]]; then
    rm -rf "$DIR"
    echo "state cleared: $DIR"
    exit 0
fi

mkdir -p "$DIR"

# Pin the plan on first run (resume-safe: a rerun must see the same order).
if [[ ${#FILES[@]} -gt 0 ]]; then
    printf '%s\n' "${FILES[@]}" > "$DIR/plan.txt"
elif [[ ! -f "$DIR/plan.txt" ]]; then
    find tests -name 'test_*.py' | sort > "$DIR/plan.txt"
fi

mapfile -t ALL_FILES < "$DIR/plan.txt"
TOTAL_FILES=${#ALL_FILES[@]}
[[ $TOTAL_FILES -eq 0 ]] && { echo "empty plan"; exit 1; }

# Materialise chunks (recreated every run; completed ones are just skipped).
NCHUNKS=$(( (TOTAL_FILES + CHUNK - 1) / CHUNK ))
rm -f "$DIR"/chunk-*.list
for i in $(seq 0 $((NCHUNKS - 1))); do
    start=$((i * CHUNK))
    end=$((start + CHUNK))
    [[ $end -gt $TOTAL_FILES ]] && end=$TOTAL_FILES
    printf '%s\n' "${ALL_FILES[@]:start:end-start}" > "$DIR/chunk-$(printf '%03d' $i).list"
done

if [[ "$MODE" == "list" ]]; then
    echo "plan: $TOTAL_FILES files -> $NCHUNKS chunks of $CHUNK  (state: $DIR)"
    done_n=0
    for i in $(seq 0 $((NCHUNKS - 1))); do
        L="$DIR/chunk-$(printf '%03d' $i).log"
        if [[ -f "$L" ]] && has_summary "$L"; then
            done_n=$((done_n + 1))
        elif [[ -f "$L" ]]; then
            echo "  chunk $(printf '%03d' $i): INCOMPLETE (resumable)"
        fi
    done
    echo "complete: $done_n/$NCHUNKS"
    exit 0
fi

CURRENT_PID=""
trap '[[ -n "$CURRENT_PID" ]] && kill "$CURRENT_PID" 2>/dev/null; log "interrupted"; exit 130' INT TERM

RC=0
for i in $(seq 0 $((NCHUNKS - 1))); do
    id=$(printf '%03d' "$i")
    L="$DIR/chunk-$id.log"
    LIST="$DIR/chunk-$id.list"
    RCFILE="$DIR/chunk-$id.rc"

    # skip-if-summary-present: already finished on an earlier run. An rc with
    # NO summary means the pytest child was killed from outside (machine
    # death, pkill) — that is an interruption, not a result: drop the rc and
    # resume the chunk below.
    if [[ -f "$RCFILE" ]]; then
        if has_summary "$L"; then
            rec=$(cat "$RCFILE")
            log "chunk $id: already recorded (rc=$rec) — skip"
            # a previously FAILED chunk must still poison this run's exit code,
            # or a rerun would report green over recorded failures.
            [[ $rec -ne 0 ]] && RC=$rec
            continue
        fi
        rm -f "$RCFILE"
        log "chunk $id: rc recorded but no summary (child killed externally) — resuming"
    fi

    RUN_LIST="$LIST"
    if [[ -f "$L" ]] && ! has_summary "$L"; then
        # Interrupted mid-chunk: resume from the last file pytest had entered
        # (everything strictly before it is proven done; the entered one re-runs).
        seen=$(grep -oE '^(tests|apps|packages)/[^:]+\.py' "$L" | awk '!s[$0]++')
        last=$(printf '%s\n' "$seen" | tail -1)
        if [[ -n "$last" ]] && grep -qxF "$last" "$LIST"; then
            RUN_LIST="$DIR/chunk-$id.resume"
            awk -v f="$last" 'found{print} $0==f{found=1; print}' "$LIST" > "$RUN_LIST"
            log "chunk $id: resuming from '$last' ($(wc -l < "$RUN_LIST") files left)"
        else
            log "chunk $id: incomplete log, no parseable progress — rerunning whole chunk"
            mv "$L" "$L.part" 2>/dev/null
        fi
    fi

    log "chunk $id: running $(wc -l < "$RUN_LIST") files"
    "$PY" -m pytest $(tr '\n' ' ' < "$RUN_LIST") "${PYTEST_ARGS[@]}" > "$L" 2>&1 &
    CURRENT_PID=$!
    if [[ $BABYSITTER -eq 1 ]]; then
        babysit "$CURRENT_PID" &
        BS_PID=$!
    fi
    wait "$CURRENT_PID"
    rc=$?
    CURRENT_PID=""
    [[ ${BS_PID:-} ]] && kill "$BS_PID" 2>/dev/null
    echo "$rc" > "$RCFILE"
    log "chunk $id: rc=$rc ($(grep -oE '[0-9]+ (passed|failed|error|skipped)' "$L" | tr '\n' ' '))"
    [[ $rc -ne 0 ]] && RC=$rc
done

# Aggregate: final line per chunk + grand totals (parsing terminal summaries).
echo
echo "── resilient run summary ($DIR) ──"
for i in $(seq 0 $((NCHUNKS - 1))); do
    id=$(printf '%03d' "$i")
    L="$DIR/chunk-$id.log"
    if [[ -f "$DIR/chunk-$id.rc" ]]; then
        tail -1 "$L" 2>/dev/null | sed "s/^/chunk $id: /"
    else
        echo "chunk $id: NOT RUN (interrupted — rerun to resume)"
        RC=1
    fi
done
grep -hoE '[0-9]+ (passed|failed|error|skipped|deselected)' "$DIR"/chunk-*.log 2>/dev/null |
    awk '{s[$2]+=$1} END {for (k in s) printf "%d %s\n", s[k], k}' | sort -k2
log "done rc=$RC"
exit "$RC"
