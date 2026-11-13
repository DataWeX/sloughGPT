#!/usr/bin/env bash
# prune-worktrees.sh — remove IDLE, CLEAN git worktrees. Never sweep by name.
#
# Why this exists (measured 2026-11-13): the repo carried 115 registered
# worktrees holding 32.2 GB while the disk sat at 98% (2.9G free). Nothing had
# ever pruned them — they accumulated across sessions in four locations
# (61 in-repo `.wt-*`, 25 `~/.cache/wt-*`, ~24 opencode session worktrees, 4
# sibling `Documents/sloughGPT-*`). Removal is safe because a worktree holds no
# history of its own: its branch and every commit stay in the main `.git`.
# Only UNCOMMITTED changes are lost, which is exactly what this refuses to drop.
#
# Safety model — three independent gates, all must pass:
#
#   1. GIT'S OWN CLEANLINESS GATE. Removal runs WITHOUT `--force`, so `git`
#      itself refuses any worktree with modified or untracked files. This is
#      the primary gate and it is not something we reimplement.
#   2. LIVENESS. A worktree with any process whose cwd is inside it is live —
#      another session is working there. Checked via /proc, never by mtime
#      alone (a long compile can go quiet; a running REPL is not idle).
#   3. RECENT ACTIVITY. Anything touched within `--max-age` minutes is skipped
#      even if currently quiet, because a session between commands looks idle.
#
# NEVER delete by name pattern. `~/.cache/wt-*` are often LIVE registered
# worktrees — a 2026-10-21 "orphaned cache" script globbed that path and wiped
# ~60 of them, destroying uncommitted work. See docs/WORKTREE_RECOVERY.md:
# `git worktree list` is the source of truth for what exists; /proc is the
# source of truth for what is alive. A glob is neither.
#
# usage:
#   scripts/prune-worktrees.sh                 # DRY RUN — reports, removes nothing
#   scripts/prune-worktrees.sh --apply         # actually remove
#   scripts/prune-worktrees.sh --apply --max-age 120
#   scripts/prune-worktrees.sh --apply --path ~/.cache/wt-foo   # one target
#
#   --max-age N   minutes of quiet required before a worktree is prunable
#                 (default 60; raise it if sessions are long and quiet)
#   --path P      restrict to a single worktree path (repeatable)
#   --apply       perform removals; without it, print the plan and exit 0
#   --list        print every registered worktree with its verdict, then exit
#
# Exit codes: 0 = ran (dry-run always 0), 1 = usage/error.
set -euo pipefail

DRY=1
MAX_AGE=60
PATHS=()
LIST_ONLY=0

while [ $# -gt 0 ]; do
    case "$1" in
        --apply)    DRY=0 ;;
        --list)     LIST_ONLY=1 ;;
        --max-age)  shift; MAX_AGE="${1:?--max-age needs a number}"; ;;
        --max-age=*) MAX_AGE="${1#*=}"; ;;
        --path)     shift; PATHS+=("${1:?--path needs a path}"); ;;
        --path=*)   PATHS+=("${1#*=}"); ;;
        -h|--help)  sed -n '2,41p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown arg: $1 (try --help)" >&2; exit 1 ;;
    esac
    shift
done

case "$MAX_AGE" in ''|*[!0-9]*) echo "--max-age must be a non-negative integer" >&2; exit 1 ;; esac

# Never prune from inside a worktree we do not control — resolve the main one.
MAIN_WT="$(git worktree list --porcelain | sed -n '1s/^worktree //p')"
[ -n "$MAIN_WT" ] || { echo "ERR: not a git worktree?" >&2; exit 1; }

# /proc scan: is any process's cwd inside $1? Prints the pid, or nothing.
# This is the only reliable liveness signal — mtime misses a process parked
# at a prompt, and `lsof +D` is far too slow to run per worktree.
cwd_owner() {
    local target="$1" p cwd
    for p in /proc/[0-9]*; do
        cwd="$(readlink "$p/cwd" 2>/dev/null)" || continue
        case "$cwd" in
            "$target"|"$target"/*) echo "${p##*/proc/}"; return 0 ;;
        esac
    done
    return 1
}

# Newest mtime under $1, ignoring build/VCS noise. Falls back to 0 when the
# tree is empty — an unfindable timestamp must read as OLD, never as "just now".
#
# Single-pass awk, NOT `sort | head`: `head -1` closes the pipe after one line,
# sort then writes to a closed fd and dies of SIGPIPE, and `set -o pipefail`
# turns that into exit 141 which kills the whole run (observed 2026-11-13 —
# the script died on worktree 3 of 34). awk reads its input to EOF, so no
# producer can ever see a broken pipe here.
newest_mtime() {
    local target="$1"
    find "$target" \
        -path '*/node_modules' -prune -o \
        -path '*/.git' -prune -o \
        -path '*/__pycache__' -prune -o \
        -path '*/.turbo' -prune -o \
        -type f -printf '%T@\n' 2>/dev/null \
        | awk 'NR == 1 || $1 > m { m = $1 } END { if (NR) { split(m, a, "."); print a[1] } }'
}

# Collect targets: explicit --path args, or every registered worktree.
TARGETS=()
if [ "${#PATHS[@]}" -gt 0 ]; then
    TARGETS=("${PATHS[@]}")
else
    while IFS= read -r line; do
        TARGETS+=("$line")
    done < <(git worktree list --porcelain | sed -n 's/^worktree //p')
fi

now="$(date +%s)"
age_limit=$((MAX_AGE * 60))

prunable=()
skipped_live=0
skipped_fresh=0
skipped_dirty=0
skipped_main=0
skipped_missing=0
bytes_planned=0

printf '%-9s %-8s %-9s %-7s %s\n' "VERDICT" "AGE(min)" "SIZE" "DIRTY" "PATH"
printf '%-9s %-8s %-9s %-7s %s\n' "-------" "-------" "----" "-----" "----"

for wt in "${TARGETS[@]}"; do
    if [ ! -d "$wt" ]; then
        printf '%-9s %-46s %s\n' "MISSING" "" "$wt"
        skipped_missing=$((skipped_missing + 1))
        continue
    fi

    if [ "$wt" = "$MAIN_WT" ]; then
        printf '%-9s %-46s %s\n' "SKIP" "main checkout" "$wt"
        skipped_main=$((skipped_main + 1))
        continue
    fi

    # Gate 2 — liveness. Checked BEFORE cleanliness so a busy tree reports as
    # busy rather than as dirty; the distinction tells the operator why.
    if pid="$(cwd_owner "$wt")"; then
        printf '%-9s %-46s %s\n' "LIVE" "pid $pid" "$wt"
        skipped_live=$((skipped_live + 1))
        continue
    fi

    # Dirty = modified OR untracked. git is the authority: ask it rather than
    # re-deriving the rule, so any future ignore/tracking change is inherited.
    if [ -n "$(git -C "$wt" status --porcelain 2>/dev/null)" ]; then
        n="$(git -C "$wt" status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
        printf '%-9s %-46s %s\n' "DIRTY" "$n entries" "$wt"
        skipped_dirty=$((skipped_dirty + 1))
        continue
    fi

    # Gate 3 — recency. A clean-but-just-touched tree means a session is
    # between commands (or an editor has it open).
    mt="$(newest_mtime "$wt")"
    [ -n "$mt" ] || mt=0
    age=$(( (now - mt) / 60 ))
    if [ "$age" -lt "$MAX_AGE" ]; then
        printf '%-9s %-8s %-9s %-7s %s\n' "RECENT" "$age" "-" "-" "$wt"
        skipped_fresh=$((skipped_fresh + 1))
        continue
    fi

    size="$(du -sm "$wt" 2>/dev/null | cut -f1)"
    [ -n "$size" ] || size=0
    bytes_planned=$((bytes_planned + size))
    printf '%-9s %-8s %-9s %-7s %s\n' "PRUNE" "$age" "${size}M" "clean" "$wt"
    prunable+=("$wt")
done

echo
echo "prunable: ${#prunable[@]}  (~$((bytes_planned / 1024)).$(( (bytes_planned % 1024) / 100 )) GB)"
echo "skipped — live: $skipped_live  recent: $skipped_fresh  dirty: $skipped_dirty  main: $skipped_main  missing: $skipped_missing"

if [ "$LIST_ONLY" = "1" ]; then exit 0; fi

if [ "${#prunable[@]}" -eq 0 ]; then
    echo "nothing to prune."
    exit 0
fi

if [ "$DRY" = "1" ]; then
    echo
    echo "DRY RUN — nothing removed. Re-run with --apply."
    exit 0
fi

echo
removed=0
failed=0
for wt in "${prunable[@]}"; do
    # NO --force: git's own cleanliness gate stays armed during the real run,
    # so a worktree that gained a file between scan and remove is refused
    # instead of silently losing it.
    if git worktree remove "$wt" 2>/tmp/opencode/prune-err.$$; then
        removed=$((removed + 1))
        echo "  removed: $wt"
    else
        failed=$((failed + 1))
        echo "  FAILED:  $wt — $(head -1 /tmp/opencode/prune-err.$$)" >&2
    fi
    rm -f /tmp/opencode/prune-err.$$
done

# Drop dangling admin entries left by anything that deleted a dir by hand.
git worktree prune

echo
echo "removed: $removed  failed: $failed"
echo "branches are untouched — every branch and commit still lives in the main .git."
