#!/usr/bin/env bash
# Main sync guard — keep local `main` from silently diverging from origin/main.
#
# Failure this prevents (2026-10-27/28): local main forked at 9002f1a0, took
# 149 local-only commits ("local only, not pushed") while origin/main moved 35
# the other way — reconciliation = 8 conflicts incl. a Next-era tree that would
# have reverted the Next removal (#217) if pushed.
#
# Rule: local main tracks origin/main. Land via PR (canonical) or commit on a
# lane branch and `git push origin HEAD:main`. Stacks of unpushed commits on
# main are the bug.
#
# Escape (offline / deliberate landing): SLO_MAIN_SYNC_SKIP=1
set -euo pipefail

[ "${SLO_MAIN_SYNC_SKIP:-0}" = "1" ] && exit 0

branch=$(git symbolic-ref --short -q HEAD || true)
[ "$branch" = "main" ] || exit 0   # lane branches and detached HEADs are unaffected

# Best effort freshness — a stale ref still counts local-only commits, and
# each subsequent commit recounts, so drift is caught within 5 commits.
git fetch origin main --quiet 2>/dev/null || true

ahead=$(git rev-list --count origin/main..HEAD 2>/dev/null || echo 0)
behind=$(git rev-list --count HEAD..origin/main 2>/dev/null || echo 0)

if [ "$ahead" -gt 5 ]; then
  cat >&2 <<EOF
✗ local main is ${ahead} commits ahead of origin/main (limit 5) — refusing.

  Local main must track origin/main. Land your work instead:
    - open a PR against main (canonical path, see docs/AGENT_SYNC.md), or
    - commit on a lane branch and: git push origin HEAD:main
  Behind origin/main: ${behind}.

  Deliberate override (rare): SLO_MAIN_SYNC_SKIP=1 git commit ...
EOF
  exit 1
fi

if [ "$behind" -gt 0 ]; then
  printf '⚠ local main is behind origin/main by %s — fetch + ff before working (docs/AGENT_SYNC.md)\n' "$behind" >&2
fi

exit 0
