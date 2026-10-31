#!/usr/bin/env bash
# worktree-link-model-cache.sh — share one HF model cache across git worktrees.
#
# /models/ is gitignored (.gitignore) and the loader resolves the cache by
# walking up from cwd (<repo>/models/hf-cache/hub), so every fresh worktree
# starts model-blind with an EMPTY cache while the weights sit in the main
# checkout. Autoload then fails ("no local weights, SLO_BOOTSTRAP_HF off") and
# readiness surfaces answer E_NO_MODEL. Copying weights per worktree is wrong
# (hundreds of MB duplicated). Symlink instead: zero extra disk, one shared
# cache, every worktree sees the weights.
#
# usage:
#   worktree-link-model-cache.sh [worktree-path] [donor-cache-path]
#
#   worktree-path    worktree to arm (default: cwd)
#   donor-cache-path shared models/hf-cache to link (default: main worktree's)
#
# Idempotent: re-running is a no-op. Never deletes a real (non-symlink) dir —
# if the worktree already has its own populated cache, it is left untouched.
set -euo pipefail

WT="${1:-.}"
DONOR="${2:-}"

# Resolve the donor: explicit arg → main worktree's models/hf-cache.
if [ -z "$DONOR" ]; then
    # --porcelain: "worktree <path>" per block — path may contain spaces.
    # sed (not head) so the pipeline never SIGPIPEs under pipefail.
    MAIN_WT="$(git -C "$WT" worktree list --porcelain | sed -n '1s/^worktree //p')"
    DONOR="$MAIN_WT/models/hf-cache"
fi

if [ ! -d "$DONOR/hub" ]; then
    echo "ERR: donor cache has no hub/ dir: $DONOR/hub" >&2
    echo "     (pass the shared cache path explicitly as arg 2)" >&2
    exit 1
fi

LINK="$WT/models/hf-cache"

# Idempotency + safety: never clobber a real directory with contents.
if [ -L "$LINK" ]; then
    if [ "$(readlink -f "$LINK")" = "$(readlink -f "$DONOR")" ]; then
        echo "OK: already linked → $DONOR"
        exit 0
    fi
    echo "re-linking $LINK → $DONOR (was: $(readlink "$LINK"))"
    rm "$LINK"
elif [ -d "$LINK" ]; then
    if [ -n "$(ls -A "$LINK" 2>/dev/null)" ]; then
        echo "SKIP: $LINK is a real, populated directory — leaving it untouched." >&2
        echo "      (remove it yourself if you want the shared cache instead)" >&2
        exit 0
    fi
    rmdir "$LINK"
fi

mkdir -p "$WT/models"
ln -s "$DONOR" "$LINK"
echo "OK: $LINK → $DONOR"
