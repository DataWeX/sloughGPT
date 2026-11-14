#!/bin/sh
# run_frontend_latest.sh — serve the sloughGPT frontend from a worktree, always
# at the latest changes.
#
#   scripts/run_frontend_latest.sh
#       sync the default worktree (~/.cache/wt-main-latest) to origin/main and
#       serve it on :5174
#   scripts/run_frontend_latest.sh <worktree-path>
#       serve that worktree AS-IS (your branch's code, no reset) on :5174
#   scripts/run_frontend_latest.sh [--watch]
#       keep syncing to origin/main and restart the server whenever main moves
#
# Guarantees (AGENTS.md port rule):
#   * fixed port (--strictPort) — never grabs or restarts a foreign holder of
#     :3000 / :5173; if the port is held by someone else, we refuse and exit 1
#   * restarts signal ONLY the process group recorded in our pidfile, and only
#     after verifying its cwd lives inside this worktree
#   * `npm ci` runs only when package-lock.json changed since the last install
#     (stamp file under /tmp/opencode — the worktree stays clean)
#   * prints the served tree's X-Slo-Web-Tree header so you can SEE which tree
#     is live in your browser
set -eu

DEFAULT_WORKTREE="$HOME/.cache/wt-main-latest"
PORT="${SLO_FRONTEND_PORT:-5174}"
PIDFILE="/tmp/opencode/run_frontend_latest-${PORT}.pid"
LOGFILE="/tmp/opencode/run_frontend_latest-${PORT}.log"

TARGET="$DEFAULT_WORKTREE"
WATCH=0

for arg in "$@"; do
    case "$arg" in
        --watch) WATCH=1 ;;
        -h | --help)
            sed -n '2,21p' "$0"
            exit 0
            ;;
        -*) echo "unknown flag: $arg" >&2
            exit 2
            ;;
        *) TARGET="$arg" ;;
    esac
done

[ -d "$TARGET" ] || {
    echo "no such worktree: $TARGET" >&2
    exit 1
}
TARGET=$(cd "$TARGET" && pwd)

# Only the default (main-tracking) worktree gets reset; a lane path is served
# as-is so you can inspect your own branch's changes.
SYNCABLE=0
if [ "$TARGET" = "$(cd "$DEFAULT_WORKTREE" 2>/dev/null && pwd)" ]; then
    SYNCABLE=1
fi

stamp_key=$(printf '%s' "$TARGET" | sha1sum | cut -c1-12)
STAMP="/tmp/opencode/run_frontend_latest-${stamp_key}.stamp"

log() { printf '%s\n' "$*"; }

sync_main() {
    git -C "$TARGET" fetch origin main --quiet
    git -C "$TARGET" checkout --quiet --detach origin/main
    log "→ synced to $(git -C "$TARGET" log --oneline -1)"
}

ensure_deps() {
    want=$(git -C "$TARGET" rev-parse HEAD:package-lock.json 2>/dev/null || echo "no-lock")
    have=$(cat "$STAMP" 2>/dev/null || echo "")
    if [ ! -d "$TARGET/node_modules" ] || [ "$want" != "$have" ]; then
        log "→ npm ci (lockfile changed or first install)…"
        (cd "$TARGET" && npm ci --no-audit --no-fund >/dev/null 2>&1) || {
            echo "npm ci failed — run manually: cd $TARGET && npm ci" >&2
            exit 1
        }
        printf '%s' "$want" >"$STAMP"
        log "→ deps installed"
    else
        log "→ deps up to date"
    fi
}

port_holder() {
    # pid of whoever answers on $PORT, or empty
    command -v fuser >/dev/null 2>&1 || return 0
    fuser -n tcp "$PORT" 2>/dev/null | tr -d ' ' || true
}

our_pid() { cat "$PIDFILE" 2>/dev/null || true; }

pid_ours() {
    # true only if $1 is alive AND its cwd is inside this worktree
    pid="$1"
    [ -n "$pid" ] || return 1
    kill -0 "$pid" 2>/dev/null || return 1
    cwd=$(readlink "/proc/$pid/cwd" 2>/dev/null || echo "")
    case "$cwd" in
        "$TARGET" | "$TARGET"/*) return 0 ;;
        *) return 1 ;;
    esac
}

stop_server() {
    pid=$(our_pid)
    if pid_ours "$pid"; then
        log "→ stopping our server (pid $pid)"
        kill -TERM "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
        i=0
        while kill -0 "$pid" 2>/dev/null && [ "$i" -lt 20 ]; do
            sleep 0.5
            i=$((i + 1))
        done
        kill -KILL "-$pid" 2>/dev/null || true
    elif [ -n "$pid" ]; then
        log "→ pidfile pid $pid is not ours (different cwd) — leaving it alone"
    fi
    rm -f "$PIDFILE"

    # Refuse to co-occupy a port held by a foreign server (port rule).
    holder=$(port_holder)
    if [ -n "$holder" ]; then
        echo "port $PORT is held by pid $holder, which this script does not own." >&2
        echo "Refusing — a human decides (see scripts/grab-port.sh)." >&2
        exit 1
    fi
}

start_server() {
    mkdir -p /tmp/opencode
    (cd "$TARGET/apps/web" && exec setsid npm run dev -- --port "$PORT" --strictPort \
        >"$LOGFILE" 2>&1) &
    pid=$!
    printf '%s' "$pid" >"$PIDFILE"
    log "→ starting vite on :$PORT (pid $pid, log $LOGFILE)"
}

verify() {
    # X-Slo-Web-Tree is sha256(realpath(apps/web))[:16] — a checkout-path
    # identity, not a git sha: it proves WHICH SERVER answered (ours, not a
    # leftover foreign holder); the printed HEAD proves which commit it runs.
    want_tree=$(printf '%s' "$(realpath "$TARGET/apps/web")" | sha256sum | cut -c1-16)
    i=0
    while [ "$i" -lt 30 ]; do
        tree=$(curl -s -o /dev/null -D - --max-time 3 "http://localhost:$PORT/" \
            2>/dev/null | tr -d '\r' | awk 'tolower($1)=="x-slo-web-tree:"{print $2}')
        if [ -n "$tree" ]; then
            if [ "$tree" != "$want_tree" ]; then
                echo "header $tree is not this worktree ($want_tree) — foreign holder?" >&2
                exit 1
            fi
            log ""
            log "✓ serving http://localhost:$PORT/  (X-Slo-Web-Tree: $tree — this worktree)"
            git -C "$TARGET" log --oneline -1 | sed 's/^/  commit: /'
            return 0
        fi
        i=$((i + 1))
        sleep 1
    done
    echo "server did not answer on :$PORT — check $LOGFILE" >&2
    exit 1
}

restart() {
    stop_server
    start_server
    verify
}

log "target: $TARGET$( [ "$SYNCABLE" = 1 ] && echo ' (main-tracking — will sync)' || echo ' (lane — served as-is)' )"
if [ "$SYNCABLE" = 1 ]; then
    sync_main
fi
ensure_deps
restart

if [ "$WATCH" = 1 ]; then
    if [ "$SYNCABLE" != 1 ]; then
        log "--watch only makes sense for the main-tracking worktree; exiting"
        exit 0
    fi
    log "watch: polling origin/main every 30s (Ctrl+C to stop the watcher; the server keeps running)"
    last=$(git -C "$TARGET" rev-parse HEAD)
    while :; do
        sleep 30
        remote=$(git ls-remote origin main 2>/dev/null | cut -f1 || printf '%s' "$last")
        if [ -n "$remote" ] && [ "$remote" != "$last" ]; then
            log "→ origin/main moved: $(printf '%.9s' "$last") → $(printf '%.9s' "$remote")"
            sync_main
            ensure_deps
            restart
            last=$(git -C "$TARGET" rev-parse HEAD)
        fi
    done
fi
