#!/usr/bin/env bash
# grab-port.sh — graft a dev port instead of murdering its holder.
#
# Ownership rule (why this exists): a port number is a locator, not an
# identity. In multi-session checkouts, `lsof -ti :5173 | xargs kill -9`
# (the old dev:restart) would kill whatever it found — including another
# session's live dev server. This helper signals a holder only when it
# provably serves THIS checkout, verified via the X-Slo-Web-Tree header
# that apps/web/vite.config.ts stamps on every response (sha256 of
# realpath(apps/web), first 16 hex chars — the same id the tree verifier
# computes). Foreign or unknown holders are identified and reported;
# we exit 1 and let a human decide. Ports are grafted, not murdered.
#
# Graft sequence for our own holder: SIGTERM → poll for release
# (GRAB_PORT_DRAIN, default 5s) → SIGKILL only as escalation.
set -euo pipefail

PORT="${1:-}"
if [ -z "$PORT" ]; then
  echo "usage: grab-port.sh <port>" >&2
  exit 2
fi

# 1. Who holds the port? (no holder -> nothing to graft)
PIDS="$(lsof -ti ":$PORT" 2>/dev/null || true)"
if [ -z "$PIDS" ]; then
  echo "grab-port: :$PORT is free"
  exit 0
fi

# 2. Identity: does the holder serve THIS checkout's tree?
WEB_DIR="$(cd "$(dirname "$0")/../apps/web" && pwd)"
TREE_ID="$(node -e "const c=require('crypto'),fs=require('fs');console.log(c.createHash('sha256').update(fs.realpathSync(process.argv[1])).digest('hex').slice(0,16))" "$WEB_DIR")"
SERVED_ID="$(curl -s -m 2 -D - -o /dev/null "http://localhost:$PORT/" 2>/dev/null \
  | tr -d '\r' | awk -F': ' 'tolower($1)=="x-slo-web-tree"{print $2}' || true)"

if [ "$SERVED_ID" != "$TREE_ID" ]; then
  echo "grab-port: REFUSING to signal :$PORT — not ours to graft" >&2
  for pid in $PIDS; do
    echo "  pid $pid: $(ps -p "$pid" -o args= 2>/dev/null || echo '<gone>')" >&2
  done
  echo "  served tree id: ${SERVED_ID:-<none — not an X-Slo-Web-Tree server>} (this checkout: $TREE_ID)" >&2
  echo "  A human must decide; kill it manually only if you mean it." >&2
  exit 1
fi

# 3. Graft: TERM, poll for release, escalate to KILL only after the drain
#    window (in-flight work — HMR clients, pending requests — gets to end).
DRAIN_SECS="${GRAB_PORT_DRAIN:-5}"
for pid in $PIDS; do
  kill -TERM "$pid" 2>/dev/null || true
done
deadline=$(( $(date +%s) + DRAIN_SECS ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  REMAIN="$(lsof -ti ":$PORT" 2>/dev/null || true)"
  [ -z "$REMAIN" ] && break
  sleep 0.25
done

REMAIN="$(lsof -ti ":$PORT" 2>/dev/null || true)"
if [ -n "$REMAIN" ]; then
  echo "grab-port: drain window (${DRAIN_SECS}s) expired, escalating SIGKILL: $REMAIN" >&2
  for pid in $REMAIN; do kill -9 "$pid" 2>/dev/null || true; done
  sleep 0.25
fi

FINAL="$(lsof -ti ":$PORT" 2>/dev/null || true)"
if [ -n "$FINAL" ]; then
  echo "grab-port: :$PORT still held by: $FINAL" >&2
  exit 1
fi
echo "grab-port: :$PORT grafted free (TERM -> ${DRAIN_SECS}s drain -> KILL if needed)"
