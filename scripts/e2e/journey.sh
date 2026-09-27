#!/usr/bin/env bash
# E2E journey: slo dev --web-port N must bind N (Next=PORT env, Vite=--port argv)
# and clean up on SIGINT. Self-contained: kills everything before exit.
set -u
LABEL="$1"; WEB_PORT="$2"; MODE="$3"   # MODE: next | vite
BASE="$(cd "$(dirname "$0")" && pwd)"
WT="${JOURNEY_WT:-$(cd "$BASE/../.." && pwd)}"
LOG_DIR="$WT/logs/e2e"; mkdir -p "$LOG_DIR"
LOG=$LOG_DIR/journey_${LABEL}.log
export FAKE_NPM_LOG=$LOG_DIR/fake_npm_${LABEL}.log
NPMLOG="$FAKE_NPM_LOG"
VPY="$WT/.venv/bin/python"
FAIL=0
ok()  { echo "PASS: $*"; }
bad() { echo "FAIL: $*"; FAIL=1; }

cd "$WT" || exit 1
if [ "$MODE" = "vite" ]; then
  sed -i 's/"dev": "next dev"/"dev": "vite"/' apps/web/package.json
fi
grep -o '"dev": "[^"]*"' apps/web/package.json

rm -f "$LOG" "$NPMLOG"
NVM_DIR=/nonexistent PATH="$BASE/fakebin:$PATH" \
  setsid script -qec "./sloughgpt --port 8177 dev --web-port $WEB_PORT" /dev/null \
  > "$LOG" 2>&1 < /dev/null &
SPID=$!

# 1) wait for the requested port to answer HTTP (<=45s)
READY_T=""
for i in $(seq 1 45); do
  ST=$("$VPY" -c "
import urllib.request
try: print(urllib.request.urlopen('http://localhost:$WEB_PORT/', timeout=1).status)
except Exception: print(0)
" 2>/dev/null)
  if [ "$ST" = "200" ]; then READY_T=$i; break; fi
  sleep 1
done

CLI_PID=$(pgrep -f "cli.py --port 8177 dev --web-port $WEB_PORT" | head -1)

if [ -n "$READY_T" ]; then ok "web answered on :$WEB_PORT at t+${READY_T}s"
else bad "web never answered on :$WEB_PORT (45s)"; fi
if [ -n "$CLI_PID" ]; then ok "cli alive (pid $CLI_PID)"; else bad "cli process gone"; fi

# 2) the child must have received the right port channel
if [ -f "$NPMLOG" ]; then
  ARGV=$(grep '^argv:' "$NPMLOG" | head -1)
  BOUND=$(grep '^BOUND_PORT:' "$NPMLOG" | head -1)
  PENV=$(grep '^PORT_ENV:' "$NPMLOG" | head -1)
  echo "  npm got: $ARGV | $PENV | $BOUND"
  case "$MODE" in
    vite)
      if echo "$ARGV" | grep -q -- "--port $WEB_PORT --host 0.0.0.0"; then ok "vite argv carries --port/--host"
      else bad "vite argv missing flags: $ARGV"; fi
      ;;
    next)
      if [ "$ARGV" = "argv: run dev" ]; then ok "next argv is plain npm run dev"
      else bad "next argv polluted: $ARGV"; fi
      if echo "$PENV" | grep -q "PORT_ENV: $WEB_PORT"; then ok "PORT env set for next"
      else bad "PORT env wrong: $PENV"; fi
      ;;
  esac
  if [ "$BOUND" = "BOUND_PORT: $WEB_PORT" ]; then ok "shim bound $WEB_PORT"
  else bad "bound wrong: $BOUND"; fi
else
  bad "npm child never spawned (no log at $NPMLOG)"
fi

# 3) SIGINT cleanup: cli exits, ports freed
if [ -n "$CLI_PID" ]; then
  kill -INT "$CLI_PID" 2>/dev/null
  DEAD=""
  for i in $(seq 1 20); do
    if ! kill -0 "$CLI_PID" 2>/dev/null; then DEAD=$i; break; fi
    sleep 1
  done
  if [ -n "$DEAD" ]; then ok "cli exited after SIGINT (${DEAD}s)"
  else bad "cli still alive 20s after SIGINT"; fi
fi
C="OPEN"
for i in $(seq 1 10); do
  C=$("$VPY" -c "
import socket
s=socket.socket(); s.settimeout(0.5)
try: s.connect(('127.0.0.1',$WEB_PORT)); print('OPEN')
except Exception: print('CLOSED')
" 2>/dev/null)
  if [ "$C" = "CLOSED" ]; then break; fi
  sleep 1
done
if [ "$C" = "CLOSED" ]; then ok "web port freed"; else bad "web port $WEB_PORT still open"; fi

# 4) reap stragglers
kill -TERM "$SPID" 2>/dev/null
sleep 1
STILL=$(pgrep -f "cli.py --port 8177 dev --web-port $WEB_PORT" | head -1 || true)
if [ -z "$STILL" ]; then ok "no leftover cli"
else kill -9 "$STILL" 2>/dev/null; bad "had to SIGKILL leftover"; fi

if [ "$MODE" = "vite" ]; then
  git checkout -- apps/web/package.json
fi
echo "RESULT[$LABEL]: $([ $FAIL -eq 0 ] && echo ALL PASS || echo FAILURES)"
exit $FAIL
