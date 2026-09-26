#!/usr/bin/env bash
# E2E: slo dev must report a crashed API ("API server exited") and end the session
# instead of showing "starting" forever. Self-contained; kills everything before exit.
set -u
BASE="$(cd "$(dirname "$0")" && pwd)"
WT="${JOURNEY_WT:-$(cd "$BASE/../.." && pwd)}"
LOG_DIR="$WT/logs/e2e"; mkdir -p "$LOG_DIR"
LOG=$LOG_DIR/journey_api.log
export FAKE_NPM_LOG=$LOG_DIR/fake_npm_api.log
VPY="$WT/.venv/bin/python"
WEB_PORT=4569
API_PORT=8177
FAIL=0
ok()  { echo "PASS: $*"; }
bad() { echo "FAIL: $*"; FAIL=1; }

cd "$WT" || exit 1
rm -f "$LOG" "$FAKE_NPM_LOG"

NVM_DIR=/nonexistent PATH="$BASE/fakebin:$PATH" \
  setsid script -qec "./sloughgpt --port $API_PORT dev --web-port $WEB_PORT" /dev/null \
  > "$LOG" 2>&1 < /dev/null &
SPID=$!

# 1) wait for the API process (reloader parent) to exist (<=20s)
API_PID=""
for i in $(seq 1 20); do
  API_PID=$(pgrep -f "uvicorn apps.api.server.main:app.*--port $API_PORT" | head -1 || true)
  [ -n "$API_PID" ] && break
  sleep 1
done
if [ -n "$API_PID" ]; then ok "api process up (pid $API_PID, t+${i}s)"
else bad "api process never appeared"; fi

# 2) wait for the web child too, so the session is fully up
for i in $(seq 1 15); do
  [ -f "$FAKE_NPM_LOG" ] && break
  sleep 1
done
[ -f "$FAKE_NPM_LOG" ] && ok "web child spawned" || bad "web child never spawned"

# 3) kill every uvicorn for this port (parent reloader + child)
PIDS=$(pgrep -f "uvicorn apps.api.server.main:app.*--port $API_PORT" || true)
[ -n "$PIDS" ] && kill -9 $PIDS 2>/dev/null && ok "killed api: $PIDS" || bad "no api pid to kill"

# 4) CLI must exit on its own, having printed the error (<=20s)
CLI_PID=$(pgrep -f "cli.py --port $API_PORT dev --web-port $WEB_PORT" | head -1 || true)
DEAD=""
for i in $(seq 1 20); do
  if [ -n "$CLI_PID" ] && ! kill -0 "$CLI_PID" 2>/dev/null; then DEAD=$i; break; fi
  sleep 1
done
if [ -n "$DEAD" ]; then ok "cli exited on its own after api death (${DEAD}s)"
else bad "cli still alive 20s after api death"; fi

if grep -aq "API server exited" "$LOG"; then ok "log contains 'API server exited'"
else bad "no 'API server exited' in log"; fi

# 5) ports freed (cleanup ran)
C="OPEN"
for i in $(seq 1 10); do
  C=$("$VPY" -c "
import socket
s=socket.socket(); s.settimeout(0.5)
try: s.connect(('127.0.0.1',$WEB_PORT)); print('OPEN')
except Exception: print('CLOSED')
" 2>/dev/null)
  [ "$C" = "CLOSED" ] && break
  sleep 1
done
[ "$C" = "CLOSED" ] && ok "web port freed" || bad "web port $WEB_PORT still open"

# 6) reap stragglers
kill -TERM "$SPID" 2>/dev/null
sleep 1
for PAT in "cli.py --port $API_PORT dev --web-port $WEB_PORT" "uvicorn apps.api.server.main:app.*--port $API_PORT"; do
  LEFT=$(pgrep -f "$PAT" || true)
  if [ -n "$LEFT" ]; then kill -9 $LEFT 2>/dev/null; bad "leftover killed: $PAT"; fi
done
# Traceback check: only CLI-internal frames are a bug (child crashes are normal)
sed 's/\x1b\[[0-9;]*[A-Za-z]//g' "$LOG" | grep -A8 "Traceback" | grep -q "cli.py\|commands/dev.py" \
  && { bad "traceback in cli log"; grep -a -A5 "Traceback" "$LOG" | head -8; } || true

echo "RESULT[api-death]: $([ $FAIL -eq 0 ] && echo ALL PASS || echo FAILURES)"
exit $FAIL
