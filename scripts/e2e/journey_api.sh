#!/usr/bin/env bash
# E2E: slo dev must report a crashed API ("API server exited") and end the session
# instead of showing "starting" forever. Phase 0: startup SIGINT must end the
# session + clean up (no orphan children). Self-contained; kills everything.
set -u
BASE="$(cd "$(dirname "$0")" && pwd)"
WT="${JOURNEY_WT:-$(cd "$BASE/../.." && pwd)}"
LOG_DIR="$WT/logs/e2e"; mkdir -p "$LOG_DIR"
LOG=$LOG_DIR/journey_api.log
export FAKE_NPM_LOG=$LOG_DIR/fake_npm_api.log
VPY="$("$BASE/../../scripts/python" --resolve 2>/dev/null || echo python3)"
WEB_PORT=4569
API_PORT=8177
FAIL=0
ok()  { echo "PASS: $*"; }
bad() { echo "FAIL: $*"; FAIL=1; }

cd "$WT" || exit 1
rm -f "$LOG" "$FAKE_NPM_LOG"

# ── Phase 0: SIGINT during startup (pre-dashboard) must end the session ──
# cmd_dev registers its flag-only handler at function entry (3e1d71da8);
# DevDashboard.serve swaps in its own on start. Either way the interrupt
# must reach the `finally` (cleanup + summary) instead of orphaning the
# spawned uvicorn/web child or leaving the session running.
LOG0="$LOG_DIR/journey_api.phase0.log"
rm -f "$LOG0"
NVM_DIR=/nonexistent PATH="$BASE/fakebin:$PATH" \
  setsid script -qec "./sloughgpt --port $API_PORT dev --web-port $WEB_PORT" /dev/null \
  > "$LOG0" 2>&1 < /dev/null &
S0=$!
C0=""
U0=""
for i in $(seq 1 15); do
  C0=$(pgrep -f "cli.py --port $API_PORT dev --web-port $WEB_PORT" | head -1 || true)
  U0=$(pgrep -f "uvicorn apps.api.server.main:app.*--port $API_PORT" | head -1 || true)
  [ -n "$C0" ] && [ -n "$U0" ] && break
  sleep 1
done
if [ -z "$C0" ] || [ -z "$U0" ]; then
  bad "phase0: cli/api never started (cli=$C0 api=$U0)"
else
  # t+2 after both children exist: main thread is in dashboard startup —
  # the window where only the entry/dashboard handlers are active.
  sleep 2
  kill -INT "$C0" 2>/dev/null
  E0=""
  for i in $(seq 1 10); do
    if ! kill -0 "$C0" 2>/dev/null; then E0=$i; break; fi
    sleep 1
  done
  if [ -n "$E0" ]; then ok "phase0: startup SIGINT honoured (exit ${E0}s)"
  else bad "phase0: cli survived startup SIGINT"; kill -9 "$C0" 2>/dev/null; fi
  if grep -aq "Dev Server Stopped" <(sed $'s/\x1b\[[0-9;]*[A-Za-z]//g' "$LOG0"); then
    ok "phase0: shutdown summary printed (finally ran)"
  else
    bad "phase0: no 'Dev Server Stopped' summary after startup SIGINT"
  fi
fi
kill -TERM "$S0" 2>/dev/null
sleep 1
ORPH0=""
ORPH_PIDS=""
for PAT in "cli.py --port $API_PORT dev --web-port $WEB_PORT" "uvicorn apps.api.server.main:app.*--port $API_PORT" "$BASE/fakebin/npm"; do
  L=$(pgrep -f "$PAT" || true)
  if [ -n "$L" ]; then ORPH0="$ORPH0 $PAT=[$L]"; ORPH_PIDS="$ORPH_PIDS $L"; fi
done
if [ -z "$ORPH0" ]; then ok "phase0: no orphan children after startup SIGINT"
else kill -9 $ORPH_PIDS 2>/dev/null; bad "phase0 orphan:$ORPH0"; fi
for P in $API_PORT $WEB_PORT; do
  PC="OPEN"
  for i in $(seq 1 5); do
    PC=$("$VPY" -c "
import socket
s=socket.socket(); s.settimeout(0.5)
try: s.connect(('127.0.0.1',$P)); print('OPEN')
except Exception: print('CLOSED')
")
    [ "$PC" = "CLOSED" ] && break
    sleep 1
  done
  if [ "$PC" = "CLOSED" ]; then ok "phase0: port $P freed"
  else bad "phase0: port $P still open after startup SIGINT"; fi
done

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
