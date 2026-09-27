#!/usr/bin/env bash
# E2E journey S: slo serve --web must bring up API + web (Vite flags through
# npm), show both URLs, and shut down cleanly on SIGINT — no stuck status,
# no leftovers, both ports freed.
set -u
BASE="$(cd "$(dirname "$0")" && pwd)"
WT="${JOURNEY_WT:-$(cd "$BASE/../.." && pwd)}"
LOG_DIR="$WT/logs/e2e"; mkdir -p "$LOG_DIR"
LOG=$LOG_DIR/journey_serve.log
export FAKE_NPM_LOG=$LOG_DIR/fake_npm_serve.log
NPMLOG="$FAKE_NPM_LOG"
VPY="$WT/.venv/bin/python"
API_P=8199
WEB_P=4570
FAIL=0
ok()  { echo "PASS: $*"; }
bad() { echo "FAIL: $*"; FAIL=1; }

cd "$WT" || exit 1

port_state() {
  "$VPY" -c "
import socket
s=socket.socket(); s.settimeout(0.5)
try: s.connect(('127.0.0.1',$1)); print('OPEN')
except Exception: print('CLOSED')
"
}

# Preconditions: both ports free. serve REUSES a healthy service and would
# later _kill_port() it on shutdown — never acceptable on someone's live box.
for P in $API_P $WEB_P; do
  if [ "$(port_state "$P")" = "OPEN" ]; then
    bad "journey port $P already in use — pick different ports"; exit 1
  fi
done
ok "ports $API_P + $WEB_P free"

MODE="${1:-success}"
CLEAN=$LOG_DIR/journey_serve.clean

# Fresh build path every run (stale .next would skip the npx build step)
rm -rf apps/web/.next
rm -f "$LOG" "$NPMLOG"

if [ "$MODE" = "buildfail" ]; then
  # ── Negative path: build fails → CLI exits AND must not leak the API ──
  # Plain setsid (no `script` pty): script's exit SIGHUPs the process group
  # and would mask the leak by killing the orphaned uvicorn for us.
  FAKE_NPX_FAIL=1 FAKE_NPX_DELAY=20 NVM_DIR=/nonexistent PATH="$BASE/fakebin:$PATH" \
    setsid ./sloughgpt serve --port $API_P --web --web-port $WEB_P --host localhost \
    > "$LOG" 2>&1 < /dev/null &
  SPID=$!

  CLI_PID=""
  DEAD=""
  for i in $(seq 1 30); do
    if [ -z "$CLI_PID" ]; then
      CLI_PID=$(pgrep -f "cli.py .*--web-port $WEB_P" | head -1 || true)
    elif ! kill -0 "$CLI_PID" 2>/dev/null; then
      DEAD=$i; break
    fi
    sleep 1
  done
  if [ -n "$DEAD" ]; then ok "cli exited after build failure (${DEAD}s)"
  else bad "cli still alive 30s after build failure"; fi

  sed 's/\x1b\[[0-9;]*[A-Za-z]//g' "$LOG" > "$CLEAN"
  if grep -q "Next.js build failed" "$CLEAN"; then ok "build failure reported"
  else bad "no build-failure message in log"; fi

  # Leak check: a leaked uvicorn finishes importing and binds ~10s in —
  # watch the API port for 15s after CLI exit.
  C="CLOSED"
  for i in $(seq 1 15); do
    C=$(port_state "$API_P")
    [ "$C" = "OPEN" ] && break
    sleep 1
  done
  if [ "$C" = "CLOSED" ]; then ok "api port not leaked (closed for 15s after exit)"
  else bad "LEAK: api port $API_P open after build failure"; fi
  LEFT=$(pgrep -f "uvicorn apps.api.server.main:app.*--port $API_P" || true)
  if [ -z "$LEFT" ]; then ok "no orphan uvicorn"
  else kill -9 $LEFT 2>/dev/null; bad "orphan uvicorn: $LEFT"; fi

  if grep -A8 "Traceback" "$CLEAN" | grep -q "cli.py\|commands/dev.py"; then
    bad "CLI traceback in log"
    grep -A8 "Traceback" "$CLEAN" | head -8 | sed 's/^/  | /'
  else ok "no CLI traceback"; fi

  kill -TERM "$SPID" 2>/dev/null
  sleep 1
  LEFT=$(pgrep -f "cli.py .*--web-port $WEB_P" || true)
  if [ -n "$LEFT" ]; then kill -9 $LEFT 2>/dev/null; bad "leftover cli"; fi
  rm -f "$CLEAN"
  echo "RESULT[serve-buildfail]: $([ $FAIL -eq 0 ] && echo ALL PASS || echo FAILURES)"
  exit $FAIL
fi

NVM_DIR=/nonexistent PATH="$BASE/fakebin:$PATH" \
  setsid script -qec "./sloughgpt serve --port $API_P --web --web-port $WEB_P --host localhost" /dev/null \
  > "$LOG" 2>&1 < /dev/null &
SPID=$!

# 1) API answers (<=75s; serve uses no --reload, single process import)
API_T=""
for i in $(seq 1 75); do
  ST=$("$VPY" -c "
import urllib.request
try: print(urllib.request.urlopen('http://localhost:$API_P/health', timeout=1).status)
except Exception: print(0)
" 2>/dev/null)
  if [ "$ST" = "200" ]; then API_T=$i; break; fi
  sleep 1
done
if [ -n "$API_T" ]; then ok "api healthy on :$API_P at t+${API_T}s"
else bad "api never healthy on :$API_P (75s)"; fi

# 2) web answers 200 (<=30s after API)
WEB_T=""
for i in $(seq 1 30); do
  ST=$("$VPY" -c "
import urllib.request
try: print(urllib.request.urlopen('http://localhost:$WEB_P/', timeout=1).status)
except Exception: print(0)
" 2>/dev/null)
  if [ "$ST" = "200" ]; then WEB_T=$i; break; fi
  sleep 1
done
if [ -n "$WEB_T" ]; then ok "web answered on :$WEB_P at t+${WEB_T}s"
else bad "web never answered on :$WEB_P (30s)"; fi

# 3) build went through the shim, then node served the standalone
if [ -f "$NPMLOG" ]; then
  ARGV=$(grep '^argv:' "$NPMLOG" | head -1)
  if [ "$ARGV" = "argv: next build" ]; then
    ok "npx next build intercepted by shim"
  else bad "build did not run through shim: $ARGV"; fi
  if grep -q "server.js" <(pgrep -af "node server.js" || true); then ok "standalone node server running"
  else bad "node server.js not running"; fi
else
  bad "build never invoked npx (no $NPMLOG)"
fi

# 4) both URLs shown to the user (TTY status block)
CLEAN=$LOG_DIR/journey_serve.clean
sed 's/\x1b\[[0-9;]*[A-Za-z]//g' "$LOG" > "$CLEAN"
if grep -q "http://localhost:$WEB_P" "$CLEAN"; then ok "log shows web URL"
else bad "web URL missing from log"; fi
if grep -q ":$API_P" "$CLEAN"; then ok "log shows api URL"
else bad "api URL missing from log"; fi

# 5) SIGINT: clean shutdown, ports freed, no leftovers
CLI_PID=$(pgrep -f "cli.py .*--web-port $WEB_P" | head -1 || true)
if [ -z "$CLI_PID" ]; then bad "cli process not found"; else ok "cli alive (pid $CLI_PID)"; fi
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

# Re-snapshot after shutdown so shutdown lines are covered by later greps
sed 's/\x1b\[[0-9;]*[A-Za-z]//g' "$LOG" > "$CLEAN"
if grep -q "Stopped" "$CLEAN"; then ok "shutdown says Stopped"
else bad "no 'Stopped' in shutdown output"; fi

for P in $API_P $WEB_P; do
  C="OPEN"
  for i in $(seq 1 10); do
    C=$(port_state "$P")
    [ "$C" = "CLOSED" ] && break
    sleep 1
  done
  if [ "$C" = "CLOSED" ]; then ok "port $P freed"
  else bad "port $P still open"; fi
done

kill -TERM "$SPID" 2>/dev/null
sleep 1
for PAT in "cli.py .*--web-port $WEB_P" "uvicorn apps.api.server.main:app.*--port $API_P" "node server.js"; do
  LEFT=$(pgrep -f "$PAT" || true)
  if [ -n "$LEFT" ]; then kill -9 $LEFT 2>/dev/null; bad "leftover killed: $PAT"; fi
done
pkill -f "xdg-open.*$WEB_P" 2>/dev/null || true

# 6) no CLI-internal traceback (child output tracebacks are normal)
if grep -q "Traceback" "$CLEAN"; then
  if grep -A8 "Traceback" "$CLEAN" | grep -q "cli.py\|commands/dev.py"; then
    bad "CLI traceback in log"
    grep -A8 "Traceback" "$CLEAN" | head -8 | sed 's/^/  | /'
  else ok "only child-process traceback"; fi
else ok "no traceback"; fi
rm -f "$CLEAN"

echo "RESULT[serve-web]: $([ $FAIL -eq 0 ] && echo ALL PASS || echo FAILURES)"
exit $FAIL
