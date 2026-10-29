#!/usr/bin/env bash
# E2E journey R: serve --web REUSES a healthy foreign API on --port and must
# leave it alive after its own shutdown — cleanup frees only ports we spawned.
# Phase 1: SIGINT during startup (before ready) must be honoured — bash `&`
#          inherits SIGINT as ignored, which CPython used to honour by
#          silently dropping the interrupt until our handler was installed.
# Phase 2: SIGINT while idle in the monitor loop — cleanup's child exits must
#          not be mistaken for crashes and web respawned after "Stopped".
set -u
BASE="$(cd "$(dirname "$0")" && pwd)"
WT="${JOURNEY_WT:-$(cd "$BASE/../.." && pwd)}"
LOG_DIR="$WT/logs/e2e"; mkdir -p "$LOG_DIR"
LOG=$LOG_DIR/journey_reuse.log
export FAKE_NPM_LOG=$LOG_DIR/fake_npm_reuse.log
VPY="$("$BASE/../../scripts/python" --resolve 2>/dev/null || echo python3)"
API_P=8198
WEB_P=4571
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

health() {
  "$VPY" -c "
import urllib.request
try: print(urllib.request.urlopen('http://localhost:$API_P/health', timeout=1).status)
except Exception: print(0)
"
}

web_state() {
  "$VPY" -c "
import urllib.request
try: print(urllib.request.urlopen('http://localhost:$WEB_P/', timeout=1).status)
except Exception: print(0)
"
}

# Preconditions: both ports free before we plant the foreign API.
for P in $API_P $WEB_P; do
  if [ "$(port_state "$P")" = "OPEN" ]; then
    bad "journey port $P already in use — pick different ports"; exit 1
  fi
done
ok "ports $API_P + $WEB_P free"

# ── Phase 1: startup SIGINT (pre-ready) must stop the CLI ──
LOG1="$LOG_DIR/journey_reuse.phase1.log"
rm -rf apps/web/.next
rm -f "$LOG1" "$FAKE_NPM_LOG"
# Headless `setsid -w`: pty teardown would SIGHUP the group and mask an
# orphaned child before the leftover check; -w gives the exit code
# (0 = handler ran, 130 = bare KeyboardInterrupt escaped).
NVM_DIR=/nonexistent PATH="$BASE/fakebin:$PATH" \
  setsid -w ./sloughgpt serve --port $API_P --web --web-port $WEB_P --host localhost \
  > "$LOG1" 2>&1 < /dev/null &
P1=$!
P1PID=""
for i in $(seq 1 10); do
  P1PID=$(pgrep -f "cli.py .*--web-port $WEB_P" | head -1 || true)
  [ -n "$P1PID" ] && break
  sleep 1
done
if [ -z "$P1PID" ]; then
  bad "phase1: cli never started"
else
  # Wait for the uvicorn child instead of a fixed sleep: the child is
  # spawned inside the command handler, after the entry-point signal
  # registration — so once it exists the handler is guaranteed installed
  # and we are in the pre-ready wait loop (the exact target window).
  # A fixed t+2 races import time under load and can land pre-registration.
  UVREADY=""
  for i in $(seq 1 20); do
    if pgrep -f "uvicorn apps.api.server.main:app.*--port $API_P" >/dev/null 2>&1; then
      UVREADY=$i; break
    fi
    sleep 1
  done
  [ -n "$UVREADY" ] || bad "phase1: uvicorn never spawned within 20s"
  kill -INT "$P1PID" 2>/dev/null
  EXITED=""
  for i in $(seq 1 10); do
    if ! kill -0 "$P1PID" 2>/dev/null; then EXITED=$i; break; fi
    sleep 1
  done
  if [ -n "$EXITED" ]; then
    ok "phase1: startup SIGINT honoured (exit ${EXITED}s)"
    wait "$P1" 2>/dev/null
    EC=$?
    if [ "$EC" = "0" ]; then
      ok "phase1: handler path ran (exit 0; bare 'Interrupted' = 130)"
    else
      bad "phase1: exit $EC after startup SIGINT (130 = handler never ran)"
    fi
  else
    bad "phase1: cli survived startup SIGINT (interrupt silently dropped?)"
    kill -INT "$P1PID" 2>/dev/null
    sleep 2
    kill -9 "$P1PID" 2>/dev/null
  fi
fi
kill -TERM "$P1" 2>/dev/null
sleep 1
for PAT in "cli.py .*--web-port $WEB_P" "uvicorn apps.api.server.main:app.*--port $API_P" "node server.js"; do
  LEFT=$(pgrep -f "$PAT" || true)
  if [ -n "$LEFT" ]; then kill -9 $LEFT 2>/dev/null; bad "phase1 leftover: $PAT"; fi
done
PF="OPEN"
for i in $(seq 1 5); do
  PF="CLOSED"
  for P in $API_P $WEB_P; do
    [ "$(port_state "$P")" = "OPEN" ] && PF="OPEN"
  done
  [ "$PF" = "CLOSED" ] && break
  sleep 1
done
if [ "$PF" = "CLOSED" ]; then ok "phase1: both ports free after startup SIGINT"
else bad "phase1: ports not free for phase2"; fi

# ── Plant a healthy foreign API (simulates the user's existing server) ──
"$VPY" -c "
import http.server

class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{\"status\":\"ok\"}')
    def log_message(self, *a):
        pass

http.server.HTTPServer(('127.0.0.1', $API_P), H).serve_forever()
" > "$LOG_DIR/foreign_api.log" 2>&1 &
FOREIGN=$!

FUP=""
for i in $(seq 1 10); do
  if [ "$(health)" = "200" ]; then FUP=1; break; fi
  sleep 1
done
if [ -n "$FUP" ]; then ok "foreign api healthy on :$API_P"
else bad "foreign api never came up"; kill -9 "$FOREIGN" 2>/dev/null; exit 1; fi

# Fresh build path every run (stale .next would skip the npx build step)
rm -rf apps/web/.next
rm -f "$LOG" "$FAKE_NPM_LOG"

NVM_DIR=/nonexistent PATH="$BASE/fakebin:$PATH" \
  setsid script -qec "./sloughgpt serve --port $API_P --web --web-port $WEB_P --host localhost" /dev/null \
  > "$LOG" 2>&1 < /dev/null &
SPID=$!

# 1) web answers (build + node start; API wait is skipped — api reused)
WEB_T=""
for i in $(seq 1 45); do
  if [ "$(web_state)" = "200" ]; then WEB_T=$i; break; fi
  sleep 1
done
if [ -n "$WEB_T" ]; then ok "web answered on :$WEB_P at t+${WEB_T}s"
else bad "web never answered on :$WEB_P (45s)"; fi

# 2) reuse proof: CLI did NOT spawn its own uvicorn for the api port
SLAIN=$(pgrep -f "uvicorn apps.api.server.main:app.*--port $API_P" || true)
if [ -z "$SLAIN" ]; then ok "no cli-owned uvicorn for :$API_P (reused)"
else bad "cli spawned uvicorn despite healthy api: $SLAIN"; fi

# 3) SIGINT: clean shutdown
CLI_PID=$(pgrep -f "cli.py .*--web-port $WEB_P" | head -1 || true)
if [ -z "$CLI_PID" ]; then bad "cli process not found"
else
  ok "cli alive (pid $CLI_PID)"
  # Land the interrupt inside the monitor loop's sleep(2) so the signal
  # fires between the loop's condition check and its body — the exact
  # window where a missing post-sleep re-check respawns web after cleanup.
  sleep 2
  kill -INT "$CLI_PID" 2>/dev/null
  DEAD=""
  for i in $(seq 1 20); do
    if ! kill -0 "$CLI_PID" 2>/dev/null; then DEAD=$i; break; fi
    sleep 1
  done
  if [ -n "$DEAD" ]; then ok "cli exited after SIGINT (${DEAD}s)"
  else bad "cli still alive 20s after SIGINT"; fi
fi

CLEAN=$LOG_DIR/journey_reuse.clean
sed 's/\x1b\[[0-9;]*[A-Za-z]//g' "$LOG" > "$CLEAN"
if grep -q "Stopped" "$CLEAN"; then ok "shutdown says Stopped"
else bad "no 'Stopped' in shutdown output"; fi

# 3b) no restart-after-shutdown: cleanup already terminated the children,
# so the monitor body must not mistake that for a crash and respawn web.
if grep -q "restarting" "$CLEAN"; then
  bad "web restarted after shutdown"
else ok "no restart after shutdown"; fi

# 4) THE core assertion: the foreign api survived our shutdown (health + pid)
sleep 3
if kill -0 "$FOREIGN" 2>/dev/null && [ "$(health)" = "200" ]; then
  ok "foreign api survived shutdown"
else
  bad "foreign api KILLED by our shutdown"
fi

# 5) port states: api port still held by foreign, web port freed (ours)
if [ "$(port_state "$API_P")" = "OPEN" ]; then ok "api port $API_P still held by foreign"
else bad "api port $API_P closed — foreign server was taken down"; fi

W="OPEN"
for i in $(seq 1 10); do
  W=$(port_state "$WEB_P")
  [ "$W" = "CLOSED" ] && break
  sleep 1
done
if [ "$W" = "CLOSED" ]; then ok "web port $WEB_P freed"
else bad "web port $WEB_P still open"; fi

# 6) no CLI-internal traceback
if grep -q "Traceback" "$CLEAN"; then
  if grep -A8 "Traceback" "$CLEAN" | grep -q "cli.py\|commands/dev.py"; then
    bad "CLI traceback in log"
    grep -A8 "Traceback" "$CLEAN" | head -8 | sed 's/^/  | /'
  else ok "only child-process traceback"; fi
else ok "no traceback"; fi

# ── Reap: our processes only, foreign last ──
kill -TERM "$SPID" 2>/dev/null
sleep 1
for PAT in "cli.py .*--web-port $WEB_P" "node server.js"; do
  LEFT=$(pgrep -f "$PAT" || true)
  if [ -n "$LEFT" ]; then kill -9 $LEFT 2>/dev/null; bad "leftover killed: $PAT"; fi
done
pkill -f "xdg-open.*$WEB_P" 2>/dev/null || true
kill -9 "$FOREIGN" 2>/dev/null
rm -f "$CLEAN"

echo "RESULT[serve-reuse]: $([ $FAIL -eq 0 ] && echo ALL PASS || echo FAILURES)"
exit $FAIL
