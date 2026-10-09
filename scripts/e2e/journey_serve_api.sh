#!/usr/bin/env bash
# E2E journey S2: `slo serve` (API-only, no --web) —
#   Phase A: SIGINT during the readiness wait must run the registered
#            handler's cleanup (was: registered after the wait — the
#            interrupt escaped as bare "Interrupted" and orphaned uvicorn).
#   Phase B: SIGINT after ready (main thread in proc.wait) → "Stopped",
#            spawned uvicorn terminated, port freed.
#   Phase C: reuse a healthy foreign API → SIGINT → foreign survives
#            (cleanup owns only ports we spawned) and no cli uvicorn ever.
set -u
BASE="$(cd "$(dirname "$0")" && pwd)"
WT="${JOURNEY_WT:-$(cd "$BASE/../.." && pwd)}"
LOG_DIR="$WT/logs/e2e"; mkdir -p "$LOG_DIR"
LOG=$LOG_DIR/journey_serve_api.log
VPY="$("$BASE/../../scripts/python" --resolve 2>/dev/null || echo python3)"
API_P=8197
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

# Preconditions: the api-only port must be free — serve REUSES a healthy
# service on it and would never spawn, breaking every phase.
if [ "$(port_state "$API_P")" = "OPEN" ]; then
  bad "journey port $API_P already in use — pick a different port"; exit 1
fi
ok "port $API_P free"

# ── Phase A: startup SIGINT (pre-ready, wait loop) ──
LOGA="$LOG_DIR/journey_serve_api.phaseA.log"
rm -f "$LOGA"
# Headless `setsid -w` (NOT `script`): a pty leader's exit SIGHUPs the
# group and would kill an orphaned uvicorn before the orphan check runs,
# masking the leak this phase exists to catch. -w propagates the CLI's
# exit code: 0 = our handler ran; 130 = bare KeyboardInterrupt escaped
# to main() (handler never registered).
setsid -w ./sloughgpt serve --port $API_P --host localhost \
  > "$LOGA" 2>&1 < /dev/null &
SPA=$!
APID=""
for i in $(seq 1 10); do
  APID=$(pgrep -f "cli.py serve --port $API_P" | head -1 || true)
  [ -n "$APID" ] && break
  sleep 1
done
if [ -z "$APID" ]; then
  bad "phaseA: cli never started"
else
  # Wait for the uvicorn child instead of a fixed sleep: it is spawned
  # inside the command handler after signal registration, so its presence
  # proves the handler ran and we are inside the pre-ready wait loop —
  # the exact window where a missing early handler orphans the child.
  UVREADY=""
  for i in $(seq 1 20); do
    if pgrep -f "uvicorn apps.api.server.main:app.*--port $API_P" >/dev/null 2>&1; then
      UVREADY=$i; break
    fi
    sleep 1
  done
  [ -n "$UVREADY" ] || bad "phaseA: uvicorn never spawned within 20s"
  kill -INT "$APID" 2>/dev/null
  EXITED=""
  for i in $(seq 1 10); do
    if ! kill -0 "$APID" 2>/dev/null; then EXITED=$i; break; fi
    sleep 1
  done
  if [ -n "$EXITED" ]; then
    ok "phaseA: startup SIGINT honoured (exit ${EXITED}s)"
    wait "$SPA" 2>/dev/null
    EC=$?
    if [ "$EC" = "0" ]; then
      ok "phaseA: handler path ran (exit 0; bare 'Interrupted' = 130)"
    else
      bad "phaseA: exit $EC after startup SIGINT (130 = handler never ran)"
    fi
  else
    bad "phaseA: cli survived startup SIGINT (interrupt dropped?)"
    kill -INT "$APID" 2>/dev/null
    sleep 2
    kill -9 "$APID" 2>/dev/null
  fi
fi
kill -TERM "$SPA" 2>/dev/null
sleep 1
LEFT=$(pgrep -f "uvicorn apps.api.server.main:app.*--port $API_P" || true)
if [ -n "$LEFT" ]; then kill -9 $LEFT 2>/dev/null; bad "phaseA orphan uvicorn: $LEFT"
else ok "phaseA: no orphan uvicorn after startup SIGINT"; fi
if [ "$(port_state "$API_P")" = "CLOSED" ]; then ok "phaseA: port $API_P freed"
else bad "phaseA: port $API_P still open after startup SIGINT"; fi

# ── Phase B: SIGINT after ready (spawned API, main thread in proc.wait) ──
LOGB="$LOG_DIR/journey_serve_api.phaseB.log"
rm -f "$LOGB"
setsid script -qec "./sloughgpt serve --port $API_P --host localhost" /dev/null \
  > "$LOGB" 2>&1 < /dev/null &
SPB=$!
API_T=""
for i in $(seq 1 75); do
  if [ "$(health)" = "200" ]; then API_T=$i; break; fi
  sleep 1
done
if [ -n "$API_T" ]; then ok "phaseB: api healthy on :$API_P at t+${API_T}s"
else bad "phaseB: api never healthy (75s)"; fi

BPID=$(pgrep -f "cli.py serve --port $API_P" | head -1 || true)
if [ -z "$BPID" ]; then
  bad "phaseB: cli process not found"
else
  sleep 1
  kill -INT "$BPID" 2>/dev/null
  DEAD=""
  for i in $(seq 1 10); do
    if ! kill -0 "$BPID" 2>/dev/null; then DEAD=$i; break; fi
    sleep 1
  done
  if [ -n "$DEAD" ]; then ok "phaseB: SIGINT after ready → exit (${DEAD}s)"
  else bad "phaseB: cli alive 10s after SIGINT on ready session"; fi
fi
CLEANB="$LOG_DIR/journey_serve_api.cleanB"
sed 's/\x1b\[[0-9;]*[A-Za-z]//g' "$LOGB" > "$CLEANB"
if grep -q "Stopped" "$CLEANB"; then ok "phaseB: shutdown says Stopped"
else bad "phaseB: no 'Stopped' in shutdown output"; fi

L=""
for i in $(seq 1 10); do
  L=$(pgrep -f "uvicorn apps.api.server.main:app.*--port $API_P" || true)
  [ -z "$L" ] && break
  sleep 1
done
if [ -z "$L" ]; then ok "phaseB: spawned uvicorn terminated"
else bad "phaseB: uvicorn survived SIGINT: $L"; kill -9 $L 2>/dev/null; fi
P="OPEN"
for i in $(seq 1 10); do
  P=$(port_state "$API_P")
  [ "$P" = "CLOSED" ] && break
  sleep 1
done
if [ "$P" = "CLOSED" ]; then ok "phaseB: port $API_P freed"
else bad "phaseB: port $API_P still open"; fi
kill -TERM "$SPB" 2>/dev/null
sleep 1

# ── Phase C: reuse a healthy foreign API; SIGINT must leave it alive ──
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
" > "$LOG_DIR/foreign_api_s2.log" 2>&1 &
FOREIGN=$!
FUP=""
for i in $(seq 1 10); do
  if [ "$(health)" = "200" ]; then FUP=1; break; fi
  sleep 1
done
if [ -n "$FUP" ]; then ok "phaseC: foreign api healthy on :$API_P"
else bad "phaseC: foreign api never came up"; kill -9 "$FOREIGN" 2>/dev/null; exit 1; fi

LOGC="$LOG_DIR/journey_serve_api.phaseC.log"
rm -f "$LOGC"
setsid script -qec "./sloughgpt serve --port $API_P --host localhost" /dev/null \
  > "$LOGC" 2>&1 < /dev/null &
SPC=$!
CPID=""
for i in $(seq 1 10); do
  CPID=$(pgrep -f "cli.py serve --port $API_P" | head -1 || true)
  [ -n "$CPID" ] && break
  sleep 1
done
if [ -z "$CPID" ]; then bad "phaseC: cli never started"
else ok "phaseC: cli alive on reused api (pid $CPID)"; fi

# reuse proof: the CLI must not have spawned its own uvicorn
OWN=$(pgrep -f "uvicorn apps.api.server.main:app.*--port $API_P" || true)
if [ -z "$OWN" ]; then ok "phaseC: no cli-owned uvicorn (api reused)"
else bad "phaseC: cli spawned uvicorn despite healthy api: $OWN"; fi

sleep 2
kill -INT "$CPID" 2>/dev/null
DEADC=""
for i in $(seq 1 10); do
  if ! kill -0 "$CPID" 2>/dev/null; then DEADC=$i; break; fi
  sleep 1
done
if [ -n "$DEADC" ]; then ok "phaseC: cli exited after SIGINT (${DEADC}s)"
else bad "phaseC: cli alive 10s after SIGINT on reused session"; fi

CLEANC="$LOG_DIR/journey_serve_api.cleanC"
sed 's/\x1b\[[0-9;]*[A-Za-z]//g' "$LOGC" > "$CLEANC"
if grep -q "Stopped" "$CLEANC"; then ok "phaseC: shutdown says Stopped"
else bad "phaseC: no 'Stopped' in shutdown output"; fi

sleep 2
if kill -0 "$FOREIGN" 2>/dev/null && [ "$(health)" = "200" ]; then
  ok "phaseC: foreign api survived our shutdown"
else
  bad "phaseC: foreign api KILLED by our shutdown"
fi
if [ "$(port_state "$API_P")" = "OPEN" ]; then ok "phaseC: port $API_P still held by foreign"
else bad "phaseC: port $API_P closed — foreign taken down"; fi

# ── Global: no CLI-internal traceback in any phase ──
TB=0
for F in "$LOGA" "$CLEANB" "$CLEANC"; do
  if grep -q "Traceback" "$F" 2>/dev/null; then
    if grep -A8 "Traceback" "$F" | grep -q "cli.py\|commands/dev.py"; then
      bad "CLI traceback in $F"
      grep -A8 "Traceback" "$F" | head -8 | sed 's/^/  | /'
      TB=1
    fi
  fi
done
[ "$TB" -eq 0 ] && ok "no CLI-internal tracebacks"

# ── Reap: our processes first, foreign last ──
kill -TERM "$SPA" "$SPB" "$SPC" 2>/dev/null
sleep 1
for PAT in "cli.py serve --port $API_P" "uvicorn apps.api.server.main:app.*--port $API_P"; do
  L=$(pgrep -f "$PAT" || true)
  if [ -n "$L" ]; then kill -9 $L 2>/dev/null; bad "leftover after reap: $PAT"; fi
done
kill -9 "$FOREIGN" 2>/dev/null
rm -f "$CLEANB" "$CLEANC"

echo "RESULT[serve-api]: $([ $FAIL -eq 0 ] && echo ALL PASS || echo FAILURES)"
exit $FAIL
