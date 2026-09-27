#!/usr/bin/env bash
# E2E journey M: `slo serve --mobile` (API + metro bundler on 8081) —
#   Phase A: SIGINT during startup (pre-ready wait) → honoured, no orphan
#            uvicorn/metro, ports 8196+8081 freed.
#   Phase B: SIGINT after ready (monitor loop) → "Stopped", children
#            terminated, no restart-after-shutdown (metro respawn race).
#   Phase C: reuse healthy foreign API → SIGINT → foreign survives,
#            metro still cleaned up.
# Metro runs through scripts/e2e/fakebin/npx (`react-native start` →
# http.server on $PORT) so readiness works without a real RN toolchain.
set -u
BASE="$(cd "$(dirname "$0")" && pwd)"
WT="${JOURNEY_WT:-$(cd "$BASE/../.." && pwd)}"
LOG_DIR="$WT/logs/e2e"; mkdir -p "$LOG_DIR"
export FAKE_NPM_LOG=$LOG_DIR/fake_npx_mobile.log
VPY="$WT/.venv/bin/python"
API_P=8196
METRO_P=8081
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

CLI_PAT="cli.py serve --mobile --port $API_P"
UV_PAT="uvicorn apps.api.server.main:app.*--port $API_P"
# The fake npx execs straight into `python3 -m http.server $PORT` (signal-
# clean: SIGTERM lands on the server directly), which replaces its cmdline —
# match the post-exec identity, scoped to metro's fixed port 8081.
METRO_PAT="http.server $METRO_P"

# Preconditions: both ports free (reuse would skip the spawn paths entirely).
for P in $API_P $METRO_P; do
  if [ "$(port_state "$P")" = "OPEN" ]; then
    bad "journey port $P already in use — pick different ports"; exit 1
  fi
done
ok "ports $API_P + $METRO_P free"

launch() {
  NVM_DIR=/nonexistent PATH="$BASE/fakebin:$PATH" \
    setsid script -qec "./sloughgpt serve --mobile --port $API_P --host localhost" /dev/null \
    > "$1" 2>&1 < /dev/null &
}

# ── Phase A: startup SIGINT (pre-ready) ──
LOGA="$LOG_DIR/journey_m.phaseA.log"
rm -f "$LOGA" "$FAKE_NPM_LOG"
launch "$LOGA"
SPA=$!
APID=""
UVID=""
METPID=""
for i in $(seq 1 15); do
  APID=$(pgrep -f "$CLI_PAT" | head -1 || true)
  UVID=$(pgrep -f "$UV_PAT" | head -1 || true)
  METPID=$(pgrep -f "$METRO_PAT" | head -1 || true)
  [ -n "$APID" ] && [ -n "$UVID" ] && [ -n "$METPID" ] && break
  sleep 1
done
if [ -z "$APID" ] || [ -z "$UVID" ] || [ -z "$METPID" ]; then
  bad "phaseA: startup incomplete (cli=$APID uvicorn=$UVID metro=$METPID)"
else
  # All three visible → handlers installed and both children spawned:
  # interrupt lands in the API-readiness wait.
  sleep 1
  kill -INT "$APID" 2>/dev/null
  EXITED=""
  for i in $(seq 1 10); do
    if ! kill -0 "$APID" 2>/dev/null; then EXITED=$i; break; fi
    sleep 1
  done
  if [ -n "$EXITED" ]; then
    ok "phaseA: startup SIGINT honoured (exit ${EXITED}s)"
    if grep -qE "Stopped|Interrupted" <(sed $'s/\x1b\[[0-9;]*[A-Za-z]//g' "$LOGA"); then
      ok "phaseA: shutdown acknowledged (Stopped/Interrupted)"
    else
      bad "phaseA: no shutdown acknowledgement"
    fi
  else
    bad "phaseA: cli survived startup SIGINT"
    kill -INT "$APID" 2>/dev/null; sleep 2; kill -9 "$APID" 2>/dev/null
  fi
fi
kill -TERM "$SPA" 2>/dev/null
sleep 1
ORPHA=""
ORPH_PIDS=""
for PAT in "$CLI_PAT" "$UV_PAT" "$METRO_PAT"; do
  L=$(pgrep -f "$PAT" || true)
  if [ -n "$L" ]; then ORPHA="$ORPHA $PAT=[$L]"; ORPH_PIDS="$ORPH_PIDS $L"; fi
done
if [ -z "$ORPHA" ]; then ok "phaseA: no orphan children (cli/uvicorn/metro)"
else kill -9 $ORPH_PIDS 2>/dev/null; bad "phaseA orphan:$ORPHA"; fi
for P in $API_P $METRO_P; do
  S="OPEN"
  for i in $(seq 1 5); do
    S=$(port_state "$P"); [ "$S" = "CLOSED" ] && break; sleep 1
  done
  if [ "$S" = "CLOSED" ]; then ok "phaseA: port $P freed"
  else bad "phaseA: port $P still open"; fi
done

# ── Phase B: SIGINT after ready (monitor loop) ──
LOGB="$LOG_DIR/journey_m.phaseB.log"
rm -f "$LOGB" "$FAKE_NPM_LOG"
launch "$LOGB"
SPB=$!
API_T=""
for i in $(seq 1 75); do
  if [ "$(health)" = "200" ]; then API_T=$i; break; fi
  sleep 1
done
if [ -n "$API_T" ]; then ok "phaseB: api healthy at t+${API_T}s"
else bad "phaseB: api never healthy (75s)"; fi
M_T=""
for i in $(seq 1 20); do
  if [ "$(port_state "$METRO_P")" = "OPEN" ]; then M_T=$i; break; fi
  sleep 1
done
if [ -n "$M_T" ]; then ok "phaseB: metro listening on :$METRO_P (+${M_T}s)"
else bad "phaseB: metro never bound :$METRO_P (20s)"; fi

BPID=$(pgrep -f "$CLI_PAT" | head -1 || true)
if [ -z "$BPID" ]; then
  bad "phaseB: cli not found"
else
  # Land inside the monitor loop's sleep(2) — the window where a missing
  # post-sleep re-check mistakes cleanup's exits for crashes and respawns.
  sleep 2
  kill -INT "$BPID" 2>/dev/null
  DEAD=""
  for i in $(seq 1 12); do
    if ! kill -0 "$BPID" 2>/dev/null; then DEAD=$i; break; fi
    sleep 1
  done
  if [ -n "$DEAD" ]; then ok "phaseB: SIGINT after ready → exit (${DEAD}s)"
  else bad "phaseB: cli alive 12s after SIGINT"; fi
fi
CLEANB="$LOG_DIR/journey_m.cleanB"
sed 's/\x1b\[[0-9;]*[A-Za-z]//g' "$LOGB" > "$CLEANB"
if grep -q "Stopped" "$CLEANB"; then ok "phaseB: shutdown says Stopped"
else bad "phaseB: no 'Stopped' in output"; fi
if grep -q "restarting" "$CLEANB"; then
  bad "phaseB: metro restarted after shutdown"
else ok "phaseB: no restart after shutdown"; fi
L=""
for i in $(seq 1 10); do
  L=$(pgrep -f "$UV_PAT" || true); L="$L $(pgrep -f "$METRO_PAT" || true)"
  L=$(echo "$L" | tr -d ' ')
  [ -z "$L" ] && break
  sleep 1
done
if [ -z "$L" ]; then ok "phaseB: children terminated (uvicorn+metro)"
else bad "phaseB: children survived: $L"; kill -9 $L 2>/dev/null; fi
for P in $API_P $METRO_P; do
  S="OPEN"
  for i in $(seq 1 10); do
    S=$(port_state "$P"); [ "$S" = "CLOSED" ] && break; sleep 1
  done
  if [ "$S" = "CLOSED" ]; then ok "phaseB: port $P freed"
  else bad "phaseB: port $P still open"; fi
done
kill -TERM "$SPB" 2>/dev/null
sleep 1

# ── Phase C: reuse healthy foreign API ──
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
" > "$LOG_DIR/foreign_api_m.log" 2>&1 &
FOREIGN=$!
FUP=""
for i in $(seq 1 10); do
  if [ "$(health)" = "200" ]; then FUP=1; break; fi
  sleep 1
done
if [ -n "$FUP" ]; then ok "phaseC: foreign api healthy on :$API_P"
else bad "phaseC: foreign api never came up"; kill -9 "$FOREIGN" 2>/dev/null; exit 1; fi

LOGC="$LOG_DIR/journey_m.phaseC.log"
rm -f "$LOGC" "$FAKE_NPM_LOG"
launch "$LOGC"
SPC=$!
CPID=""
for i in $(seq 1 15); do
  CPID=$(pgrep -f "$CLI_PAT" | head -1 || true)
  [ -n "$CPID" ] && break
  sleep 1
done
if [ -z "$CPID" ]; then bad "phaseC: cli never started"
else ok "phaseC: cli alive on reused api (pid $CPID)"; fi

OWN=$(pgrep -f "$UV_PAT" || true)
if [ -z "$OWN" ]; then ok "phaseC: no cli-owned uvicorn (api reused)"
else bad "phaseC: cli spawned uvicorn despite healthy api: $OWN"; fi
CM=""
for i in $(seq 1 20); do
  if [ "$(port_state "$METRO_P")" = "OPEN" ]; then CM=1; break; fi
  sleep 1
done
if [ -n "$CM" ]; then ok "phaseC: metro up alongside reused api"
else bad "phaseC: metro never bound :$METRO_P"; fi

sleep 2
kill -INT "$CPID" 2>/dev/null
DEADC=""
for i in $(seq 1 12); do
  if ! kill -0 "$CPID" 2>/dev/null; then DEADC=$i; break; fi
  sleep 1
done
if [ -n "$DEADC" ]; then ok "phaseC: cli exited after SIGINT (${DEADC}s)"
else bad "phaseC: cli alive 12s after SIGINT"; fi
CLEANC="$LOG_DIR/journey_m.cleanC"
sed 's/\x1b\[[0-9;]*[A-Za-z]//g' "$LOGC" > "$CLEANC"
if grep -q "Stopped" "$CLEANC"; then ok "phaseC: shutdown says Stopped"
else bad "phaseC: no 'Stopped' in output"; fi

sleep 2
if kill -0 "$FOREIGN" 2>/dev/null && [ "$(health)" = "200" ]; then
  ok "phaseC: foreign api survived our shutdown"
else
  bad "phaseC: foreign api KILLED by our shutdown"
fi
if [ "$(port_state "$API_P")" = "OPEN" ]; then ok "phaseC: port $API_P still held by foreign"
else bad "phaseC: port $API_P closed — foreign taken down"; fi
SM="OPEN"
for i in $(seq 1 10); do
  SM=$(port_state "$METRO_P"); [ "$SM" = "CLOSED" ] && break; sleep 1
done
if [ "$SM" = "CLOSED" ]; then ok "phaseC: metro port $METRO_P freed"
else bad "phaseC: metro port $METRO_P still open"; fi

# ── Global: no CLI-internal traceback ──
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

# ── Reap ──
kill -TERM "$SPA" "$SPB" "$SPC" 2>/dev/null
sleep 1
for PAT in "$CLI_PAT" "$UV_PAT" "$METRO_PAT"; do
  L=$(pgrep -f "$PAT" || true)
  if [ -n "$L" ]; then kill -9 $L 2>/dev/null; bad "leftover after reap: $PAT"; fi
done
kill -9 "$FOREIGN" 2>/dev/null
rm -f "$CLEANB" "$CLEANC"

echo "RESULT[serve-mobile]: $([ $FAIL -eq 0 ] && echo ALL PASS || echo FAILURES)"
exit $FAIL
