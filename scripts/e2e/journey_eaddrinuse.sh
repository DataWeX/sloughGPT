#!/usr/bin/env bash
# E2E journey D: slo dev with a BUSY port must print the remediation hint
# ("port N in use" + kill command) and exit cleanly — not hang, not crash.
# Case 1: API port busy.  Case 2: web port busy.  Self-contained.
set -u
BASE="$(cd "$(dirname "$0")" && pwd)"
WT="${JOURNEY_WT:-$(cd "$BASE/../.." && pwd)}"
LOG_DIR="$WT/logs/e2e"; mkdir -p "$LOG_DIR"
VPY="$("$BASE/../../scripts/python" --resolve 2>/dev/null || echo python3)"
FAIL=0
ok()  { echo "PASS: $*"; }
bad() { echo "FAIL: $*"; FAIL=1; }

cd "$WT" || exit 1

bind_port() {  # $1=port -> echoes pid of a listener held open
  "$VPY" -c "
import socket, time, sys
s = socket.socket()
s.bind(('0.0.0.0', int(sys.argv[1])))
s.listen(5)
time.sleep(600)
" "$1" >/dev/null 2>&1 &
  echo $!
}

port_state() {  # $1=port -> OPEN/CLOSED
  "$VPY" -c "
import socket
s=socket.socket(); s.settimeout(0.5)
try: s.connect(('127.0.0.1',$1)); print('OPEN')
except Exception: print('CLOSED')
"
}

run_case() {  # $1=case label, $2=busy port, $3=api port, $4=web port, $5=expected service
  local LABEL=$1 BUSY=$2 API_P=$3 WEB_P=$4 SVC=$5
  local LOG=$LOG_DIR/journey_${LABEL}.log
  local CLEAN=$LOG_DIR/journey_${LABEL}.clean
  export FAKE_NPM_LOG=$LOG_DIR/fake_npm_${LABEL}.log
  rm -f "$LOG" "$CLEAN" "$FAKE_NPM_LOG"

  local BPID
  BPID=$(bind_port "$BUSY")
  sleep 0.3
  if kill -0 "$BPID" 2>/dev/null; then ok "$LABEL: blocker holds :$BUSY"; else bad "$LABEL: blocker failed to bind :$BUSY"; return; fi

  NVM_DIR=/nonexistent PATH="$BASE/fakebin:$PATH" \
    setsid script -qec "./sloughgpt --port $API_P dev --web-port $WEB_P" /dev/null \
    > "$LOG" 2>&1 < /dev/null &
  local SPID=$!

  # CLI must self-exit with the remediation printed (<=60s: API import ~5-25s)
  local CLI_PID CLI_DEAD=""
  CLI_PID=$(pgrep -f "cli.py --port $API_P dev --web-port $WEB_P" | head -1 || true)
  local i
  for i in $(seq 1 60); do
    if [ -z "$CLI_PID" ]; then
      CLI_PID=$(pgrep -f "cli.py --port $API_P dev --web-port $WEB_P" | head -1 || true)
    elif ! kill -0 "$CLI_PID" 2>/dev/null; then
      CLI_DEAD=$i; break
    fi
    sleep 1
  done
  if [ -n "$CLI_DEAD" ]; then ok "$LABEL: cli exited on its own (${CLI_DEAD}s)"
  else bad "$LABEL: cli still alive 60s after start"; fi

  # ANSI-free view of the log for stable greps
  sed 's/\x1b\[[0-9;]*[A-Za-z]//g' "$LOG" > "$CLEAN"

  if grep -q "port $BUSY in use" "$CLEAN"; then ok "$LABEL: remediation says 'port $BUSY in use'"
  else bad "$LABEL: no 'port $BUSY in use' in log"; tail -5 "$CLEAN" | sed 's/^/  | /'; fi
  if grep -q "xargs kill -9" "$CLEAN"; then ok "$LABEL: kill hint printed"
  else bad "$LABEL: no kill hint"; fi
  if grep -q "$SVC server exited" "$CLEAN"; then bad "$LABEL: generic death message used for port conflict"
  else ok "$LABEL: generic death message suppressed"; fi
  # Traceback check: only CLI-internal frames are a bug — child processes
  # legitimately dump their own EADDRINUSE traceback into the log.
  if grep -q "Traceback" "$CLEAN"; then
    if grep -A8 "Traceback" "$CLEAN" | grep -q "cli.py\|commands/dev.py"; then
      bad "$LABEL: CLI traceback in log"
      grep -A8 "Traceback" "$CLEAN" | head -8 | sed 's/^/  | /'
    else
      ok "$LABEL: only child-process traceback (expected when a port is taken)"
    fi
  else ok "$LABEL: no traceback"; fi

  # ports freed by cleanup (CLI's _kill_port also releases the blocker)
  kill -9 "$BPID" 2>/dev/null
  local C1 C2
  for i in $(seq 1 10); do
    C1=$(port_state "$BUSY"); C2=$(port_state "$API_P")
    [ "$C1" = "CLOSED" ] && [ "$C2" = "CLOSED" ] && break
    sleep 1
  done
  [ "$C1" = "CLOSED" ] && ok "$LABEL: busy port $BUSY freed" || bad "$LABEL: busy port $BUSY still open"
  [ "$C2" = "CLOSED" ] && ok "$LABEL: api port $API_P freed" || bad "$LABEL: api port $API_P still open"
  C1=$(port_state "$WEB_P")
  [ "$C1" = "CLOSED" ] && ok "$LABEL: web port $WEB_P freed" || bad "$LABEL: web port $WEB_P still open"

  kill -TERM "$SPID" 2>/dev/null
  sleep 1
  local LEFT
  LEFT=$(pgrep -f "cli.py --port $API_P dev --web-port $WEB_P" || true)
  if [ -n "$LEFT" ]; then kill -9 $LEFT 2>/dev/null; bad "$LABEL: leftover cli killed"; fi
  LEFT=$(pgrep -f "uvicorn apps.api.server.main:app.*--port $API_P" || true)
  if [ -n "$LEFT" ]; then kill -9 $LEFT 2>/dev/null; bad "$LABEL: leftover uvicorn killed"; fi
  rm -f "$CLEAN"
}

run_case apibusy 8177 8177 4594 API
run_case webbusy 4595 8177 4595 Web

echo "RESULT[eaddrinuse]: $([ $FAIL -eq 0 ] && echo ALL PASS || echo FAILURES)"
exit $FAIL
