#!/usr/bin/env bash
# Host the edge gateway for real downloads (no Docker).
#
#   sidecar (files) → gateway :8080 → client
#
# Usage:
#   scripts/host_gateway.sh              # serve repo root files on :8080
#   ROOT=/path/to/models scripts/host_gateway.sh
#   PORT=9090 scripts/host_gateway.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-$REPO/.venv/bin/python}"
GW_BIN="${GW_BIN:-$REPO/apps/gateway/target/release/slough-gateway}"
GW_PORT="${PORT:-8080}"
SIDECAR_PORT="${SIDECAR_PORT:-18000}"
# App-edge mode: skip the file sidecar, point the core wherever you want:
#   CORE_URL=http://127.0.0.1:8000 NO_SIDECAR=1 PORT=8081 scripts/host_gateway.sh
NO_SIDECAR="${NO_SIDECAR:-0}"
CORE_URL="${CORE_URL:-http://127.0.0.1:$SIDECAR_PORT}"
ROOT="${ROOT:-$REPO}"
HOST="${HOST:-0.0.0.0}"
LOG_DIR="${LOG_DIR:-/tmp/slough-gateway-host}"
mkdir -p "$LOG_DIR"

if [ ! -x "$PY" ]; then
  PY="$(command -v python3)"
fi

if [ ! -x "$GW_BIN" ]; then
  echo "building gateway release binary..."
  (cd "$REPO/apps/gateway" && cargo build --release --locked)
fi

# Free ports if leftover (sidecar port only when we own it)
PORTS_TO_FREE="$GW_PORT"
if [ "$NO_SIDECAR" != "1" ]; then
  PORTS_TO_FREE="$PORTS_TO_FREE $SIDECAR_PORT"
fi
for p in $PORTS_TO_FREE; do
  if command -v fuser >/dev/null 2>&1; then
    fuser -k "${p}/tcp" 2>/dev/null || true
  fi
done
sleep 0.2

cleanup() {
  [ -n "${GW_PID:-}" ] && kill "$GW_PID" 2>/dev/null || true
  [ -n "${SIDECAR_PID:-}" ] && kill "$SIDECAR_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

if [ "$NO_SIDECAR" != "1" ]; then
  echo "sidecar root=$ROOT → 127.0.0.1:$SIDECAR_PORT"
  "$PY" "$REPO/scripts/gateway_file_sidecar.py" \
    --root "$ROOT" --host 127.0.0.1 --port "$SIDECAR_PORT" \
    >"$LOG_DIR/sidecar.log" 2>&1 &
  SIDECAR_PID=$!

  for _ in $(seq 1 40); do
    # sidecar up? (any response including 404)
    if curl -s -o /dev/null -m 1 "http://127.0.0.1:$SIDECAR_PORT/" 2>/dev/null; then
      break
    fi
    if ! kill -0 "$SIDECAR_PID" 2>/dev/null; then
      echo "sidecar died — see $LOG_DIR/sidecar.log" >&2
      cat "$LOG_DIR/sidecar.log" >&2 || true
      exit 1
    fi
    sleep 0.1
  done
fi

echo "gateway → $HOST:$GW_PORT (core=$CORE_URL)"
MAN_CORE_URL="$CORE_URL" \
MAN_GATEWAY_PORT="$GW_PORT" \
MAN_GATEWAY_SUPERVISE="${MAN_GATEWAY_SUPERVISE:-1}" \
RUST_LOG="${RUST_LOG:-slough_gateway=info}" \
"$GW_BIN" >"$LOG_DIR/gateway.log" 2>&1 &
GW_PID=$!

ok=0
for _ in $(seq 1 60); do
  if curl -sf "http://127.0.0.1:$GW_PORT/health" >/dev/null 2>&1; then
    ok=1
    break
  fi
  sleep 0.25
done
if [ "$ok" != "1" ]; then
  echo "gateway failed to become healthy — see $LOG_DIR/gateway.log" >&2
  cat "$LOG_DIR/gateway.log" >&2 || true
  exit 1
fi

IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
echo ""
echo "=== gateway ONLINE ==="
echo "health:     http://127.0.0.1:$GW_PORT/health"
echo "LAN:        http://${IP:-<lan-ip>}:$GW_PORT/health"
echo "file root:  $ROOT"
echo "example:    curl -H 'Accept-Encoding: zstd' -o out.bin -D - \\"
echo "              http://127.0.0.1:$GW_PORT/checkpoints/step_25.pt"
echo "logs:       $LOG_DIR/{gateway,sidecar}.log"
echo ""
echo "Ctrl+C to stop (or leave running with nohup / systemd)."
echo ""

# Keep foreground so trap cleans up; stream gateway log tail
tail -n +1 -f "$LOG_DIR/gateway.log" &
TAIL_PID=$!
wait "$GW_PID"
