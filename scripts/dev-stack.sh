#!/usr/bin/env bash
# dev-stack.sh — canonical dev launcher for SloughGPT API + Web.
#
# Features:
#   - Colored console output with [api] / [web] prefixes
#   - Auto-restart on crash with exponential backoff (max 5 retries)
#   - Clean shutdown on Ctrl+C (SIGTERM to children)
#   - Web auto-restarts if it dies independently
#
# Usage:
#   ./scripts/dev-stack.sh              # both API (:8000) + Web (:5173)
#   SLO_API_PORT=9000 ./scripts/dev-stack.sh  # custom API port
#   MAN_DEV_GATEWAY=1 ./scripts/dev-stack.sh  # also start Rust edge (:8080)
#
# Replaces: npm run dev:stack (concurrently). This script is the single entrypoint.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# ── Config ─────────────────────────────────────────────────────
export SLO_API_PORT="${SLO_API_PORT:-8000}"
WEB_PORT="${WEB_PORT:-5173}"
MAN_GATEWAY_PORT="${MAN_GATEWAY_PORT:-8080}"
MAN_DEV_GATEWAY="${MAN_DEV_GATEWAY:-0}"
MAX_RETRIES=5
INITIAL_BACKOFF=2

# ── Colors ─────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
DIM='\033[2m'
BOLD='\033[1m'
RESET='\033[0m'

# ── State ──────────────────────────────────────────────────────
API_PID=""
WEB_PID=""
GW_PID=""
STOPPING=0
API_RESTARTS=0
WEB_RESTARTS=0
GW_RESTARTS=0

# ── Cleanup ────────────────────────────────────────────────────
cleanup() {
  STOPPING=1
  # Kill children in reverse order
  for pid in "$GW_PID" "$WEB_PID" "$API_PID"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
  # Wait for them to exit (max 5s each)
  for pid in "$GW_PID" "$WEB_PID" "$API_PID"; do
    if [[ -n "$pid" ]]; then
      for _ in $(seq 1 10); do
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.5
      done
      kill -9 "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
  echo ""
  echo -e "${DIM}Services stopped.${RESET}"
}
trap cleanup EXIT INT TERM

# ── Helpers ────────────────────────────────────────────────────
log_api()  { echo -e "${CYAN}[api]${RESET}  $*"; }
log_web()  { echo -e "${MAGENTA}[web]${RESET}  $*"; }
log_gw()   { echo -e "${GREEN}[gw]${RESET}   $*"; }
log_ok()   { echo -e "  ${GREEN}✓${RESET} $*"; }
log_warn() { echo -e "  ${YELLOW}⚠${RESET} $*"; }
log_err()  { echo -e "  ${RED}✗${RESET} $*"; }

wait_for_health() {
  local port=$1 label=$2
  for _ in $(seq 1 40); do
    if curl -sf "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

# Pipe a subprocess's stdout/stderr through a prefix function.
# Usage: prefix_output "api" "$PID" < <(cmd)
prefix_output() {
  local label=$1
  while IFS= read -r line; do
    case "$label" in
      api) echo -e "${CYAN}[api]${RESET}  ${DIM}${line}${RESET}" ;;
      web) echo -e "${MAGENTA}[web]${RESET}  ${DIM}${line}${RESET}" ;;
    esac
  done
}

# ── Start API ──────────────────────────────────────────────────
start_api() {
  log_api "Starting (port $SLO_API_PORT)..."
   local py="python3"
   if [ -x ".venv/bin/python3" ]; then
     py=".venv/bin/python3"
   fi
   $py apps/api/server/main.py 2>&1 &
  API_PID=$!

  if wait_for_health "$SLO_API_PORT" "api"; then
    log_ok "API healthy on :$SLO_API_PORT"
    return 0
  fi

  # Health check failed — did the process die?
  if ! kill -0 "$API_PID" 2>/dev/null; then
    wait "$API_PID" 2>/dev/null
    local code=$?
    log_err "API exited with code $code before becoming healthy"
    return 1
  fi

  log_err "API did not become healthy within 40s"
  kill -TERM "$API_PID" 2>/dev/null || true
  wait "$API_PID" 2>/dev/null || true
  return 1
}

# ── Start Web ──────────────────────────────────────────────────
start_web() {
  log_api "Starting web (port $WEB_PORT)..."
  (cd "$ROOT/apps/web" && exec npm run dev 2>&1) &
  WEB_PID=$!
  # Give Next.js a moment to bind
  sleep 3
  if kill -0 "$WEB_PID" 2>/dev/null; then
    log_ok "Web dev server on :$WEB_PORT"
    return 0
  fi
  log_err "Web process failed to start"
  return 1
}

# ── Start Gateway (opt-in: MAN_DEV_GATEWAY=1) ─────────────────
start_gateway() {
  local bin="$ROOT/apps/gateway/target/release/slough-gateway"
  if [[ ! -x "$bin" ]]; then
    log_warn "Gateway binary missing at $bin — run: cargo build --release --locked -p slough-gateway"
    return 1
  fi
  log_gw "Starting (port $MAN_GATEWAY_PORT)..."
  MAN_GATEWAY_PORT="$MAN_GATEWAY_PORT" \
  MAN_CORE_URL="http://127.0.0.1:${SLO_API_PORT}" \
  "$bin" 2>&1 &
  GW_PID=$!
  for _ in $(seq 1 10); do
    if curl -sf "http://127.0.0.1:${MAN_GATEWAY_PORT}/health" >/dev/null 2>&1; then
      log_ok "Gateway on :$MAN_GATEWAY_PORT → API :$SLO_API_PORT"
      return 0
    fi
    kill -0 "$GW_PID" 2>/dev/null || break
    sleep 0.5
  done
  if ! kill -0 "$GW_PID" 2>/dev/null; then
    log_err "Gateway exited before healthy"
    return 1
  fi
  log_warn "Gateway not healthy within 5s (may still be starting)"
  return 0
}

# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
echo ""
echo -e "${BOLD}SloughGPT Dev Stack${RESET}"
echo -e "${DIM}API: http://localhost:$SLO_API_PORT  |  Web: http://localhost:$WEB_PORT${RESET}"
if [[ "$MAN_DEV_GATEWAY" == "1" ]]; then
  echo -e "${DIM}Gateway: http://localhost:$MAN_GATEWAY_PORT${RESET}"
fi
echo ""

# ── Phase 1: Start API (with retry) ──
attempt=0
backoff=$INITIAL_BACKOFF
while [[ $attempt -lt $MAX_RETRIES ]]; do
  [[ $STOPPING -eq 1 ]] && break
  start_api && break
  attempt=$((attempt + 1))
  if [[ $attempt -ge $MAX_RETRIES ]]; then
    log_err "API failed after $MAX_RETRIES attempts. Exiting."
    exit 1
  fi
  log_warn "Restarting in ${backoff}s... (attempt $attempt/$MAX_RETRIES)"
  sleep "$backoff"
  backoff=$((backoff * 2))
done

# ── Phase 2: Start Web ──
start_web || true

# ── Phase 3: Optional gateway edge ──
if [[ "$MAN_DEV_GATEWAY" == "1" ]]; then
  start_gateway || true
fi

echo ""
echo -e "${BOLD}All services ready!${RESET}"
echo -e "  ${DIM}API:${RESET}  http://localhost:$SLO_API_PORT"
echo -e "  ${DIM}Docs:${RESET} http://localhost:$SLO_API_PORT/docs"
echo -e "  ${DIM}Web:${RESET}  http://localhost:$WEB_PORT"
if [[ -n "$GW_PID" ]] && kill -0 "$GW_PID" 2>/dev/null; then
  echo -e "  ${DIM}Edge:${RESET}  http://localhost:$MAN_GATEWAY_PORT"
fi
echo -e "  ${DIM}Press Ctrl+C to stop${RESET}"
echo ""

# ── Watch loop: restart crashed processes ──
while [[ $STOPPING -eq 0 ]]; do
  sleep 2

  # Check API
  if [[ -n "$API_PID" ]] && ! kill -0 "$API_PID" 2>/dev/null; then
    wait "$API_PID" 2>/dev/null
    local_code=$?
    [[ $STOPPING -eq 1 ]] && break
    API_RESTARTS=$((API_RESTARTS + 1))
    if [[ $API_RESTARTS -ge $MAX_RETRIES ]]; then
      log_err "API crashed too many times ($API_RESTARTS). Stopping web and exiting."
      STOPPING=1
      break
    fi
    log_warn "API exited (code $local_code). Restarting... ($API_RESTARTS/$MAX_RETRIES)"
    backoff=$INITIAL_BACKOFF
    attempt=0
    while [[ $attempt -lt $MAX_RETRIES ]]; do
      start_api && {
        log_ok "API recovered"
        API_RESTARTS=0  # reset counter on success
        break
      }
      attempt=$((attempt + 1))
      log_warn "  Restart in ${backoff}s... ($attempt/$MAX_RETRIES)"
      sleep "$backoff"
      backoff=$((backoff * 2))
    done
  fi

  # Check Web
  if [[ -n "$WEB_PID" ]] && ! kill -0 "$WEB_PID" 2>/dev/null; then
    wait "$WEB_PID" 2>/dev/null
    local_code=$?
    [[ $STOPPING -eq 1 ]] && break
    WEB_RESTARTS=$((WEB_RESTARTS + 1))
    if [[ $WEB_RESTARTS -ge $MAX_RETRIES ]]; then
      log_err "Web crashed too many times ($WEB_RESTARTS). Stopping API and exiting."
      STOPPING=1
      break
    fi
    log_warn "Web exited (code $local_code). Restarting... ($WEB_RESTARTS/$MAX_RETRIES)"
    start_web || true
    [[ $? -eq 0 ]] && WEB_RESTARTS=0
  fi

  # Check Gateway (only if we started it)
  if [[ -n "$GW_PID" ]] && ! kill -0 "$GW_PID" 2>/dev/null; then
    wait "$GW_PID" 2>/dev/null
    local_code=$?
    [[ $STOPPING -eq 1 ]] && break
    GW_RESTARTS=$((GW_RESTARTS + 1))
    if [[ $GW_RESTARTS -ge $MAX_RETRIES ]]; then
      log_err "Gateway crashed too many times ($GW_RESTARTS). Continuing with API+Web."
      GW_PID=""
      GW_RESTARTS=0
    else
      log_warn "Gateway exited (code $local_code). Restarting... ($GW_RESTARTS/$MAX_RETRIES)"
      if start_gateway; then
        GW_RESTARTS=0
      fi
    fi
  fi
done
