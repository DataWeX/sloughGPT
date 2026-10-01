#!/usr/bin/env bash
# install.sh — install the SloughGPT stack as systemd --user units.
#
# Why: services launched from an agent shell live in that session's cgroup
# and die with it (2026-09-30: opencode restarts killed the shared :8000
# server three times). systemd --user units outlive the launching shell.
#
# Usage:
#   scripts/systemd/install.sh                  # install + enable + (re)start
#   scripts/systemd/install.sh --install-only   # copy units, enable nothing
#   scripts/systemd/install.sh --takeover       # also free ports from foreign
#                                               # (non-unit) listeners first
#   scripts/systemd/install.sh --linger         # loginctl enable-linger $USER
#                                               # (survive logout / boot start)
#
# Idempotent: safe to re-run; units are re-copied and daemon-reload run.
# Units point at the MAIN checkout (/home/mana/Documents/Default Project/sloughGPT)
# — machine-level config, not a worktree.
set -euo pipefail

SRC="$(cd "$(dirname "$0")" && pwd)"
DST="$HOME/.config/systemd/user"
UNITS=(slough-api slough-web slough-gateway)

INSTALL_ONLY=0
TAKEOVER=0
LINGER=0
for arg in "$@"; do
  case "$arg" in
    --install-only) INSTALL_ONLY=1 ;;
    --takeover)     TAKEOVER=1 ;;
    --linger)       LINGER=1 ;;
    -h|--help)      grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown flag: $arg (see --help)" >&2; exit 2 ;;
  esac
done

log()  { echo "[install] $*"; }
warn() { echo "[install] WARN: $*" >&2; }

# ── Preflight ──────────────────────────────────────────────────
command -v systemctl >/dev/null 2>&1 || { echo "systemctl not found" >&2; exit 1; }
systemctl --user daemon-reload 2>/dev/null || {
  echo "systemd --user session not available (is this a systemd host?)" >&2; exit 1; }

ROOT="/home/mana/Documents/Default Project/sloughGPT"

# Gateway needs its release binary; skip enabling it when absent so we
# never enable a unit that cannot start.
ACTIVE_UNITS=("${UNITS[@]}")
if [ ! -x "$ROOT/apps/gateway/target/release/slough-gateway" ]; then
  warn "gateway binary missing — install its unit but do not enable/start it"
  warn "build with: cargo build --release --locked -p slough-gateway"
  ACTIVE_UNITS=(slough-api slough-web)
fi
if [ ! -x "$ROOT/apps/web/node_modules/.bin/vite" ]; then
  warn "vite missing (apps/web/node_modules not installed) — skipping web"
  ACTIVE_UNITS=("${ACTIVE_UNITS[@]/slough-web}")
fi

# ── Install (idempotent) ───────────────────────────────────────
install -d -m 755 "$DST"
for u in "${UNITS[@]}"; do
  install -m 644 "$SRC/$u.service" "$DST/$u.service"
done
systemctl --user daemon-reload
log "installed ${UNITS[*]} → $DST"

if [ "$LINGER" = "1" ]; then
  loginctl enable-linger "$USER"
  log "linger enabled for $USER (units survive logout / start at boot)"
fi

if [ "$INSTALL_ONLY" = "1" ]; then
  log "install-only: nothing enabled or started"
  exit 0
fi

# ── Takeover: free managed ports from foreign (shell-child) listeners ──
port_pids() { ss -ltnp 2>/dev/null | awk -v p=":$1$" '$4 ~ p {print $NF}' \
              | grep -oP 'pid=\K[0-9]+' | sort -u; }
port_for() {
  case "$1" in
    slough-api) echo 8000 ;;
    slough-web) echo 5173 ;;
    slough-gateway) echo 8080 ;;
  esac
}

for u in "${ACTIVE_UNITS[@]}"; do
  [ -z "$u" ] && continue
  port=$(port_for "$u")
  if [ "$TAKEOVER" = "1" ]; then
    for pid in $(port_pids "$port"); do
      # Only kill when the port is owned by something systemd doesn't track:
      # a unit-owned process disappears from this set after `systemctl stop`.
      if ! systemctl --user show -p MainPID --value "$u" 2>/dev/null | grep -qx "$pid"; then
        log "$u: taking over :$port from shell pid $pid"
        kill -TERM "$pid" 2>/dev/null || true
      fi
    done
    sleep 2
    for pid in $(port_pids "$port"); do
      kill -9 "$pid" 2>/dev/null || true
    done
    sleep 1
  fi
done

# ── Enable + (re)start ─────────────────────────────────────────
FAILED=()
for u in "${ACTIVE_UNITS[@]}"; do
  [ -z "$u" ] && continue
  systemctl --user enable "$u" >/dev/null 2>&1 || true
  if systemctl --user restart "$u"; then
    log "$u: restarted"
  else
    FAILED+=("$u")
  fi
done

# Leave explicitly-disabled extras alone but report them.
for u in "${UNITS[@]}"; do
  skip=0
  for a in "${ACTIVE_UNITS[@]}"; do [ "$a" = "$u" ] && skip=1; done
  [ "$skip" = "0" ] && log "$u: installed but not enabled (preflight)"
done

echo ""
systemctl --user --no-pager --lines=0 status "${ACTIVE_UNITS[@]}" 2>/dev/null | grep -E "●|Loaded:|Active:" || true

if [ "${#FAILED[@]}" -gt 0 ]; then
  warn "failed to start: ${FAILED[*]}"
  warn "if a port is busy, re-run with --takeover; logs: journalctl --user -u <unit> -n 20"
  exit 1
fi
log "all enabled units active"
