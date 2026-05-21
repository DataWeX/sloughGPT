#!/usr/bin/env bash
# Start FastAPI (default :8000) and Next.js dev (:3000). Ctrl+C stops both.
# Same processes from repo root: npm install && npm run dev:stack (see package.json).
# SLOUGHGPT_API_PORT pins the API port (required for Next.js proxy to match).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export SLOUGHGPT_API_PORT="${SLOUGHGPT_API_PORT:-8000}"

cleanup() {
  if [[ -n "${API_PID:-}" ]] && kill -0 "$API_PID" 2>/dev/null; then
    kill "$API_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "Starting API: SLOUGHGPT_API_PORT=$SLOUGHGPT_API_PORT python3 apps/api/server/main.py"
python3 apps/api/server/main.py &
API_PID=$!

echo "Waiting for API /health on port $SLOUGHGPT_API_PORT..."
for _ in $(seq 1 90); do
  if curl -sf "http://127.0.0.1:${SLOUGHGPT_API_PORT}/health" >/dev/null; then
    echo "API healthy."
    break
  fi
  if ! kill -0 "$API_PID" 2>/dev/null; then
    echo "❌ API process exited before /health responded. Fix errors above or free port $SLOUGHGPT_API_PORT."
    wait "$API_PID" || true
    exit 1
  fi
  sleep 1
done
if ! curl -sf "http://127.0.0.1:${SLOUGHGPT_API_PORT}/health" >/dev/null; then
  echo "❌ API did not become healthy within 90s (http://127.0.0.1:${SLOUGHGPT_API_PORT}/health)."
  exit 1
fi

echo "Starting web: npm run dev (apps/web)"
(cd "$ROOT/apps/web" && npm run dev)
