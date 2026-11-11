# Unified serve (one origin/IP)

Architecture:
- **Backend / core infra:** Python FastAPI (`apps/api/server/main.py`, default `:8000`)
- **Core gateway:** Rust Axum (`apps/gateway`) — serves static SPA + byte-relays API/SSE to Python sidecar on one origin (`:8080` default)
- **Frontend:** Vite + React (`apps/web`), production build `dist-vite/` (same-origin)

## One script
`scripts/serve.py` is the canonical launcher:

```bash
# auto: unified if gateway binary + dist-vite exist, else dev (vite proxy), else api-only
./scripts/python scripts/serve.py
# or
python3 scripts/serve.py

# unified single-origin (requires gateway binary + apps/web/dist-vite)
python3 scripts/serve.py --mode unified
# build prerequisites if tools exist
python3 scripts/serve.py --mode unified --build-web --build-gateway

# dev single-origin on :5173 (Vite proxies backend paths to API :8000)
python3 scripts/serve.py --mode dev

# API only
python3 scripts/serve.py --api-only
```

Env:
- `SLO_API_PORT` (default 8000), `MAN_GATEWAY_PORT` (default 8080), `WEB_PORT` (default 5173)
- `SLO_API_URL` (Vite dev proxy target), `SLO_PYTHON` (interpreter override)
- Gateway: `MAN_CORE_URL`, `MAN_STATIC_DIR` set by script in unified mode

## Why gateway is the unified origin
`apps/gateway` serves `dist-vite/index.html` for navigations, static assets, and relays data/mutating requests + SSE to the Python sidecar. This puts UI and API on the same host:port/IP (no CORS split). Vite dev mode can approximate single-origin via `server.proxy` in `apps/web/vite.config.ts`.

## Current sandbox notes
This environment may lack Python and Rust toolchains; in that case `serve.py` reports what’s missing. The web dev server can still run on `:5173` with proxy config ready for when API is available at `:8000`.
