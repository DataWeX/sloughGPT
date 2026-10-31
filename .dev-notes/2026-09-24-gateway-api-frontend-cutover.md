---
id: 20260924_gateway-api-frontend-cutover
status: done
priority: high
sprint: 2026-09
title: Wire frontend through gateway (API edge cutover)
gh: —
tags: gateway, frontend, edge, cutover
---

## Task
ROADMAP: frontend still hits `:8000` direct; edge should terminate API traffic
with stateless compression. Process-level (no Docker).

## Scope
1. API on `:8000` (PYTHONPATH includes repo root so `domain` resolves).
2. Gateway `:8080` with `MAN_CORE_URL=http://127.0.0.1:8000` (was file sidecar).
3. `apps/web/.env.local` → `NEXT_PUBLIC_API_URL=http://localhost:8080`.
4. Verify: health JSON compressed through edge; Range/SSE identity rules still hold.
5. File-download e2e stays on host scripts / smoke alternate ports.

## Verification (2026-09-24)
- **Stale binary fixed**: primary release was Sep 13 (pre-port); rebuilt `cargo build --release --locked` → PID 29905.
- `openapi.json` via `:8080`: zstd 65,355 / gzip 60,045 vs identity 509,668; `x-uncompressed-content-length: 509668`; lossless decode.
- `/models` via `:8080`: zstd wire 440 vs identity 1713; CORS `access-control-allow-origin: *`.
- Gateway `/health` is a **local** handler (170 B identity — under `MIN_COMPRESS_BYTES=1024`); sidecar `healthy: true`. Huge API `/health` is not on this path.
- SSE: `POST /chat/stream` identity (no content-encoding), events stream through edge; `/training/stream` open-identity (timeout=keep-alive).
- Frontend: `.env.local` `:8080` + Vite `loadEnv` + `inlinePublicEnv` plugin (Vite 8 skips built-in `define` for non-bundled client dev modules). Live transform: `PUBLIC_API_URL = "http://localhost:8080"`.
- Typecheck: only pre-existing `app/Providers.tsx` SessionProvider error (same on HEAD). `lib/config.test.ts` 5/5 green.

## Notes
- Branch at time of write: `feat/vite-phase6-cutover` (concurrent session); do not commit unless asked.
- File-download e2e unchanged: host scripts / smoke alternate ports.
