---
id: 20260924_gateway-thin-edge-filters
status: done
priority: high
sprint: 2026-09
title: Thin edge: path filters + chat-only + health parse
gh: —
tags: gateway, edge, filters
---

## Task
ROADMAP Gateway remaining work: thin generic edge (filters, chat-only) now that byte-relay + compression + frontend cutover are live.

## Scope
1. Strict path filters at edge (traversal / optional deny prefixes).
2. Opt-in chat-only mode (`MAN_GATEWAY_CHAT_ONLY=1`).
3. Fix `check_sidecar_health` to read API envelope `data.model_type` / `data.model_loaded`.
4. cargo test + fmt/clippy; rebuild; live verify; update ROADMAP bullet.

## Verification (2026-09-24)
- Branch `feat/gateway-thin-edge`. New `apps/gateway/src/filters.rs` + health parse in `main.rs`. 19/19 cargo tests green; `fmt --check` + `clippy -D warnings` clean; `cargo build --release --locked`.
- Traversal `--path-as-is /models/../etc/passwd` → **403** `path rejected by edge filter` (no Python).
- Chat-only on :8081 (`MAN_GATEWAY_CHAT_ONLY=1 MAN_GATEWAY_DENY=/shell,/datasets`): `/health /models /openapi.json` pass; `/shell`+`/datasets` → deny 403; `/training/start /knowledge /vm` → chat-only 403; `/chat` reaches Python (405 GET = filter passed).
- Boundary-aware deny: `/shell` blocks `/shell/run`, not `/shelladmin`.
- `/health` now reports `model_name: Qwen/Qwen2.5-0.5B-Instruct` from `data.model_type` (was `unknown`).
- Compression regression: openapi identity 509668 / zstd ~65424 + `content-encoding: zstd` + `x-uncompressed-content-length: 509668`; SSE identity.
- Full mode (default :8080): `/shell` not edge-blocked (404 from Python). `/health/detailed` shows `chat_only` + `deny_prefixes`.
- Docs: `docs/ROADMAP.md` Gateway bullet + `docs/engineering-overview.md` §5 updated.
- Live gateway PID in `/tmp/slough-api-host/gateway.pid`.

## Notes
- Do not commit unless asked. No new Cargo deps. Auth/rate-limits stay in Python (exposure shape only).
