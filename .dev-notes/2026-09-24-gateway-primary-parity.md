---
id: 20260924_gateway-primary-parity
status: done
priority: high
sprint: 2026-09
title: Port gateway edge client + Range fix to primary worktree
gh: —
tags: gateway, downcraft, parity, compression
---

## Task
Port gateway-real-e2e work (clone feat/gateway-real-e2e, merged main 6dfb2637)
into primary `feat/vite-phase5-deploy-ci` so primary CLI gets edge compression.

## Result (DONE 2026-09-24, uncommitted on user branch)

Ported (working tree only — no commit unless asked):
- `packages/downcraft/downcraft/download/http.py` edge client
- `apps/gateway/src/main.rs` Range/validator forward + HEAD + identity sidecar AE
- `apps/gateway/src/compression.rs` UCL header + 204/205 skip
- `scripts/host_gateway.sh`, `scripts/gateway_file_sidecar.py`
- edge tests + `pyproject.toml` zstandard dep
- Re-pointed primary `.venv` editable install from clone → primary package

Verification:
- editable import resolves under `/home/mana/Documents/Default Project`
- downcraft 467/467 green (incl. TestEdgeCompression)
- ruff check clean; format clean on all `.py` (only pre-existing README.md format noise)
- cargo fmt + clippy -D warnings + 9/9 gateway tests green
- live via primary CLI → :8080: sha256 `d0de318d…` match; zstd wire 4,083,716 / identity 5,555,744 (26.5%); Range 206

Out of scope (open follow-ups):
- Commit on user's branch (ask first)
- Frontend API cutover (`NEXT_PUBLIC_API_URL` → gateway) — separate journey
- Docker/nginx validation (unavailable)
