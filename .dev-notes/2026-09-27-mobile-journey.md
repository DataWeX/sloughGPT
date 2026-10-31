---
id: 20260927_068_mobile_serve_journey
status: done
title: journey M — serve --mobile runtime E2E (startup/ready SIGINT + API reuse survival)
tags: [cli, mobile, signal, journey, dev-server]
---
# journey M: serve --mobile (api + metro)

## Problem
_cmd_api_and_mobile got early signal handling in 3e1d71da8 but has NO runtime
E2E — only the registration-before-spawn unit source-order test. Metro (8081)
spawns are unproven: startup SIGINT cleanup, ready SIGINT, API reuse survival.

## Plan
1. Read mobile spawn/metro command; check apps/mobile/package.json + fakebin gaps.
2. New scripts/e2e/journey_m.sh: startup INT → no orphan uvicorn/metro, ports
   8196+8081 freed; ready INT → Stopped; foreign API reuse → foreign survives.
3. Fake metro shim if needed (like fakebin/npx for next build).
4. Gates: ruff (if py touched), cli suite, journey M. Commit fast, land.

<!-- claimed by opencode 2026-09-27 -->


<!-- completed by opencode 2026-09-27 -->

## Result (2026-09-27) — done (commit 318cd8724, main=318cd8724)

1. journey M (`scripts/e2e/journey_m.sh`, 24 asserts, 3 phases): startup
   INT (cli+uvicorn+metro visible) → exit + ack + no orphans + ports
   8196/8081 freed; ready INT → Stopped + children terminated + no
   restart-after-shutdown; foreign-API reuse → foreign survives, metro
   cleaned up. Ports 8196+8081.
2. fakebin/npx: `react-native start` → http.server on $PORT (metro
   readiness without an RN toolchain).
3. Two masking traps found & fixed: (a) pgrep pattern `fakebin/npx` never
   matches post-exec cmdline (phase B check passed spuriously) → match
   `http.server 8081`; (b) pty `script` teardown SIGHUPs the group and
   kills leaked metro before the orphan check (M8 initially survived) →
   metro runs in its own session (trap TERM, no HUP trap); plain setsid
   alternative rejected: StatusBlock prints no ack in non-tty.
4. Mutation M8 (drop metro slot from _cleanup ×4) → FAIL all 3 phases
   (phaseA orphan, phaseB children survived, phaseC port open).
5. Gates: journey M 24/24 (×2 green), S 13/13, buildfail 6/6 (npx shim
   regression), unit 415+1skip. Rebuilt onto d54860f4e (no overlap).
