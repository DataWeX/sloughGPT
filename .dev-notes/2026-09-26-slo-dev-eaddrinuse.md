---
id: 20260926_061_slo_dev_eaddrinuse
status: done
title: slo dev port-busy: API EADDRINUSE gets remediation hint, web/API eaddrinuse unit+E2E coverage
tags: [cli, startup, eaddrinuse, journey, dev-server]
---
# slo dev port-busy remediation (journey D)

## Problem
Port already in use is the most common non-technical-user startup failure.
- API port busy: `_poll_services` takes the generic branch → "API server exited (code 1)",
  no `_handle_eaddrinuse` remediation (lsof pid + kill hint). `_stop_check`/post-serve
  only ever handle `web` eaddrinuse; `eaddrinuse_port` is a bare port, service hardcoded "web".
- Zero test coverage: `grep eaddrinuse apps/cli/tests/` → nothing (unit or E2E).
- `_handle_eaddrinuse` calls `subprocess.run(lsof ...)` unguarded → FileNotFoundError
  crash on systems without lsof.

## Plan
1. Poll API branch: `_is_eaddrinuse(api_lines)` → `status["api"]="eaddrinuse"` (skip generic message).
2. `_stop_check`: set `eaddrinuse_port[0] = (port, service)` for api + web; post-serve unpack.
3. Guard lsof `subprocess.run` (FileNotFoundError/TimeoutExpired/OSError → skip pid listing).
4. Unit tests: api + web eaddrinuse → stop + `log.key_value(service, "port N in use")`.
5. E2E journey D: bind port first → run slo dev → assert remediation hint + clean exit + no traceback.
6. Gates: ruff, CLI suite, journey D. Commit via plumbing, land on main.

<!-- claimed by opencode 2026-09-26 -->

## Result (2026-09-26) — done (commits 6f1fe54fa + 5db762ce1, main=5db762ce1)

1. API EADDRINUSE routes to remediation (labelled api/web via (port, service) pair); generic death message suppressed; lsof guarded; service-aware alt-port hint.
2. Unit tests: +2 (deterministic flag-gated poll) — 48 in file, 458 total CLI+startup gates, ruff clean.
3. E2E journeys ALL PASS: Anext 8/8, Bvite 7/7, api-death 6/6, eaddrinuse 18/18 (API busy11s exit, web busy 8s exit, child-only tracebacks allowed).
4. Harness merged in-tree as scripts/e2e/ (3 journeys + fake npm shim, JOURNEY_WT override, logs → git-ignored logs/e2e/); temp worktree + external folder removed.
