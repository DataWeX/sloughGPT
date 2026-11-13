---
id: 20260926_062_serve_web_journey
status: done
title: journey S — serve --web E2E (build shim, standalone node server, SIGINT clean)
tags: [cli, e2e, serve, journey]
---
# serve --web journey (scripts/e2e/journey_serve.sh)

## Scope
`serve --web` — the original complaint command — had no journey. Added as
journey S in the in-tree harness (main=24b1c6851, commit on top of 5db762ce1).

## What it walks
1. Ports prechecked free (serve REUSES healthy services and `_kill_port`s them
   on shutdown — never touch someone's live instance; other sessions run
   serve on 8000/8131, journey uses 8199/4570).
2. `npx next build` (main flow builds standalone first) → fake `npx` shim
   writes a working `standalone/server.js` (node http, honours PORT/HOSTNAME).
   Real build is minutes + network (first attempt died on npm EINTEGRITY).
3. API healthy t+10s, web 200 t+1s, both URLs in the TTY status block.
4. SIGINT → "Stopped" within 5s, ports 8199+4570 freed, no leftovers,
   no CLI traceback.

## Result
13/13 ALL PASS. Journey bug found and fixed along the way (ANSI-cleaned log
was snapshotted before SIGINT — shutdown lines missing from greps).
Recovery worktree index repaired (2 new files showed as staged deletions
after the ref move; `git checkout HEAD -- scripts/e2e/...` fixed it).

<!-- claimed by opencode 2026-09-26 -->

## Result (2026-09-26) — cleanup ownership + build-fail leak fixed (main=7811a7b82)

- `_cleanup` kills a port only when its proc was spawned by us (proc is not None);
  reused/foreign services survive shutdown and early-error paths. +3 unit tests
  pin the contract (461 passed total).
- Build-fail path now calls `_cleanup(api_proc, None, …)` — was leaking uvicorn.
- journey_serve.sh `buildfail` mode: FAKE_NPX_FAIL=1 FAKE_NPX_DELAY=20 (late
  failure, like a real build — early failures die from EPIPE before binding and
  mask the leak), plain setsid (script pty's group-SIGHUP masked it too).
  Mutation-proven: removing the cleanup call → FAIL "LEAK: api port open" +
  "orphan uvicorn". With fix: buildfail 6/6, success 13/13.

## Result (2026-09-26) — signal handling + journey R (main=47a8f3922)

- Three bugs: (1) bash `&` inherits SIGINT=SIG_IGN, CPython honours it →
  interrupts before handler install silently dropped (Ctrl+C did nothing)
  and SIGTERM was default-kill (orphans). Fix: default_int_handler at
  cli.py module top + full cleanup handler at `_cmd_api_and_web` entry +
  `shutdown` checks at every startup phase boundary. (2) monitor restart
  race: handler runs during sleep(2), body mistakes cleanup exits for
  crashes → respawns web after "Stopped". Fix: post-sleep re-check.
  (3) web_proc init next to api_proc for the early handler.
- journey R (`scripts/e2e/journey_reuse.sh`, 15 assertions, two phases):
  startup SIGINT at t+2 → honoured + ports free; plant foreign API, reuse,
  SIGINT in monitor sleep → foreign survives, no restart-after-shutdown.
- Mutation-proven: M1 ownership → "foreign api KILLED"; M2 no re-check →
  "web restarted after shutdown"; M3 original signals → "cli survived
  startup SIGINT". Green 2/2, success 13/13, buildfail 6/6, 461 unit tests.
- Note: /tmp/opencode wiped twice more mid-session (7th/8th) — committed
  via plumbing BEFORE mutation testing after the first loss.
- Follow-ups: `_cmd_api_only`/`_cmd_api_and_mobile` still register their
  handlers after their wait loops (same silent-drop window, covered by the
  cli.py top-level default_int_handler → clean "Interrupted" but without
  per-command cleanup); journey A/D could grow the same startup-INT phase.

## Result (2026-09-26) — early signals for api-only/mobile/cmd_dev (main=3e1d71da8)

- Closed the follow-up from the previous section: `_cmd_api_only` and
  `_cmd_api_and_mobile` now register full cleanup handlers BEFORE their
  spawns/wait loops (was: after — an interrupt escaped as bare
  "Interrupted" and orphaned uvicorn/metro); `cmd_dev` registers its
  flag-only handler at function entry. api-only startup timeout now
  terminates its spawned API + frees the port (was: leak). Mobile gets
  the same monitor post-sleep re-check as web. `shutdown` must be bound
  BEFORE the first `_update_status()` call (guard reads it — caught by
  the new unit tests immediately).
- journey S2 (`scripts/e2e/journey_serve_api.sh`, 18 asserts): startup
  SIGINT at t+2 → exit + Stopped + no orphan uvicorn + port freed; ready
  SIGINT (spawned API) → Stopped + uvicorn terminated + port freed;
  foreign-API reuse → no cli uvicorn + foreign survives.
- Battery: unit 415+1skip, R 15/15, S success 13/13, buildfail 6/6,
  API 6/6, eaddrinuse 18/18, S2 18/18. Mutations: M4 (drop timeout
  cleanup) → unit FAIL; M5 (late registration) → unit KeyboardInterrupt
  at the SIGINT injection line + source-order test (all three variants
  must register before `subprocess.Popen`). Journey-level M5 re-run was
  blocked twice by /tmp wipes (9th/10th) — landed on unit-level catches.
- Landing: rebuilt onto 69811c133 (zero overlap), ff in recovery worktree.
- Follow-ups: `_cmd_api_and_mobile` has NO runtime journey (mirror code,
  covered by source-order unit test only); full-suite `pytest` contends
  with other sessions' heavy runs (load 27) — core-py full run times out
  under contention; `tests/server/test_agents_router.py` 11F is
  pre-existing (identical at base — router lane).

## Result (2026-09-27) — leak-proof orphan checks + M5 journey-proven (main=a1e48396b)

- Ported journey M's leak-proof design into R phase 1 + S2 phase A:
  headless `setsid -w` replaces pty `script -qec` (pty teardown SIGHUPs
  the process group and kills leaked children before the orphan check —
  those asserts were masked); the non-tty-invisible Stopped/Interrupted
  grep is replaced by an exit-code assert (0 = handler path ran);
  orphan + port probes are now actually leak-sensitive.
- Timing: fixed t+2 INT raced import time under load (one run INT'd
  mid-import → traceback + EC=130, correctly flagged "handler never
  ran" but harmless) — replaced with wait-for-uvicorn-child: the child
  spawns inside the command handler after registration, so its presence
  guarantees handler-installed + pre-ready window.
- **M5 finally journey-proven** (the twice-wiped run): late signal
  registration → S2 phase A `FAIL: phaseA orphan uvicorn`, RC=1.
- New pre-existing bug found, NOT yet fixed: `apps/cli/src/cli.py`
  `main()` has `except SystemExit: pass` (slo_cli inline refactor
  3759cadaf) which swallows ALL exit codes — `sloughgpt badcmd` → 0,
  framework KI path `sys.exit(130)` → 0. The new EC asserts therefore
  read 0 on both paths today (orphan checks carry detection); fix
  proposal sent for approval.
- Battery: unit 415+1skip (targeted apps/cli), R 15/15, S2 18/18 (all
  green on final files), M5 caught; executable bit restored (755).
- Landing: rebuilt onto 73ab55d8f (zero overlap), ff in recovery;
  commits 925fad92a + a1e48396b; .wt-lib worktree removed (13 = baseline).

## Result (2026-09-27) — exit codes preserved through main() (main=c5a576f3a)

- Fixed the bug found by the leak-proof work: `main()` swallowed
  SystemExit (`except SystemExit: pass`, slo_cli inline refactor
  3759cadaf) → every failure path exited 0 (usage errors, framework KI
  `sys.exit(130)`, command `sys.exit(1)`). `main()` now captures the
  code, runs the suggestion block, then `sys.exit(code)`; string codes
  → stderr + 1; KeyboardInterrupt still exits 130 immediately.
- Unknown commands printed and returned (exit 0): `_show_error` in
  `core/framework.py` now `sys.exit(1)` — single choke point for
  no-match and fuzzy-declined paths.
- New `apps/cli/tests/test_cli_main.py` (7 tests) pins the contract;
  `test_slo_cli::test_run_unknown_command` updated to expect
  SystemExit 1. Mutations both caught: M-A (drop `_show_error` exit)
  → 2 FAIL; M-B (revert main() capture) → 3 FAIL.
- Battery: unit 422+1skip, ruff clean, R 15/15, S2 18/18 (EC asserts
  now meaningful: handler path = 0), SDK tests 47 pass (separate entry
  point), manual: badcmd=1 badflag=1 help=0 (was 0/0/0).
- Post-land M5 re-proof: late registration now fails S2 phase A TWICE —
  `exit 130` (contract live end-to-end) + `orphan uvicorn`. Worktree
  `.wt-ec` + branch removed post-verification.
- Incidents: disk hit 100% mid-run (ENOSPC truncated cli.py mid-mutation
  — recovered: git restore + re-apply; 6.4G free after). Recovery
  worktree had been switched to `feat/brand-man` by another session, so
  the ff there moved that branch (it now contains the fix — harmless);
  main was checked out in `sloughgpt-router-wiring` (clean, no overlap)
  and fast-forwarded there: 254b59096..c5a576f3a.
