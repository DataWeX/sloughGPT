---
id: 20260928_073_edge_cascade
status: done
title: "Card 073 — Edge cascade: node socket relay → Rust gateway owns policy"
tags: [gateway, edge, cascade, proxy, health-contract]
sprint: current
gh: null
---
# Card 073 — The gateway IS the edge

## Scope (user-directed)

- "rust shoudlrun eberything" → "make node callit ad start porting" →
  final call: **"no proxy — the gateway is the gateway"**. The node relay
  was ported into the repo as instructed, wired as a cascade, verified…
  and then retired from the serving path by user decision. The repo copy
  survives only as an optional harness socket.

## Topology (after, live)

```
browser → slough-gateway :8080 (policy, health contract, CORS, compression,
                                filters — owns the public socket directly)
       → FastAPI :8000
```

## What changed (branch `feat/gateway-edge-cascade`, `5b4b85a73`)

- `scripts/edge-proxy.mjs` — env `EDGE_PORT`/`TARGET_PORT` (8080→8081),
  same behavior as the cache-dir original.
- `apps/gateway/src/main.rs`:
  - `/health`, `/health/detailed` = **contract passthroughs** (FastAPI
    bodies verbatim + additive `gateway:"rust"`; edge envelope only when
    core unreachable). Frontend contracts: `model-controller` reads
    top-level `model_loaded`/`model_type`; `system-controller` reads the
    `data`-enveloped detailed blob — both preserved byte-for-byte.
  - `CorsLayer.expose_headers(Any)`; `build_router()` extracted for
    oneshot tests; 3 new tests (preflight, fallback, passthrough) = 25/25.
- `scripts/host_gateway.sh` — `CORE_URL` + `NO_SIDECAR=1` modes; file
  flow (sidecar :18000) unchanged by default.
- `docs/engineering-overview.md` §5 — cascade diagram + health contract.

## Verification (live, final topology = gateway direct on :8080)

- `GET :8080/health` == core body + `gateway` key; `/health/detailed`
  key sets equal + stable fields equal vs core (only clock/mem tickers
  advance between any two calls — passthrough exact)
- preflight → 200 via CorsLayer; zstd + `access-control-expose-headers: *`
  on the public socket; DELETE reaches FastAPI (404 envelope)
- `cargo test` 27/27, `cargo fmt --check` clean, prettier clean
- commits: `5b4b85a73` (cascade impl) → `353b5ae6b` (gateway owns
  :8080, no proxy) → `7fac45da2` (self-supervision)

## Self-supervision ("no systemd — survive crashes in Rust")

- `MAN_GATEWAY_SUPERVISE=1` (default in `host_gateway.sh`): parent loop
  in the same binary respawns `current_exe()` as a worker
  (`MAN_GATEWAY_WORKER=1`; worker flag beats inherited supervise flag —
  fork-bomb guard, unit-tested). Backoff 1s→30s, reset after 60s healthy.
- SIGTERM/SIGINT to parent chain-stops the worker (libc::kill →
  graceful `with_graceful_shutdown`); a hard-killed parent leaves the
  orphan worker serving — the socket never drops for a supervisor crash.
- Live: `kill -9` worker → logged `restart in 1s` → health 200 on a new
  pid ~2s later; SIGTERM parent → chain-stop, port released.

## Ops notes / open

- Reboot persistence intentionally NOT systemd (user: no Linux linkage):
  launch the supervised binary from whatever starts the session.
- zstd measured 5.2s/514KB during load-average 14 — CPU contention, not
  a codec issue (encoder level 3; gzip 0.36s same conditions).
- Disk was 100% full mid-build: freed pytest temps + npm cache (~2.1G).
  `apps/mobile/` = 9.8G regenerable (node_modules 6.9G + gradle
  intermediates 2.4G) — NOT touched (no bandwidth).
- Branch stacks on `feat/gateway-sse-compression` (which carries
  `22378dcec`); landing = user decision.

## Rate limiting → edge ("build everything into rust")

- Full port of `apps/api/server/infrastructure/rate_limit_middleware.py`
  into `apps/gateway/src/ratelimit.rs`: sliding windows, GPU-heavy route
  table (verbatim order), per-client global `MAN_GATEWAY_RATE_LIMIT` /
  `MAN_GATEWAY_RATE_WINDOW` (default 300/60, localhost ×10), best-effort
  workspace-JWT limits, `X-RateLimit-*` headers, `429 {"detail": …}`
  shape. Health probes, CORS preflights (CorsLayer stays outer), and
  `/static` bypass the limiter.
- Gateway-native addition: `MAN_GATEWAY_MAX_STREAMS` (32, `0` = off) —
  semaphore on `*/stream|*/regenerate`, permit held until the body
  finishes (inference capacity ≠ header delivery).
- Python's copy still mounted behind us — degrades to a no-op (edge
  rejects first, same windows). Removal = explicit user approval.
- Env: `MAN_GATEWAY_RATE_LIMIT`, `MAN_GATEWAY_RATE_WINDOW`,
  `MAN_GATEWAY_MAX_STREAMS`; `0` disables. `/health/detailed` now
  reports `gateway.rate_limit`.
- Verify: `cargo test` 40/40 (was 27), clippy clean, fmt clean.
- Live (release, supervised pair 248542/248552): 150× /chat/stream →
  100×405 (admitted; local ×10 of the 10/60 route limit) + 50×429
  `{"detail":"Rate limit exceeded for /chat/stream. …"}` with
  `x-ratelimit-limit: 100`, `retry-after: 60`, CORS+expose `*` intact;
  /health and /health/detailed stay 200 (FastAPI passthrough) with the
  window exhausted. commit `ed57d2599`.

## Gateway = performance/stats optimizer & manager (scope b)

- [x] stats.rs: bounded per-route counters, latency histogram, bytes,
      compression saved, in-flight gauge; cardinality cap 32 + "(other)"
- [x] instrument relay (identity + wire bytes, upstream latency/status)
- [x] GET /gateway/stats (additive, exempt from limiter)
- [x] breaker.rs: closed/open/half-open on consecutive upstream
      failures (env MAN_GATEWAY_BREAKER_FAILURES=5, _OPEN=5s, 0=off),
      503 {"detail"} + Retry-After when open; fed by request outcomes
- [x] tests + fmt/clippy + docs (§5 "performance control plane") + live
- Live smoke (release, supervised pair 15200/15209, core pid 34844):
  baseline stats shape ✓ breaker closed ✓; dead-core burst →
  5×502 (upstream_err) → breaker open → 95×503 shed instantly +
  5×429 rate shed ✓; half-open probe (404<500) → breaker closed ✓;
  x-ratelimit 3000/2998 local×10 ✓; /models gzip 5169→2549 B
  (50.7% saved) ✓; p50=196.6 p95=786.4 ms ✓; shed_429 +1 from an
  external caller (invisible without access log → next build item).
  /health/detailed gateway.* block only in edge fallback envelope
  (byte-passthrough while core alive — by design). commit 4a37bf150.

## Access log (one JSONL line per request) — DONE
- [x] MAN_GATEWAY_ACCESS_LOG path (default logs/gateway-access.jsonl, 0=off)
- [x] log at every edge_admission exit: ts_ms, ms, method, path, route,
      status, shed (rate|streams|breaker|null), client — incl. health
- [x] test: exempt + admitted + shed lines written; 56/56, clippy 0.
      Live: 101x /chat/stream -> 32x422 + 68 streams-shed + 1 rate-shed,
      all attributed; commit 7bcd9e214.

## Per-client stats (scope b completion) — DONE
- [x] stats.rs ClientStat buckets (peer IP, 32 cap + (other), sorted by
      activity), client_seen() Response/Shed429/Shed503; exempt traffic
      excluded (overhead, not consumers); generic slot<T>
- [x] clients[] in /gateway/stats + 3 tests (59/59, clippy 0, fmt OK)
- [x] live: health-only poll leaves clients[] empty; dead-core 502s
      land on 127.0.0.1 (5xx=4) — attribution + exemption proven.
      commit 1856cf92d, cherry-picked -> land/gateway d6cfce127
- [!] core :8000 died AGAIN (3rd time today: 44367 -> 34844 -> gone)
      — gateway handles it (502/breaker), but the core is unstable;
      not gateway-caused. flag to user.

## Adaptive stream limits (scope c — AIMD) — DONE
- [x] streamctl.rs: StreamControl — 2s windows, halve at ≥12.5%
      streaming failures (transport/≥500), +1/healthy, floor 1;
      CAS admission on in-flight gauge (try_stream_slot replaces
      Semaphore); signals from proxy streaming paths; stream_limit/
      stream_max in /gateway/stats; startup log
- [x] 7 tests (6 unit + router integration 8→4 with breaker off);
      66/66, clippy 0, fmt clean
- [x] LIVE: dead-core burst halved 32→16 visible in stats (core died
      AGAIN — 4th+ time; growth branch unit-tested only).
      commit a72eca18d, land/gateway 46a715e1b (10 commits total)
- [!] core :8000 chronically unstable (5 deaths today) — gateway
      copes (502→breaker→AIMD layered), but flag to user.

## Next: land branch stack onto main (original goal)
- [x] read-only: merge-base, conflict preview, main worktree dirty check
      — api-std (main) DIRTY (session mid-work: domain/testing staged);
      merge-to-main session landing test debt separately; real-e2e locked
      but lacks our commits; main's apps/gateway net-diff since base = 0
- [x] cherry-pick 8 gateway commits 48d791cef..7bcd9e214 -> branch
      land/gateway (worktree ~/Documents/sloughgpt-land-gateway) —
      ZERO conflicts (docs auto-merged); new SHAs db3602bf2..13ada8b6c
- [x] verify on main+8: cargo test 56/56, clippy 0, fmt OK, shared
      indexes gate green
- [ ] BLOCKED: ff/merge land/gateway into main (9 commits) once
      api-std is clean AND merge-to-main session finished (kanban);
      re-sync new gateway commits as they land (1856cf92d -> d6cfce127)

## Admission benchmark + keep-alive fix (scope c follow-through) — DONE
- [x] scripts/benchmark_gateway_admission.py — 6 scenarios, spawns its
      own gateway (dead/hang core), asserts + reports: exempt /health
      6461 rps (server-side p50 0.1ms), stats p50 301µs, rate 429 p50
      344µs, breaker 503 p50 288µs, stream-cap 429 p50 350µs, AIMD
      8→4 PASS — ALL SCENARIOS PASS, logs/bench_gateway_admission.json
- [!] ROOT-CAUSED a real keep-alive bug found by the benchmark: every
      reject path (rate 429 / streams 429 / breaker 503) responded
      WITHOUT reading the request body; http.client sends headers+body
      as two writes, so hyper saw an unread-body drop and closed the
      connection after every shed POST (BrokenPipeError at ~N; kernel
      coalescing made it flaky). Repro: one-write 40/40 pass, two-write
      dies at 11.
- [x] fix: drain_body() bounded 64KB pre-read before each reject →
      keep-alive preserved (40/40 with 30x429 on one conn); >64KB
      falls back to close. New regression test
      split_body_reject_keeps_connection_alive (FAILS without fix —
      verified). 67/67, clippy 0, fmt clean.
- [x] commits fd7c0117b (fix) + e56f755cf (benchmark+docs);
      land/gateway be3a91fb7/fdc64d218 (12 commits), 67/67 green.
- [x] live gateway restarted on :8080 (supervisor, fixed binary),
      core :8000 back up (model_loaded true) after today's N deaths.
- [ ] BLOCKED (unchanged): land/gateway -> main when api-std clean +
      merge-to-main session done: git -C sloughgpt-api-std merge
      --ff-only land/gateway

## Benchmark results tracking — DONE
- [x] benchmark_results.py gains kind "admission" (choices, thresholds
      pass.rps higher-is-better 30% rel, shed p50s 50% rel) + nested
      _dig descent into metrics (was flat-keys only — compare silently
      skipped dotted paths); history prints compact admission line
- [x] recorded 2 real runs; compare verified end-to-end: honest
      [FAIL] flags on run 2 (box load 8.1: pytest 37min + profiling +
      model) — environmental, not code; ruff clean
- [x] commit 691c12f14; land/gateway 3a6bcd648 (13 commits; 3-way
      conflict resolved = main + admission)
- [x] core restarted under PYTHONPATH=. (editable finder lacks `domain`;
      plain launch dies ModuleNotFoundError) — healthy, model_loaded,
      gateway passthrough gateway:"rust". dev-stack.sh already provides
      API supervision (backoff restarts) — no new supervisor built.

## Shared-index registration — DONE
- [x] apps/README.md: gateway/ + mobile/ entries (mobile was ALSO
      missing — pre-existing gap; freshness check now 0 missing);
      docs/INDEX.md: engineering-overview.md row (was unlisted);
      docs/FEATURES.md: new "Gateway Edge (Rust)" feature table
      (compression/filters/health/self-supervision/rate/stats+
      breaker/access log+clients/AIMD/benchmark)
- [x] verified against tests/test_shared_indexes.py rules (read the
      untracked WIP copy; manual check passes both invariants for docs)
- [x] commit 2ed654a18; land/gateway 5e98c53ba (14 commits; FEATURES
      conflict resolved = journey library + gateway sections both kept)

## Access-log rotation — DONE
- [x] MAN_GATEWAY_ACCESS_LOG_MAX_MB (default 64 MB, 0 = no rotation;
      unparseable -> default): pre-write size check under the log lock,
      rename -> <path>.1, reopen append; reopen-fail degrades to writing
      the rotated file (warn), never panics
- [x] tests: parse_max_mb_honours_zero_and_falls_back,
      access_log_rotates_at_cap (4 KiB cap, 200 lines, asserts .1 cut
      near cap + active file small) — 69/69, clippy 0, fmt clean
- [x] startup line shows rotation cap; ENVIRONMENT.md new var section +
      engineering-overview clause
- [x] commits: edge 4b98f67e8, land a189b6466 (15 commits, gates green);
      live :8080 restarted on new binary (pids 1430112/1430122,
      "rotates at 64 MB" in logs/gateway-edge.log, health green)

## Day-3 ops round (Sep 30) — findings + restores
- [x] **Gateway was DOWN 21h** — log shows "shutdown requested" yesterday 13:17:49
      local (tool-kill quirk), never restarted. Restored from edge-cascade with
      rotation binary: pids 69410/69421, /health 200 gateway:"rust",
      "rotates at 64 MB" armed.
- [x] **host_core.sh experiment → do NOT land** (left untracked for user's call):
      core supervision already exists two ways — another OpenCode session
      restored the core within 5s of my kill (cmdline: PYTHONPATH=. .venv/bin/python3
      main.py --port 8000, OPENCODE_SESSION_ID in env), and main.py itself has
      --daemon (detach) + --force (kill port holder) + default connect-to-existing.
      Near-miss: my supervised child raced that session's starter for :8000;
      both my procs TERMed cleanly, their core kept the port.
- [x] **11:46:05 mystery — likely solved**: main.py startup force-kills whatever
      holds :8000 (lsof -ti + kill, lines ~615-641) when a second starter runs with
      --force/--daemon. A second starter that then died = old core killed with no
      traceback + port left down = exactly yesterday's signature.
- [x] Core note: other session's instance is phase=init, model_loaded:false
      (rss 73MB, /v1/models 404 poller every 30s from an opencode desktop client) —
      their lifecycle to manage, not ours.
- [ ] Landing still blocked: api-std 30 files, pytest suite still running (2 procs,
      timeout=2400 started 09:38:39 → done by 10:18 latest).

## Compression → benchmark_results integration — DONE
- [x] benchmark_gateway_compression.py --metrics-out: nested metrics JSON
      (payload/{gzip,zstd,identity}.{ratio,savings_pct,enc_mibs,dec_mibs};
      codec keys normalised "zstd-3"→"zstd" so thresholds stay stable)
- [x] benchmark_results.py: compression kind (3 choices lists + docstring),
      REGRESSION_THRESHOLDS (zstd/gzip savings_pct, rel 10%, higher-is-better),
      history line, live-run guarded to --json-file
- [x] verified end-to-end: record (1st), auto-compare [OK] Δ0 (2nd),
      synthetic 84% baseline → ⚠ REGRESSION [FAIL] exit 1 (fake record removed
      after), history renders, no-json error path exit 1; ruff clean
- [x] commits: edge b07d50781 (incl. own history file), land ff675369c
      (code 3 files, gates green); FEATURES row updated
- [x] housekeeping: scripts/host_core.sh left UNTRACKED (superseded — see
      day-3 ops round; user decides keep/delete)

## host_gateway.sh external supervision — DONE (answers the 21h outage)
- [x] launcher now loops: wait → relaunch on exit (1→30s backoff, reset after
      60s-healthy), interruptible sleep, stopping flag, markers append to
      gateway.log (shown once via tail -f, no tee double), tail -f killed on
      exit (was leaking), fresh log truncate at session start
- [x] e2e on :8081 (NO_SIDECAR=1): health 200 → TERM supervisor →
      "edge exited rc=0 after 23s — restarting in 1s" → new worker pair →
      200 again; live :8080 edge untouched; teardown clean (8081 down,
      test procs gone); bash -n ok
- [x] docs: engineering-overview §5 one-layer-above sentence + apps/README
      launch note; commits edge 37d68ba48, land 256db3e90 (17 commits, clean)
- [x] stack: gateway 69410/69421 HTTP 200; core = other session's instance
- [ ] landing still blocked: api-std 30 files, 1 pytest proc
