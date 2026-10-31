---
id: 20261020_wire_layer_phase3_lifespan_ab
title: "Wire layer phase 3: lifespan, uvicorn A/B, RFC6455 WS"
status: wip
date: 2026-10-20
tags: [wire, asgi, lifespan, uvicorn, websocket, benchmark, ecea49c8]
board_id: ecea49c8-8b67-4556-bd1b-5521399f1e40
---

# Wire layer phase 3: lifespan, uvicorn A/B, RFC6455 WS

Card `ecea49c8` (own the HTTP wire) — the two phase-3 items that did NOT
need the shared-entrypoint coordination window. Committed on the stack
tail `feat/npu-caps` (the wire branch is an ancestor of the landing stack;
keeping one branch per lane avoids rewriting pushed evidence).

## Lifespan (ASGI protocol) — DONE 2026-10-20

- `wire/asgi_host.py` gains `Lifespan` + `LifespanError`: one dedicated
  long-lived loop thread per server drives the spec handshake — the app's
  single `__call__` task parks in `receive()` between phases, so startup
  and shutdown handlers see the same objects. `startup()` runs after bind,
  BEFORE the accept thread spawns (no request can beat it); `shutdown()`
  runs after all connections close. All failure paths unwind the loop
  thread (failed message, raw app crash, timeout — no leaks).
- `WireServer(lifespan=...)`: `"on"` strict (startup failure fails start
  and unwinds the listener), `"auto"` default = uvicorn-compatible
  (log + serve anyway → lifespan-unaware hermetic apps keep working),
  `"off"` skip. Handler mode never constructs a driver. Failed
  `shutdown.failed` is logged, never raised out of `stop()`.
- Loop-affinity constraint documented: lifespan handlers must create
  loop-AGNOSTIC resources (request cycles run on per-request loops) —
  this repo's sync-core philosophy makes its FastAPI lifespan compliant.
- Tests: `tests/test_lifespan.py` 11 new (ordering, spec-conformant scope,
  strict/auto/off modes, crash+timeout fast-fail with no thread leak,
  idempotent shutdown, **real FastAPI lifespan interop**: `{"phase":"up"}`
  served pre-stop, `down` post-stop). Full wire suite **48 green**, ruff
  check+format clean. One pre-existing test updated honestly: the http
  scope is no longer `scopes[0]` (lifespan scope now comes first).

## A/B harness vs uvicorn — DONE 2026-10-20

`scripts/benchmark_transport.py` gains `_UvicornTarget` (in-process
uvicorn 0.53, own thread + loop, pre-bound ephemeral port via
`serve(sockets=)`, same app, lifespan off on both sides, same raw-socket
client, rows interleaved in one run) + `uvicorn_asgi_gapped` capacity row
+ `--no-uvicorn`.

Two runs (load 2.7 → 1.2):

| row | run 1 | run 2 |
|---|---|---|
| keepalive (sync handler) p50 | 0.121 ms | 0.124 ms |
| keepalive_asgi p50 | 0.378 ms | 0.354 ms |
| uvicorn_asgi (tight loop) p50 | 41.085 ms | 41.017 ms |
| uvicorn_asgi_gapped (100ms idle) p50 | — | 0.937 ms |
| ASGI adapter overhead | +0.257 ms | +0.230 ms |

Mechanism localized (probe: ttfb vs total vs gapped): uvicorn TTFB
0.774 ms — headers fast — but total 41 ms = head and body go out as TWO
writes (grep: no TCP_NODELAY anywhere in uvicorn) and the client's
delayed-ACK lets Nagle hold the body ~40 ms on a tight keep-alive loop.
With 100ms idle per request uvicorn runs 0.94 ms → its real capacity;
our wire serializes head+body in ONE send so the stall never arms
(2.6× even vs uvicorn's best case). Honest framing recorded in the
benchmark docstring + output line.

## RFC6455 WebSocket (flip gate) — DONE 2026-10-20

The last self-contained phase-3 item: minimal RFC6455 WS, the gate named
on the card for any default-transport flip. Built in the idle window while
the coordination window stayed closed.

**Where it lands** (one-module-per-concern, matching the existing split):
- `wire/websocket.py` — sync stdlib half: handshake validation, 101 +
  `Sec-WebSocket-Accept` (sha1 of key+GUID, base64), server-frame encoder
  (unmasked), client-frame reader (unmask, all three length forms: 7/16/64
  bit), fragmentation across continuation frames, ping→auto-pong. No
  `asyncio` import (PGQ event-loop-free discipline, same as `framing`).
- `wire/asgi_host.py` — `run_websocket` + one-session-per-connection loop:
  ASGI `websocket.connect/accept/receive/send/close/disconnect` contract,
  a **dedicated single-worker reader executor** per session (a cancelled
  `receive` shares the in-flight read via `shield`, never strands a second
  reader on the buffered stream).
- `wire/server.py` — `_serve_conn` routes `is_ws_upgrade(request)` to
  `run_websocket` (session consumes the connection; no keep-alive after).

**Scope proven by the API's real surface**: the only WS consumer is
`inference.py` `add_api_websocket_route("/ws/generate", ws_generate)` using
Starlette `accept/receive_text/send_json/close`. Test
`test_fastapi_websocket_route_interop` pins exactly that route shape end
to end over a real socket.

**Two real bugs the tests caught** (both fixed, both had a regression test
first):
1. *Cancelled-receive teardown deadlock.* Starlette's `wait_for(receive,
   30s)` cancels a pending receive; the in-flight read stayed on the
   default executor, which `asyncio.run` joins at session end while the
   connection is still open → hang. Fixed: session-owned reader executor,
   `shutdown(wait=False, cancel_futures=True)`.
2. *Reader-close deadlock (second order).* `reader.close()` waits on the
   makefile lock held by that blocked read; `close()` on a socket does not
   interrupt a blocked `recv` — only `shutdown` does. `_serve_conn` teardown
   now does `conn.shutdown(SHUT_RDWR)` before `reader.close()`.

Tests: `tests/test_websocket.py` 14 new (handshake 101/400/426, text +
binary 16/64-bit echo, fragmentation, ping→pong, unmasked-client-frame
1002, server-push-while-receive-pending, close/EOF, reject→403, plain-HTTP
untouched, cancelled-receive teardown, FastAPI interop). Full wire suite
**62 green**, ruff check+format clean.

One pytest gotcha recorded in-file: this file uses
`from __future__ import annotations`, so FastAPI resolves the endpoint's
`websocket: WebSocket` annotation against **module** globals — importing
`WebSocket` inside the test silently degrades it to a required query param
and the route 403-rejects instead of accepting. `FastAPI`/`WebSocket` are
imported at module level for this reason.

Benchmark: `benchmark_transport.py` re-run after the teardown change
(added `shutdown` on every HTTP connection path) — no regression:
keepalive p50 0.087 ms, keepalive_asgi p50 0.352 ms, adapter overhead
+0.265 ms, uvicorn A/B delta unchanged (−99.1% tight-loop).

## Tier-1 hardening — uvicorn parity gaps closed (same card, wire-local)

The A/B comparison enumerated what we lose vs uvicorn; the hardening
pass closes everything that lives inside `packages/wire-py`:

- **Proxy headers** (`asgi_host._apply_forwarded`): `X-Forwarded-For`
  (leftmost) + `X-Forwarded-Proto: https` translate into the ASGI
  scope's `client`/`scheme` — only from trusted direct peers
  (`WireServer(forwarded_allow_ips=...)`, default `"127.0.0.1"` uvicorn
  parity; `""` = trust nobody). Untrusted peers keep the real TCP
  identity. This is what makes wire usable behind the Rust gateway.
- **Keep-alive idle timeout** (`keep_alive_timeout=5.0`): socket
  timeout armed ONLY around the request read (first byte included —
  bounds slowloris); cleared before any response write so SSE and slow
  readers run unbounded. Buffered bytes cost no syscall → pipelined
  requests never false-trip. Timeout at a frame boundary = silent
  close, same as EOF.
- **Body cap** (`max_body`, default 64 MiB → 413): we buffer whole
  bodies, so the cap is the memory bound; Content-Length is checked on
  the declaration (413 without reading a byte), chunked streams count
  toward the same cap, trailer lines are now capped too.
- **WS message cap** (`ws_max_size`, default 16 MiB): enforced on the
  DECLARED frame length before any allocation, and on fragment
  assembly; over-cap closes with RFC6455 1009 (Message Too Big).
  `WsProtocolError` now carries the close code (1002 default).
- **Expect: 100-continue**: bare `HTTP/1.1 100 Continue` sent via the
  `on_expect` callback only AFTER framing + cap validated (never for a
  request about to be refused); unknown expectation → 417 + close.
- **Access log**: uvicorn-format line
  (`127.0.0.1:port - "GET /x HTTP/1.1" 200`) on the `wire.access`
  logger — handler mode (server.py), ASGI mode (asgi_host.py) and WS
  upgrade (101). Logs the raw TCP peer, not the forwarded identity;
  `WireServer(access_log=False)` silences.

Correction to the A/B claim list: CL/TE smuggling rejection,
conflicting-CL rejection, obs-fold rejection and chunked REQUEST
bodies were already in `framing.py` from phase 1 — the only real body
gap was the size cap.

Deferred deliberately: request-body STREAMING (architectural — needs
an ASGI receive-shape change + its own card), permessage-deflate
(loopback pointlessness), dev hot-reload (shared-tree entrypoint,
window-blocked).

Tests: `tests/test_hardening.py` 13 new (proxy translate / trust-off /
peer-identity, idle-close silent + after-response, pipelined-survives-
timer, 413 CL + chunked + under-cap pass, 100-continue exchange, 417,
access-log handler + ASGI) + 2 WS-cap tests in `test_websocket.py`
(1009 single frame + fragmented assembly). Full wire suite **77
green**, ruff clean.

Test-side flake found & fixed while pinning pipelining: the response
helpers discarded the surplus when two tiny responses coalesce into
one `recv` (the access log proved the server had already sent both),
leaving the second read to wait until the server's idle timer closed
the socket. Helpers now thread a shared read buffer: 43/60 flake →
0/15 loop + full suite green.

Benchmark re-run: no regression (keepalive p50 0.099 ms, keepalive_asgi
0.383 ms, adapter +0.284 ms, uvicorn A/B −99.1% tight-loop unchanged);
the benchmark now constructs its WireServers with `access_log=False`.

## Remaining for card ecea49c8 (blocked on coordination window)

- `SLO_TRANSPORT` wiring in the shared dev entrypoint (main tree has
  ~110 dirty files, parallel lanes active).
- Journey gates (apps/web) before any default-transport flip.
- ~~RFC6455 WS~~ DONE — the flip gate is cleared; only the shared-tree
  wiring + gates remain.

## Streaming request bodies (card c24543a4) — the deferred Tier-1 #4, landed

Bodies over `body_buffer_threshold` (default 64 KiB) reach the app
incrementally through ASGI `receive()` (`http.request` + `more_body`)
instead of buffered whole; `max_body` became a RUNNING total (413 via
`BodyTooLarge` — matters for chunked, which has no declared size to
pre-check). Small bodies keep the exact today-path: `read_request`
collects inline through one `BodyReader` and hands a complete
`Request.body` — the hot path (chat JSON) is byte-identical, pinned by
`test_small_body_stays_a_single_frame`.

Design: `BodyReader` (framing.py, sync) reads through the connection's
buffered reader so pipelined bytes are never stolen; the threshold
decision's read-ahead is `queue()`d and emitted first, so switching to
streaming never loses or reorders bytes. Reads run OFF the event loop
on a per-request single-worker executor (the WS frame-reader seam) —
`server.py` creates it only when `request.body_reader` is set, tears it
down `shutdown(wait=False, cancel_futures=True)` in a finally. Handler
mode stays buffered (drains via `read_all()` + `dataclasses.replace`) —
streaming is ASGI-mode only, documented in the card.

Two real bugs caught by the RED tests:

1. **Cancellation lost body bytes.** Starlette's `wait_for(receive())`
   cancels the wrapped future; the executor read still completed and
   its bytes were discarded, and the retry hit a cancelled future.
   The WS cancel-and-reset pattern would corrupt an HTTP body (bytes
   may never drop). Fix: `asyncio.shield(read_future)` — cancellation
   abandons the AWAIT, never the future or its result; the retry
   awaits the same future. Pinned by
   `test_cancelled_receive_shares_the_in_flight_read` (two-piece body,
   two 50 ms deadlines outlived).
2. **Access-log race in my own hardening test**: the client finishes
   reading the response before the server thread reaches
   `_log_access`; the raw caplog assert could see an empty set. Both
   access-log tests now poll bounded (`_wait_for_access_line`, 2 s).

Also fixed `_BODILESS` → `_BODYLESS` (identifier typo from the
hardening pass; every "clever" search for BODYLESS missed it).

Benchmark: new `upload_wire` / `upload_uvicorn` rows (10 MiB
Content-Length, byte-exactness asserted per round, `_UploadApp`
drains through `receive()`). First run exposed a real gap — 64 KiB
per executor round-trip = 160 thread↔loop hops → 201 MiB/s vs
uvicorn 612 (3.4×). Raising the read to 1 MiB/chunk (constant
`_BODY_READ_CHUNK`) → **488 MiB/s (1.55×)**; 2 MiB gave nothing
(442), proving the floor is the seam handoff itself, not the hop
count — keeping the core sync per the execution philosophy. Final
record at load ~15: upload_wire 35.1 ms p50 / 428 MiB/s vs uvicorn
15.6 ms / 681 MiB/s; all other rows unchanged (A/B −98.7%,
adapter +0.365 ms). Follow-up idea recorded on the card: a bounded
read-ahead queue (N MiB prefetch) would overlap client unload with
app processing — pursue only if a bulk-ingest journey demands it.

Suite **83 green** (77 + 6 streaming), ruff clean. Follow-up noted
on card c24543a4; the remaining ecea49c8 items are unchanged
(SLO_TRANSPORT wiring + journey gates, window-blocked).
