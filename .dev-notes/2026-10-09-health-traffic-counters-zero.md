---
title: Health traffic counters pinned at zero — middleware wrote the wrong sink
date: 2026-10-09
status: done
branch: feat/aml-stream-loader
tags: [health, metrics, observability, bug, api, doctor, mole]
---

# Problem

`/health/detailed` reports `request_count: 0`, `error_count: 0`,
`p95_latency_ms: 0.0` and the diagnosis *"No requests yet — all clear."* on a
server with hours of real traffic. Mole's `api.traffic` finding inherits it and
grades **ok** every run, so the check is structurally blind: it can never fail.

# Root cause

Two sinks, one writer, and the writer picked only one:

- `MetricsMiddleware.dispatch` (`apps/api/server/infrastructure/middleware.py`)
  calls `collector.record_request(path, status, elapsed)` — **`MetricsCollector`**.
- `controllers/health.py:959` reads `ss.request_count`, `ss.error_count`,
  `ss.get_p95_latency()`, `ss.get_path_latencies()`, `ss.get_error_history()`
  — **`ServerState`**.

All four `ServerState` writers have **zero production callers**:
`record_request`, `record_error`, `record_request_latency`, `record_error_detail`.
The only writes are the initialisers. The data existed all along — in the wrong
object.

**Why the suite stayed green:** `test_dashboard_router.py` and
`test_health_router.py` substitute a `MagicMock` for `ss` (or a literal dict)
and inject `request_count = 150` directly. `request_count` is a read-only
property on the real object, so a real instance could never have been used.
`test_middleware.py::TestMetrics` covers the collector path but had no
server_state twin — the seam was the one untested edge.

# Fix

`MetricsMiddleware` mirrors each finished request into `ServerState` via a new
`_record_health_state` helper: `record_request()` + `record_request_latency(...)`
always, `record_error_detail(...)` on >= 400 (which also bumps `_error_count`,
so no double count). No new endpoint, no contract change, no shared-package
change — one seam, the two existing consumers.

# Verification

- New tests in `apps/api/server/tests/test_middleware.py::TestMetrics` exercise
  the seam for real (no mocks): request mirrored, error mirrored to history,
  p95 non-zero.
- **A/B proof the tests are non-vacuous** (3 requests: 2x `/ok`, 1x 404),
  seam disabled vs. wired:

  | | request_count | error_count | p95 | history | errors |
  |---|---|---|---|---|---|
  | before | 0 | 0 | 0.0 ms | empty | empty |
  | after | 3 | 1 | 0.5 ms | `/fail 404` | `/fail 404 msg='Not Found'` |

- **Cost** (20k ops): 1.82 µs/op success, 3.06 µs/op error — ~0.05% of a 3.4 ms
  request. Two extra lock acquisitions per request on the hot path.
- **No memory growth**: `_request_history` is `deque(maxlen=500)` and
  `_error_history` is `deque(maxlen=20)` — verified empirically (500 entries
  retained after 40,000 appends).
- Gates: ruff clean; `apps/api/server/tests` **1463 passed** (1460 + 3 new).

# Landed

`864c449db` on `feat/aml-stream-loader`. **Not yet live**: the `:8000` server
runs from the *main* worktree (`cwd = …/sloughGPT`), so this reaches the running
stack only after the branch merges to main and the server restarts. No doc or
shared index needed — bug repair, not a new app/package/feature.
