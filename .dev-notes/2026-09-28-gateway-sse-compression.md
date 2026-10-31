---
id: 20260928_071_gateway_sse_compression
status: doing
title: "Card 071 — Edge gateway: SSE compression feature + benchmark class + layering docs"
tags: [gateway, compression, sse, benchmark, docs]
sprint: current
gh: null
---
# Card 071 — Edge gateway SSE compression

## Scope (user-directed)

- **SSE compression as a feature**: `text/event-stream` currently sits in the
  gateway's `SKIP_CT_PREFIXES` skip list — the model token stream, the biggest
  bandwidth item during inference, goes out uncompressed. Remove it from the
  skip list; compress from byte 0 (SSE has no Content-Length → no size gate,
  no peek/hold → no TTFB cost). A/B lever already exists: omit `Accept-Encoding`
  → identity, send it → gzip/zstd.
- **Benchmarking**: `scripts/benchmark_gateway_compression.py` only covers the
  "file download" payload class. Add an SSE token-stream class and record it.
- **Architecture docs (concise)**: `docs/engineering-overview.md` §5 — write the
  edge → sidecar → domain layering down, including the edge-owns-compression
  invariant (`accept-encoding: identity` toward the sidecar, one encoder, one
  hop, no double-encode).
- **Convention audit**: gateway passes app contract headers
  (`Authorization`, `X-Correlation-ID`, envelope bodies, `Retry-After`) —
  strips only the 8 RFC 9110 hop-by-hop headers; CORS allow-any matches
  bearer-token auth. Findings go in the docs; no code fix expected.

## Status

- **done** — branch `feat/gateway-sse-compression`, commits `48d791cef`
  (SSE compressible + 3 tests + docs layering) and `2d40fac65` (benchmark
  class + recorded run), after two concurrent-session sweeps reverted
  in-flight files (index re-synced via `git checkout HEAD --`).
- Verified: cargo test 22/22, cargo fmt clean, ruff clean,
  `smoke_gateway_compression.sh` SMOKE_DONE, live SSE wire test
  31600 B → gzip 1145 / zstd 559 lossless, no TTFB hold.
- Bench: 8 MiB token stream → gzip 95.8% savings, zstd 98.7% savings.
- Open: edge wiring (cascade behind node edge-proxy vs Rust takes :8080)
  undecided; landing undecided (branch also carries concurrent VM commits).
