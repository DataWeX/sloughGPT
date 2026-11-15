---
id: 20261107_105147_downcraft-wire-savings-reporting-wire-vs-decoded-bytes-cli-s
status: done
board: 20261107_105147_downcraft-wire-savings-reporting-wire-vs-decoded-bytes-cli-s
title: "downcraft: wire-savings reporting (wire vs decoded bytes), CLI savings display, multipart Range x Accept-Encoding audit"
tags: ["downcraft", "compression"]
sprint: ""
gh: null
---

# downcraft: wire-savings reporting (wire vs decoded bytes), CLI savings display, multipart Range x Accept-Encoding audit

Landed 52730d595 on origin/main (single-commit cherry-pick of 07a5de74c — the lane branch was cut from wip/avion-observation and carried another session's avion commits, so a rebase hit a docs/AGENT_SYNC.md conflict mid-stack and was aborted). AGENT_SYNC entry 6776902b0. Gates green: 497 downcraft tests (15 new in tests/test_wire_stats.py), ruff check clean, format at pre-existing baseline, scripts/benchmark_downcraft_compression.py rerun (sha256 ok all rows). Scope landed: WireStats wire-vs-decoded accounting (wire counted up-front from Content-Length, decoded flushed incrementally from disk, saved_* None instead of invented), download() result keys, CLI wire/savings line, multipart aggregation, download_range Accept-Encoding: identity pin + content-coded 206 rejection.
