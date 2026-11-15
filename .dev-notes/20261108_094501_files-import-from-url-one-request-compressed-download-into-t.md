---
id: 20261108_094501_files-import-from-url-one-request-compressed-download-into-t
status: done
board: 20261108_094501_files-import-from-url-one-request-compressed-download-into-t
title: "files: import from URL — one-request compressed download into the file manager with honest progress %"
tags: ["files", "downcraft", "web"]
sprint: ""
gh: null
---

# files: import from URL — one-request compressed download into the file manager with honest progress %

Server-side downcraft fetch (SSRF-guarded: scheme allowlist, private-range block, size cap via on_chunk raise) + files.download_url/files.download_status descriptor endpoints + /files page Import-from-URL affordance with decoded/decoded pct bar and final WireStats savings line. Progress in disk byte-space (invariant: never mix wire into the bar — same lesson as Range identity pin). Tests: single-request assert, pct honesty, SSRF rejections, vitest progress line. Landed d4c0ed398 on origin/main. Scope: files.py POST /files/import + GET /files/import/{job_id} (in-memory bounded job registry, SSRF guard — http(s) only, non-global addresses blocked, 50 MB cap raised from on_chunk with part+dest cleanup) and /files page Import URL row + 500ms polling Progress (decoded/decoded pct, indeterminate while total unknown) + final wire-savings receipt, all API calls via http-client. Gates green: ruff, 47 server tests (single-request assert, gzip decoded-space totals, mid-run total never wire, SSRF rejections, size-cap cleanup), 58 adjacent py tests, 50 web vitest, tsc clean. Not touched: downcraft, RAG ingest, model download. Follow-up landed9b8b205b2 + b708347f5 + 290f4d12a: _job_put pruner wired (registry was unbounded — promise-only bound fixed, running jobs never evicted, 48/48 server tests) and scripts/benchmark_files_import.py (SOP create-one clause): floor/pipe/e2e × identity/gzip, 23/23 invariants — one GET per import, sha256 storage, pct==100 only at done, gzip saves 68.9%, identity exactly 0; 8 MiB identity 1009/647/209 MiB/s, status poll ~1ms mean; history in scripts/benchmark_files_import.json. WAN row landed c90b1dba0: opt-in --wan/--wan-url/--wan-repeat adds an informational real-network e2e row (default = our own repo via raw.githubusercontent/Fastly; variance never gates, loopback stays the baseline). Integrity without a known payload = repeat-sha stability (--wan-repeat >= 2); one-request stays pinned by loopback rows. First WAN run 28/28 checks (23 loopback + 5 WAN): 663741 B done==total, pct==100, gzip 59.0% saved, repeat-sha stable 8ea7ba5b.
