---
id: 20261108_134212_server-authoritative-clock-for-leasettleta-computation
status: done
board: 20261108_134212_server-authoritative-clock-for-leasettleta-computation
title: "Server-authoritative clock for lease/TTL/ETA computation"
tags: ["api", "backend", "tests"]
sprint: ""
gh: null
---

# Server-authoritative clock for lease/TTL/ETA computation

Source: fungolabs launchpad analysis (2026-11-08) — GET /api/time returns {now}, it is fetched into the query cache and every 'opens in / sales on hold' label derives from it; drops carry clock:{lag,held}. Client Date.now() skew breaks auctions, and the same class of bug affects our lease/TTL/ETA/relative-time logic (apps/web/lib/time-ago.ts, app-shell, protocol/store, api-monitor-store all use Date.now()). PLAN: (1) grep the repo for Date.now() usages that gate behaviour (not display) and list them. (2) Expose one server time source (existing endpoint or a clock field on an existing envelope) and a single offset helper (serverNow - Date.now()) in apps/web/lib. (3) Convert gating usages to the helper; leave pure display formatting alone unless it drifts. VERIFY: unit test that fakes a 90s skew and asserts no lease/TTL path expires early/late; affected suites green.
