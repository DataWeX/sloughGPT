---
id: 20261102_110421_unify-processevent-contract-fields-one-event-schema-source-g
status: done
board: 20261102_110421_unify-processevent-contract-fields-one-event-schema-source-g
title: "Unify ProcessEvent contract fields — one EVENT_SCHEMA source, gen_ts_contracts projection, drop dead resolved"
tags: ["contract", "frontend", "backend"]
sprint: ""
gh: null
---

# Unify ProcessEvent contract fields — one EVENT_SCHEMA source, gen_ts_contracts projection, drop dead resolved

Expert-reviewed + user go 2026-11-02. DRIFT MEASURED: (1) frontend declares resolved? never emitted by backend (dead field, lying comment); (2) backend emits contract_version per event, frontend ProcessEvent lacks it; (3) kind:string vs constant training; (4) intent union frontend vs free-form dict backend; (5) state/event vocab implicit in 3 places (_ACTIONS_BY_STATE, _TITLES, dock STATE_TONE). MECHANISM: EVENT_SCHEMA declaration in training/process_events.py (field -> TS type -> required/optional) + ACTION_INTENTS/STATES/EVENTS vocab; gen_ts_contracts.py gains a second source section emitting ProcessAction/ProcessEvent into all 3 .gen.ts artifacts (freshness --check rides existing gate); frontend re-exports generated types via lib/protocol/types.ts. Python parity test: make_event keys == EVENT_SCHEMA keys. STATE_TONE keyed by declared states + full-coverage test. SCOPE LOCK: wire bytes IDENTICAL, stays CONTRACT_VERSION=1 (declaration-only; renames = v2, separate). NOT touched: endpoints, SSE, dedupe/dispatch, dispatched/stale (client-local UI state). Gates: gen --check, backend 58, startup 54, tsc, eslint delta, vitest closure, run_affected, dock screenshot zero visual change.
