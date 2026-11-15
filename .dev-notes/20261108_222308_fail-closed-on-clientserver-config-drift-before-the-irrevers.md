---
id: 20261108_222308_fail-closed-on-clientserver-config-drift-before-the-irrevers
status: done
board: 20261108_222308_fail-closed-on-clientserver-config-drift-before-the-irrevers
title: "Fail closed on client/server config drift (before the irreversible action)"
tags: ["api", "frontend", "security", "tests"]
sprint: ""
gh: null
---

# Fail closed on client/server config drift (before the irreversible action)

SCOPE-ADD from the FungoLabs gap audit (docs/misc/FUNGOLABS_LAUNCHPAD_ANALYSIS.md pattern #9) — belongs with card 20261108_134211_runtime-config-descriptor-for-web-client-flagslimitsendpoint (that CLI has no board update, so it is its own card; close them together).

PATTERN: FungoLabs halts a fund transfer when the client's contract and the platform's disagree — 'This page and the platform name different Genesis pools, so nothing is sent from here. Reload the page.' Fail CLOSED before the irreversible action, instead of degrading.

OUR POSITION (from the audit): for GENERATED contracts we already have the STRONG version — packages/sdk-ts contracts.gen.ts + apps/web/lib/protocol/contracts.gen.ts are freshness-gated by tests/contract/test_ts_contracts.py, so a drift fails the BUILD. What we have nothing for is HAND-WRITTEN config/limits/flags: a stale cached /bootstrap or a client hardcoding a limit the server has changed will silently mismatch at runtime.

GAP TO CLOSE: when the versioned client config (see the sibling card) and the server's current values disagree — or the client presents a config version the server no longer knows — the call/state must fail closed with a clear sentence (our own equivalent of 'Reload the page'), not fall back to the stale value. Applies to anything the client enforces locally (limits, feature flags, fee-like constants).

DO NOT: touch generated-contract gating (already correct), do not block on mismatch for pure display values (colours, labels) — only for values that gate an action.

VERIFY: test that serves config version N while the client holds N-1 and asserts the action is refused with an explicit message; plus a test that a matching version proceeds. Sibling card: 20261108_134211_runtime-config-descriptor-for-web-client-flagslimitsendpoint.
