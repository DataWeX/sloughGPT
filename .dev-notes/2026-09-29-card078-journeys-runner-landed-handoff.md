---
id: 20260929_078_journeys_runner_landed_handoff
title: "Handoff — spec-linked journey runner landed on main as domain/journeys; re-target markers/tests; watcher dead"
status: todo
date: 2026-09-29
tags: [coordination, journeys, landing]
---

# Handoff: journey runner = `domain/journeys` (landed on main)

Owner: the `feat/journey-library` session (note `2026-09-28-journey-library.md`).
Written by the `feat/ux-flows-library` session, 2026-09-29 05:1x, with user
approval (landing plan: zero-overlap cherry-picks, their staged work preserved).

## What landed on main (was `74f392f5e`, now +6 commits)

| SHA | Content |
|---|---|
| `ba5ea1e59` | hf-hub disk-truth fix (card 074 input: stale-state husk) |
| `a7bc47829` | card 074 fix — fused QKV quant-aware fallback, 4 TDD tests |
| `2f87f5aee` | card 077 fix — seeded `tiny_model` fixture, kills kvq suite flake |
| `4e4b2fb11` | journeys library `domain/testing/ux_flows` (initial) |
| `a2edaf02f` | re-home → `domain/journeys` (main deleted `domain/testing` as fake-green) |
| `f1ecf8a6f` | depth-independent repo-root discovery in steps bootstrap |

Evidence: content diff vs `feat/ux-flows-library` empty on all touched paths;
`tests/test_ux_flows_library.py` 5/5; ruff clean; phase-1 suites (hf_hub +
fused_qkv + quant integration) green on main's tree; **their 29-file staged
set in sloughgpt-api-std byte-identical before/after**
(`~/.cache/slog-journeys/card076-landing-before.txt`).

## The decision (user-approved)

**One spec-linked runner: `python -m domain.journeys`** — `domain/journeys/
flows.py` is the registry; every `Flow` carries its `docs/UX_FLOWS.md`
heading anchor; offline registry tests live at `tests/test_ux_flows_library.py`
(no stack needed; live sweep = `SLO_UX_LIVE=1`).

## Action items for the journey-library session

1. **Re-target your unique pieces onto `domain/journeys`** (they do not
   conflict file-wise, but two runners must not coexist):
   - `pytest.ini` / `packages/core-py/pytest.ini` browser + live **markers**
     (genuinely additive — port them);
   - `test_ux_flows_browser.py` — re-point entry to `domain.journeys`;
   - `test_ux_flow_registry.py` contract assertions — fold into
     `tests/test_ux_flows_library.py` if they add anything mine doesn't lock;
   - `scripts/run_ux_flows.py` spec-anchor edits — **superseded**; the script
     can stay as a thin shim or be deprecated, do not restore it as a
     second registry.
2. **The auto-merge watcher is dead** — `merge-watcher.sh` PID 247374
   disappeared ~04:38 with NO `.done`/`.timeout` marker (killed, not
   timed out). Its ff-merge of `feat/journey-library` (`4d02d8a21`) can no
   longer fast-forward anyway — main moved. Retire it; re-queue
   `feat/routers-doc-truthup` (071) independently.
3. `feat/journey-library` itself: after re-targeting, close it as superseded
   (do not merge as-is — it would reintroduce `scripts/run_ux_flows.py` as a
   second spec-anchored runner).

## Untouched / not mine

- The 29-file staged changeset in sloughgpt-api-std (legacy `domain/testing/
  _internal/*` re-adds, corpus_loader deletion, `M scripts/run_ux_flows.py`…) —
  left exactly as found; whoever owns it decides its fate.
- The staged arken-era avion content in the fix-recovery worktree — lineage
  kept for compatibility per user (voyager → arken → avion, aliases remain).
