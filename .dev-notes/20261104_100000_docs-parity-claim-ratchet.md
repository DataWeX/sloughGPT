---
title: Docs parity claim block drifts when controllers change (auto-refresh missing)
date: 2026-11-04
tags: [ci, docs, ratchet, parity]
status: todo
---

# Docs parity claim block drifts when controllers change

## Problem
`tests/test_docs_api_parity.py` pins `controllers_lines=2918` in
`docs/DOC_VS_CODE_GAP_AUDIT.md` by hand. When a controller grows, the claim
is not refreshed and test-root goes red **after** the change already landed
on main.

Evidence:
- Main `289ca6686` (datasets structured listing, +14/−4 in
  `controllers/datasets.py`) landed with the claim still at 2908 — main's own
  push CI red for 4+ consecutive runs (`41792487465`, `41785357332`,
  `41780933728`…).
- PR branch inherited it in merge-20 (controllers byte-identical to main),
  failed dispatch `41786852868` test-root; fixed by hand in `52ca4a52e`
  (2908→2918, same class as batch-9's 2844→2908 refresh).
- Root cause recorded twice in the block's own comment history — the ratchet
  is doing its job, but only reactively.

## Proposed fix (choose one)
1. **Auto-refresh in the check**: `scripts/check_docs_api_parity.py
   --write-claims` regenerates the claim block from measured values; CI keeps
   `--check` (fail on drift), contributors run `--write-claims` when touching
   `apps/api/server/controllers/`. Cheap, mechanical, no policy change.
2. **Pre-merge guard**: a tiny `git diff --name-only origin/main...
   apps/api/server/controllers/` check in the PR workflow that reminds (or
   fails) when the claim block wasn't touched in the same PR.
3. Lower the claim to architecture-only truth (drop exact line counts) —
   weaker ratchet, only if 1/2 prove noisy.

Recommend 1 + a one-line note in AGENTS.md's test rule.

## Notes
- This card is main-side: fix lands via a small PR to main (or rides the
  next main-touching lane), not through merge/main-into-npu.
