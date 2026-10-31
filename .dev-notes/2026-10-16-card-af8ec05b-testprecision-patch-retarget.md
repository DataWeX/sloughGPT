---
id: af8ec05b
title: "test_models_router TestPrecision 1F both trunks: patch targets domain.slolib._internal.gpu.get_accelerator but the endpoint reads domain.slolib.get_accelerator"
status: review
board_id: Main
date: 2026-10-16
branch: fix/models-precision-patch
tags: [tests, models-router, patch-targets, ported]
---

# Card af8ec05b — TestPrecision patch retarget (LANDED)

## Landed — sha e3f8ecf26 on `fix/models-precision-patch` (worktree `.wt-precfix`, base ed6334e17)

- One retarget: `@patch("domain.slolib._internal.gpu.get_accelerator")` →
  `@patch("domain.slolib.get_accelerator")` in
  `tests/server/test_models_router.py::TestPrecision::test_cpu_path_uses_suggestion`,
  with an inline comment naming the read point.
- Verified the read point is function-deferred: `set_precision` does
  `from domain.slolib import get_accelerator` at
  `apps/api/server/routers/models.py:1569` (inside the handler), so the
  package attribute is what it reads — the mock now lands.
- **Ported from `5d0d7e2e7`** (source sha; carries this exact one-liner,
  stranded on 4 branches: feat/cupy-ctk-gpu, feat/decode-device-residency,
  feat/residency-kill-switch, merge-to-main — never reached a trunk). Only
  the precision-test retarget ported; the device-decode half of that commit
  belongs to cards 11ad9f7f family, not here.

## Gates

- `tests/server/test_models_router.py`: **45 passed** (baseline at base
  ed6334e17: 44 passed + 1 failed — red reproduced in a clean worktree
  first).
- `ruff check` + `ruff format --check` on the file: clean.

## Why not merged here

The fix file carries another session's live WIP in the main tree
(download-refactor: `model_id`→`resource_id`, url/dest params — 325 lines
uncommitted), so neither a main-tree edit nor a main-tree merge is legal
(rule 2 + merge would collide with their hunks). The branch sits ready and
rides the next quiet window with the standing review queue
(0767c19f1 power, 95531c7da CORS, e855a3f4a flags-decide, 93271ece6
isolated-gen, a65ab6f2c flag-surface).

## Cross-references

- Same fix class as card **3ef4f1c0** (obsolete @patch target in
  test_kb_router) and **2368d4aa3** (stale patch targets at consumer read
  points) — the recurring pattern: patch where the value is READ, not where
  it is defined.
- Found-during: card **1f5dec62** merge verification.
