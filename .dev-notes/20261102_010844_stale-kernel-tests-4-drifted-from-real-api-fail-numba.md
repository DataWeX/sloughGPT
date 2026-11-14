---
id: 20261102_010844_stale-kernel-tests-4-drifted-from-real-api-fail-numba
status: done
board: 20261102_010844_stale-kernel-tests-4-drifted-from-real-api-fail-numba
title: "Fix 4 stale kernel tests that drifted from the real numba API (only fail under SLO_USE_NUMBA=1)"
tags: [tests, kernels, numba, stale-test]
sprint: ""
gh: null
---
# 4 stale kernel tests — drifted from the real kernel API (numba path only)

## Problem

Four tests in `tests/core-py/test_slonet_kernels.py` fail whenever the
compiled numba kernels are active (`SLO_USE_NUMBA=1`). They passed for years
only because the numpy fallbacks are different call shapes; since
`b70a0d085` made numba opt-in, they fail under opt-in and **cannot catch
regressions in the path they were written for** — they were silently masking
kernel drift. Under the default (numpy) path they are green (128 passed /
1 skipped / 0 failed), so CI does not see them; they only bite anyone who
opts in.

## The 4 failing tests (observed 2026-11-01, `SLO_USE_NUMBA=1` → 4 failed, 27 passed)

1. `TestNbEmbed::test_basic` — `AttributeError`
2. `TestNbEmbed::test_clipping` — `AttributeError`
3. `TestNbAddPos::test_basic` — `AttributeError`
4. `TestLmHeadArgmax::test_basic` — `nopython_type_inference` failure

## Diagnosis (preliminary)

- `nb_embed`: the test expects a return value / no `out` argument, but the
  real kernel writes into a preallocated `out` and returns nothing — test
  written against an older signature.
- `TestLmHeadArgmax::test_basic`: passes a 1-D input where the kernel wants
  2-D (and type inference rejects it).

## Definition of done

- All 4 tests assert against the CURRENT kernel API (or are deleted if they
  duplicate coverage that exists elsewhere — check redundancy first per
  Test Rule §2).
- `SLO_USE_NUMBA=1 pytest tests/core-py/test_slonet_kernels.py` → 31 passed /
  0 failed.
- Default-path suite stays 128 passed / 1 skipped / 0 failed.
- Run through `scripts/run_affected.sh`; seam gate `test_slonet_kernels.py`
  rides along since it IS the surface.

## Notes

- Discovered while landing `a6c8f3b31` + `b70a0d085` (prefill-OOM fix + numba
  opt-in). Pre-existing drift — verified failing identically at HEAD before
  those commits, so not a regression from them.

CLOSED 2026-11-12 — fixed all 4 tests to the real kernel API (in-place embed/add_pos buffers, observable clipping, 2-D lm_head x + permuted-W 2nd assertion); RED 4F/27P → GREEN opt-in 31P/0F, default 30P/1S (the 128 figure above was stale); closure 7 failed / 30 skipped / 0 passed / 1340 deselected in 669.73s (all 7 triaged zero-attributable: 4 documented ledger, vm_life load-flake green isolated 7.09s, datasets ImportUrl x2 = pre-existing main regression from e9973ce4e carded 20261112_062928); kernels tests recorded green in the killed first pass; pushed `fix/stale-kernel-tests` ed3514237 + 0394c2af5.
