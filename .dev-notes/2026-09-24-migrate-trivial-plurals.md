---
id: 20260924_054_migrate_trivial_plurals
title: Dual-tree step 2e: flip remaining trivial plural reals to forward shims
status: done
created: 2026-09-24
tags: [architecture, migration, dual-tree]
sprint: current
gh: null
body: |
  Flip the last trivial plural real files to forward shims; landed alongside
  parallel card 054_migrate_final_plurals (root __init__ retarget) which covered
  most of the same scope. Unique deliverables: native C asset relocation +
  training/gpu shim marker.
status_detail: done
---

## Result
- Branch `feat/migrate-trivial-plurals`, worktree `/home/mana/Documents/sloughGPT-knowledge`.
- Commit `bea036ead` (original 7-file flip, +81/−1601) — largely superseded in flight by
  parallel card `20260924_054_migrate_final_plurals` / `bb7b0946c` (root `domains/__init__`
  moved into `domain/__init__.py`, shims for collections/generation/native/slolib/
  huggingface/consciousness, test-package markers).
- Merged main into branch (`9bec700e9`, conflicts → main) + card 055 (`d1d0299bc`);
  surviving delta = **4 files**:
  - `git mv` `libtransformer_forward.dylib` + `transformer_forward.c/.h` from
    `packages/core-py/domains/inference/native/` → `domain/inference/_internal/native/`
    so `bindings._find_lib` (`Path(__file__).parent`) resolves the lib beside itself.
  - `domains/training/gpu/__init__.py` normalized to the standard shim docstring
    (kept exact `get_accelerator`/`__all__` surface).
- Fast-forwarded main to `d1d0299bc`.
- **Gates**: ruff clean; plural reals scan on main **0** (incl. their marker docstrings);
  identity probe 10/10 — root flip verified (`from domains import ComponentException is
  domain.ComponentException`), star-surface parity for slolib/hf/generation, native assets
  present at twin only; main smoke worktree-resolved OK; batch **1616 collected**:
  failure sets **IDENTICAL to 0850db392 baseline** (111F / 1286P+142 new all green / 75E,
  1 deselected) across contract+startup+root+bridge+hf+accelerator+native+consciousness+
  generation+journeys.

## Notes
- **Editable-install quirk:** `sloughgpt` editable maps `domains` → primary checkout
  (`Default Project/sloughGPT/packages/core-py/domains`), not the active worktree.
  All plural-path probes/batches must `sys.path.insert(0, "packages/core-py")` or they
  silently test the primary tree. `domain.*` resolves from cwd (worktree) — fine.
- Pre-existing baseline failures (0 new): test_gpu_accelerator 62, test_native_engine_real
  41 (Mach-O dylib on Linux), test_slolib_gpu 3 (env-dependent vram/cpu_count),
  test_style_task_managers 3, test_context_managers 1, test_startup_preload 1
  (`domains.api.sse_envelope` gap in primary), test_user_journeys 75 errors.
- Root `domains/__init__.py` retarget (originally scoped OUT of this card) was done by the
  parallel card — verified green here (test_domains_init 28 + test_cognitive_processor 23).
- Remaining non-shim plural content: none. `packages/core-py/domains` = pure shim tree.
