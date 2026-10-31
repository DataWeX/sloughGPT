---
id: 20260924_039_legacy_torch_benchmark_scripts
title: Legacy torch benchmark scripts — reference audit (NO delete)
status: done
sprint: current
gh: ""
body: >
  Audit-only: prove whether the 5 scripts under packages/core-py/scripts are
  referenced anywhere (CI, docs, tests, Makefile, packaging). Do NOT delete.
  Deletion requires a separate explicit user approval.
---

# 039 audit — legacy torch benchmark scripts

## Targets
1. packages/core-py/scripts/benchmark_clean.py
2. packages/core-py/scripts/benchmark_final.py
3. packages/core-py/scripts/benchmark_quant_kernels.py
4. packages/core-py/scripts/benchmark_vs_torch.py
5. packages/core-py/scripts/train_sloran.py

## Status
- [x] Claimed
- [x] Repo-wide reference scan
- [x] CI/packaging/docs scan
- [x] torch import profile
- [x] Findings + recommendation (no delete)

## Claimed
2026-09-24T07:40:34.763093+00:00


## Audit findings (2026-09-24 07:41 UTC)

### References (N=5 scripts)
| Script | Basename refs elsewhere | CI/docs/tests | torch required? |
|--------|------------------------|---------------|-----------------|
| benchmark_clean.py | **0** | **0** | No — try/except soft |
| benchmark_final.py | **0** | **0** | No — try/except soft |
| benchmark_quant_kernels.py | **0** | **0** | **No torch at all** (numpy/C kernels) |
| benchmark_vs_torch.py | **0** | **0** | Soft at top; **hard `import torch` at L200** inside a callable |
| train_sloran.py | **0** (packages copy) | **0** | **No torch at all** |

### Packaging / CI
- `packages/core-py/pyproject.toml` **excludes** `["tests*", "scripts*", "benchmark*"]` from the distribution — scripts are **not installed**.
- Only path hit: root `pyproject.toml:118` ruff per-file-ignore glob `packages/core-py/scripts/*.py` (directory-level; remains valid if files go).
- **No** references in `.github/`, Makefile, setup.py, package.json, docker, or tests.
- **No** markdown/rst docs reference them (outside `data/rag_json` content dump).

### Correction to card body
Card said all 5 “import torch/torch.nn unconditionally”. Actual:
- **2/5 never import torch** (`benchmark_quant_kernels`, `train_sloran`).
- **2/5 soft-import** under `try/except ImportError` (`benchmark_clean`, `benchmark_final`).
- **1/5** (`benchmark_vs_torch`) soft at module top **plus** a hard import at L200.

### Duplicates / leftovers
- **Separate** root `scripts/train_sloran.py` exists (not in the delete list; also unreferenced by basename outside itself + rag dump).
- `.wt-merge/packages/core-py/scripts/` still holds pre-cutover copies (other card / worktree cleanup).

### Recommendation (NO delete performed)
1. **Safe to git-delete** the 5 files under `packages/core-py/scripts/` — zero functional references, excluded from packaging.
2. Optional follow-ups (need separate approval each):
   - Delete or keep root `scripts/train_sloran.py` independently.
   - Drop or narrow the ruff ignore on line 118 after deletion.
   - Clean `.wt-merge` copies when that worktree is retired.
3. `benchmark_quant_kernels` is the only script that is **not** torch-era; if kernel A/B is still useful, consider promoting a copy to `scripts/` **before** any delete.

**Status: audit complete. Deletion NOT executed — awaiting explicit user approval.**
