---
id: 20261101_113754_chorecleanup-retire-packagescore-py-vestigial-husk-14-tracke
status: done
board: 20261101_113754_chorecleanup-retire-packagescore-py-vestigial-husk-14-tracke
title: "chore(cleanup): retire packages/core-py — vestigial husk (14 tracked files, no importable domain/)"
tags: ["cleanup", "core-py", "debt"]
sprint: ""
gh: null
---

# chore(cleanup): retire packages/core-py — vestigial husk (14 tracked files, no importable domain/)

EVIDENCE (measured 2026-11-01):
- Only 14 git-tracked files under packages/core-py/ (README.md, __init__.py, conftest.py, conftest_debug.py, benchmark_backends.py, pyproject.toml, config/defaults.yaml.example, sloughgpt.db, + 5 one-off scripts under scripts/).
- NO importable domain/ (singular). pyproject declares include=["domain*"] but the package it points at does not exist as importable source.
- packages/core-py/domains/ (plural) holds 2 untracked .so artifacts + __pycache__; 0 tracked .py files.
- 0 test .py files under packages/core-py/tests/ (only __pycache__).
- The real core stack is repo-root domain/ (660 .py files).

IMPACT: pyproject is unsatisfiable (declares a package that isn't there); conftest.py imports tests.conftest which has no test files; the 5 one-off scripts + benchmark_backends.py are the only potentially-live code and need individual confirmation before deletion.

PLAN: confirm each of the 5 scripts + benchmark_backends.py is dead (grep for callers/references), then delete the directory. DO NOT bulk-delete — File Safety rule: confirm the scripts are unreferenced first. Out of scope for the provenance card (20261101_113742).
