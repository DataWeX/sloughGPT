---
name: cleanup-agent
description: Searches for code-quality issues when the todo list is empty or the user says "continue building". Finds f-string logging, bare excepts, a missing future import, print() in production code, TODO comments, and unused imports.
---

# Cleanup Agent

Searches for code quality issues when todo list is empty or user says "continue building".

## Tooling note — `rg` is NOT installed on this machine

Every `rg` command below dies with `command not found`, and `| wc -l` then
reports a phantom **0** — a clean-looking lie. Check first, fall back to grep:

```bash
command -v rg || echo "use grep"
grep -rnE '<pattern>' --include='*.py' --exclude-dir=node_modules --exclude-dir=__pycache__ <dirs>
# ALWAYS print a scope control first, so a zero is proven real:
find <dirs> -name '*.py' | wc -l
```

A count of 0 means nothing until the scope control shows files were scanned.

## Ratchet first

`tests/test_code_hygiene.py` encodes patterns 1–4 as zero-tolerance tests
(with reviewed exemptions). Run it before any manual sweep — it is the
fastest signal, and new violations that reach the tree fail it.

## How to Use

When there are no active todos, run this agent to find and fix common issues.

## Patterns to Search

All commands below are validated GNU-grep/ruff versions (see tooling note —
every original `rg` line silently fails here).

### 1. f-string log calls (should be lazy %s)
```bash
grep -rnE '(logger|logging)\.(info|warning|error|debug)\(f['\''"]' --include='*.py' --exclude-dir=__pycache__ domain packages apps scripts
```
Ratcheted: `tests/test_code_hygiene.py::test_no_fstring_logger_calls` (zero as of 2026-10-04; fixed lr_schedulers.py + avion/session.py that day).

### 2. Missing `from __future__ import annotations`
```bash
grep -rL 'from __future__ import annotations' --include='*.py' --exclude-dir=__pycache__ domain packages/core-py
```
STATUS (2026-10-04): open scope decision on card `0d4ad67b` — 860 of 1665 files missing; no explicit mandate exists (PYTHON_FIRST.md only exemplifies it), ruff cannot enforce it. NOT ratcheted; do not mass-edit without the user's scope pick.

### 3. Bare except clauses
```bash
grep -rnE 'except[[:space:]]*:' --include='*.py' --exclude-dir=__pycache__ domain packages apps scripts
```
Note: ruff's E722 is disabled in `pyproject.toml` — the ratchet test and this scan are the only guards. Zero as of 2026-10-04.

### 4. print() in production code — CHECK CONTEXT FIRST
`print()` in TUI renderers, CLI tools, protocol handshakes (`ENGINE_READY`),
`file=sys.stderr` bootstrap warnings, `__main__.py` entry points, and
argparse `main()` report runners is **correct** — stdout IS the product
surface there; converting it to logger would be a bug. Only flag prints in
pure library paths.
```bash
grep -rn '^[[:space:]]*print(' --include='*.py' --exclude-dir=__pycache__ domain
```
Ratcheted with file-level exemptions + reasons:
`tests/test_code_hygiene.py::test_print_only_in_exempted_stdout_channels`.
Fix the code or get an exemption reviewed — never silence the test.

### 5. TODO/FIXME/HACK/XXX comments
```bash
grep -rnE '(TODO|FIXME|HACK|XXX):' --include='*.py' --include='*.ts' --include='*.tsx' --exclude-dir=node_modules --exclude-dir=__pycache__ --exclude-dir=dist domain packages apps scripts
```
Ratcheted; kanban cards are the todo system, markers rot in the tree. Zero as of 2026-10-04.

### 6. Unused imports
```bash
ruff check --select F401 domain packages/core-py
```
Zero as of 2026-10-04 (ruff 0.16.8). The old `rg '^import '` line never measured anything — it just listed imports.

## Workflow

1. Check todo list (todowrite)
2. If empty, run the ratchet first: `pytest tests/test_code_hygiene.py`, then the pattern scans above
3. For each issue found:
   - Create a todo item
   - Fix the issue
   - Mark as done
4. Update kanban board if needed

## Kanban Integration

When creating cleanup tasks, add to kanban:
```json
{
  "id": "cleanup_<date>_<issue>",
  "title": "Fix <issue type>",
  "description": "<details>",
  "column": "todo",
  "priority": "low",
  "tags": ["cleanup", "code-quality"]
}
```

## AGENTS.md Rules

- Follow all rules in AGENTS.md
- Never delete user data files
- Check in before making changes
- Use Python developer skill patterns
