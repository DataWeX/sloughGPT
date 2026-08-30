---
description: >
  Run test coverage improvement. Usage: /coverage <module_path>
  Measures coverage, writes tests, finds bugs, and verifies no regressions.
agent: coverage
---

# Coverage Command

Systematic test coverage improvement for Python modules.

## Usage

```
/coverage <module_path>
```

## Examples

```
/coverage packages/core-py/domains/training/export.py
/coverage packages/core-py/domains/infrastructure/task_queue.py
/coverage packages/core-py/domains/inference/slonet_provider.py
```

## What It Does

1. **Measures** current line coverage for the target module
2. **Writes** tests to close every coverage gap
3. **Finds** real bugs exposed by the new tests
4. **Fixes** bugs with minimal changes
5. **Pragmas** genuinely unreachable branches
6. **Verifies** no regressions
7. **Reports** results in standard format

## Workflow

### Step 1: Measure
```bash
python3 -m pytest tests/test_<module>*.py \
  --cov=domains/<area>/<module> \
  --cov-report=term-missing -q
```

### Step 2: Write Tests
For each uncovered line:
- Read the source to understand the uncovered code
- Read existing tests to match patterns
- Write targeted tests for the uncovered paths
- Run new tests to confirm they pass

### Step 3: Find & Fix Bugs
Watch for:
- `KeyError` → missing `.get()` with default
- `AttributeError` after `hasattr` → value is `None`
- `TypeError` → parameter name mismatch
- Dead code after return → unreachable branches

### Step 4: Pragma Unreachable
Only pragma provably unreachable branches:
```python
try:
    from optional_lib import something
except ImportError:
    pragma: no cover  # Optional dependency
```

### Step 5: Verify
```bash
python3 -m pytest tests/test_<module>*.py -q
python3 -m py_compile domains/<area>/<module>.py
```

### Step 6: Report
```
ROUND N: domains/<area>/<module>.py 100% (<total> stmts)
- Tests added: <count>
- Coverage: <before>% -> <after>%
- Bugs found: <count>
- Pragma'd: <count> unreachable branches
- Regression: <test count> passed
```

## Coverage Targets

| Priority | Area | Path |
|----------|------|------|
| 1 | Training | `domains/training/*.py` |
| 2 | Infrastructure | `domains/infrastructure/*.py` |
| 3 | Inference | `domains/inference/*.py` |
| 4 | Feedback | `domains/feedback/*.py` |
| 5 | Agents | `domains/agents/*.py` |

## Rules

- Never skip measurement — know the baseline first
- Never add tests that don't increase coverage
- Never pragma a branch that could execute in production
- Always match existing test patterns
- Always verify no regressions
- Always fix real bugs when found
- Stop at 95%+ if remaining misses are unreachable
