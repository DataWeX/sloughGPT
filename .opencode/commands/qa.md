---
description: >
  Run QA verification on changes. Usage: /qa [scope]
  Verifies no regressions after code changes across the full stack.
agent: qa-verifier
---

# QA Verifier Command

Verify no regressions after code changes.

## Usage

```
/qa [scope]
```

## Examples

```
/qa                          # Full verification
/qa python                   # Python only
/qa web                      # Frontend only
/qa packages/core-py         # Specific path
```

## What It Does

1. **Syntax Check** — `python3 -m py_compile` on changed files
2. **Type Check** — `npx tsc --noEmit` for frontend
3. **Unit Tests** — Run affected test suites
4. **Import Check** — Verify no circular imports or missing deps
5. **Regression Check** — Compare before/after behavior

## Verification Commands

### Python
```bash
python3 -m py_compile <file>
python3 -m pytest tests/test_<module>.py -x -q
make test-py
```

### Frontend
```bash
cd apps/web && npx tsc --noEmit
npm run test:lib
npm run test:components
```

## Output

The agent will report:
1. Files checked
2. Tests run and results
3. Issues found (if any)
4. Sign-off status (PASS/FAIL)
