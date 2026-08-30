---
description: >
  Run OOP refactoring with QA verification. Usage: /architect <file_or_module>
  Refactors the specified code using OOP principles and verifies no regressions.
agent: architect
---

# OOP Architect Command

Refactor the specified code using OOP principles with QA verification.

## Usage

```
/architect <file_or_module>
```

## Examples

```
/architect packages/core-py/domains/infrastructure/model_server.py
/architect packages/core-py/domains/inference/slonet_provider.py
/architect apps/web/features/chat/components/ChatInput.tsx
```

## Workflow

1. **Analyze** — Read the target file and identify OOP anti-patterns
2. **Plan** — Document specific changes and expected improvements
3. **Implement** — Make changes incrementally, one at a time
4. **Verify** — Run syntax checks, imports, and tests after each change
5. **Sign-off** — Get QA approval before completing

## What Gets Refactored

| Pattern | Action |
|---------|--------|
| Duplicated init code | Extract to manager/mixin class |
| No `__slots__` | Add slots to reduce memory |
| God class (>500 lines) | Extract strategy/helper classes |
| Singleton boilerplate | Use `SingletonMeta` metaclass |
| Import in hot path | Move to module level |
| Side-effect properties | Move to explicit methods |
| Dict-based state | Convert to dataclass with slots |

## Verification

After each change:
```bash
python3 -m py_compile <file>
make test-py ARGS="tests/test_<module>.py -x -q"
```

Before completion:
```bash
make test-py
cd apps/web && npm run test:lib
```

## Output

The agent will report:
1. Files changed
2. Changes made
3. Tests run and results
4. Expected memory/CPU savings
5. QA sign-off status
