---
description: >
  Run documentation generation or update. Usage: /doc <target>
  Generates or updates documentation for the specified module or component.
agent: doc-aware-engineer
---

# Doc Command

Generate or update documentation for code.

## Usage

```
/doc <target>
```

## Examples

```
/doc packages/core-py/domains/infrastructure/model_server.py
/doc apps/web/features/chat/components/ChatInput.tsx
/doc packages/core-py/domains/training/
```

## What It Does

1. **Read** — Understand the code structure and public APIs
2. **Document** — Add/update docstrings, type hints, and comments
3. **Generate** — Create or update markdown documentation
4. **Verify** — Ensure documentation matches implementation

## Documentation Standards

### Python
- Module-level docstring: purpose, usage, examples
- Class docstring: attributes, methods, usage
- Method docstring: parameters, return, exceptions
- Type hints on all public methods

### TypeScript
- JSDoc on exported functions and components
- Type definitions for all props
- Usage examples in comments

## Output

The agent will report:
1. Files documented
2. Documentation added/updated
3. Verification results
