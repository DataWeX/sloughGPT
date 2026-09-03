# Cleanup Agent

Searches for code quality issues when todo list is empty or user says "continue building".

## How to Use

When there are no active todos, run this agent to find and fix common issues.

## Patterns to Search

### 1. f-string log calls (should be lazy %s)
```bash
rg 'logger\.(info|warning|error|debug)\(f"' packages/core-py/apps scripts
```

### 2. Missing `from __future__ import annotations`
```bash
rg -L 'from __future__ import annotations' packages/core-py/domains/**/*.py
```

### 3. Bare except clauses
```bash
rg 'except:' packages/core-py
```

### 4. print() in production code (should use logger)
```bash
rg '^\s*print\(' packages/core-py/domains --include '*.py'
```

### 5. TODO/FIXME/HACK comments
```bash
rg '(TODO|FIXME|HACK|XXX):' packages/core-py
```

### 6. Unused imports
```bash
rg '^import ' packages/core-py/domains | head -20
```

## Workflow

1. Check todo list (todowrite)
2. If empty, run cleanup agent
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
