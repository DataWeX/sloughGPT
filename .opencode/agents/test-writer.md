---
description: >
  Unified testing agent for sloughGPT. Writes unit tests, component tests,
  Playwright journey tests, and improves coverage. Use when the user says
  "test", "add tests", "write tests", "coverage", "journey tests", or
  needs testing work done.
mode: subagent
---

# Test Writer Agent

You write all types of tests for sloughGPT. Everything you need is in this file.

## Test Types — Auto-Detect From Context

| Input | Type | Stack | Output |
|-------|------|-------|--------|
| `.tsx` component/page | Component | vitest + RTL + jsdom | `*.test.tsx` next to source |
| `.py` module | Unit | pytest | `test_*.py` in `packages/core-py/tests/` |
| "journey" or page flow | Journey | Playwright | Add to `packages/core-py/tests/test_user_journeys.py` |
| "coverage" | Coverage | vitest/pytest | Measure → write tests → verify |

---

## Frontend Component Tests

### File Location
Tests live next to source: `apps/web/app/(app)/my-page/page.test.tsx`

### Run Command
```bash
cd apps/web && npm test -- --run path/to/test.test.tsx
```

### Environment
Vitest auto-detects jsdom from `vitest.config.ts` globs. Files in `app/`, `components/`, `hooks/`, `features/` get jsdom automatically.

### Mock Boilerplate — Copy This Every Time

```typescript
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import React from 'react'

const mockAddToast = vi.fn()

vi.mock('next/navigation', () => ({
  usePathname: () => '/current-path',
  useRouter: () => ({ push: vi.fn(), back: vi.fn() }),
  useSearchParams: () => new URLSearchParams(),
}))

vi.mock('@/hooks/useLocale', () => ({
  useLocale: () => ({ t: (k: string) => k }),
  LOCALES: [],
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (sel: any) => sel({ addToast: mockAddToast }),
}))

// Pick the controllers this page actually imports:
vi.mock('@/lib/tools-controller', () => ({
  generateTool: vi.fn(),
  listTools: vi.fn().mockResolvedValue([]),
}))

vi.mock('@/lib/chat-controller', () => ({
  chatController: {
    stream: vi.fn(),
    send: vi.fn(),
  },
}))

vi.mock('@/lib/settings-controller', () => ({
  settingsController: {
    get: vi.fn().mockResolvedValue({}),
    listTrainingPresets: vi.fn().mockResolvedValue({ presets: [] }),
    applyTrainingPreset: vi.fn().mockResolvedValue({}),
  },
}))

vi.mock('@/lib/download-utils', () => ({
  downloadJson: vi.fn(),
}))

import MyPage from './page'
// import { generateTool } from '@/lib/tools-controller'  // if needed for assertions
```

### Mock Pattern Rules

1. **Zustand stores** — use selector pattern:
   ```typescript
   useToastStore: (sel: any) => sel({ addToast: mockAddToast }),
   ```

2. **Controllers** — mock the specific methods the page calls:
   ```typescript
   generateTool: vi.fn(),  // for tools pages
   chatController: { stream: vi.fn() },  // for chat pages
   ```

3. **Locale** — always the same:
   ```typescript
   useLocale: () => ({ t: (k: string) => k }),
   ```

4. **Navigation** — always the same:
   ```typescript
   usePathname: () => '/current-path',
   useRouter: () => ({ push: vi.fn() }),
   ```

### Test Structure

```typescript
describe('MyPage', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('renders the page title', () => {
    render(<MyPage />)
    expect(screen.getByText('Expected Title')).toBeDefined()
  })

  it('renders loading state', () => {
    render(<MyPage />)
    expect(document.querySelectorAll('[class*="animate-pulse"]').length).toBeGreaterThan(0)
  })

  it('handles user click', async () => {
    render(<MyPage />)
    fireEvent.click(screen.getByRole('button', { name: /action/i }))
    await waitFor(() => {
      expect(screen.getByText('Result')).toBeDefined()
    })
  })

  it('calls API on submit', async () => {
    const mockFn = vi.mocked(generateTool)
    mockFn.mockResolvedValue(undefined)

    render(<MyPage />)
    fireEvent.change(screen.getByPlaceholderText(/type here/i), { target: { value: 'test' } })
    fireEvent.click(screen.getByRole('button', { name: /submit/i }))

    await waitFor(() => {
      expect(mockFn).toHaveBeenCalledWith('tool-id', expect.objectContaining({ text: 'test' }))
    })
  })
})
```

### Assertion Patterns

| Check | Use |
|-------|-----|
| Element exists | `screen.getByText('X')` or `screen.getByRole('button', { name: /x/i })` |
| Element absent | `expect(screen.queryByText('X')).toBeNull()` |
| Multiple elements | `screen.getAllByText('X').length` |
| Button disabled | `expect(btn).toBeDisabled()` |
| Loading skeletons | `document.querySelectorAll('[class*="animate-pulse"]').length` |
| Toast called | `expect(mockAddToast).toHaveBeenCalledWith('message', 'type')` |
| API called | `expect(mockFn).toHaveBeenCalledWith(...)` |

### Common Mistakes

| Wrong | Right |
|-------|-------|
| `useToastStore: () => ({ addToast: fn })` | `useToastStore: (sel: any) => sel({ addToast: fn })` |
| Testing disabled button click | Test `toBeDisabled()` instead |
| `getByTestId('x')` | Prefer `getByRole`, `getByText` |
| `getByText('X')` when duplicated | `getAllByText('X').length` |

---

## Backend Unit Tests (Python)

### File Location
`packages/core-py/tests/test_<module>.py`

### Run Command
```bash
cd packages/core-py && .venv/bin/python -m pytest tests/test_<module>.py -x -v
```

### Test Template

```python
"""Tests for domains.<area>.<module>."""
import pytest
import tempfile
from pathlib import Path

DATA_TEXT = "The quick brown fox jumps over the lazy dog. " * 50

class TestMyModule:
    def test_basic(self):
        from domains.my_module import my_function
        result = my_function("input")
        assert result == "expected"

    def test_error(self):
        from domains.my_module import my_function
        with pytest.raises(ValueError, match="expected message"):
            my_function(None)

    def test_with_file(self, tmp_path):
        from domains.my_module import process_file
        f = tmp_path / "test.txt"
        f.write_text(DATA_TEXT)
        result = process_file(str(f))
        assert result is not None
```

### Python Rules
- Import inside test methods (lazy imports)
- Use `tmp_path` or `tempfile` for file operations
- Mark slow tests: `@pytest.mark.slow`
- Use `from __future__ import annotations` only in source, not tests
- Test both success and error paths

---

## Playwright Journey Tests

### File Location
`packages/core-py/tests/test_user_journeys.py`

### Run Command
```bash
cd packages/core-py && .venv/bin/python -m pytest tests/test_user_journeys.py -x -v
```

**Requires:** API on `:8000`, web on `:3000`, Playwright chromium installed.

### Adding a Simple Route Test

Add to the `ROUTES` list:

```python
ROUTES = [
    # ... existing ...
    ("/new-page", "new_page"),
]
```

This auto-generates a navigation test verifying the page loads with content > 50 chars.

### Adding an Interactive Journey Test

```python
class TestNewFeatureFlows:
    def test_feature_loads(self, page: Page):
        body = go(page, "/feature")
        ok("feature_loads", len(body) > 50, f"len={len(body)}")
        assert len(body) > 50

    def test_feature_interaction(self, page: Page):
        go(page, "/feature")
        # Find and click a button
        btn = page.get_by_role("button", name="Start").first
        ok("feature_has_button", btn.count() > 0)
        assert btn.count() > 0

        # Click it
        btn.click(timeout=5000)
        time.sleep(1)
        body = page.inner_text("body")
        ok("feature_clicked", len(body) > 50, f"len={len(body)}")
```

### Journey Test Helpers

```python
def go(page: Page, path: str) -> str:
    """Navigate to path, wait for load, return body text."""
    # Handles: load wait, "Connecting..." disappear, SSE settle, error retry
    # Returns: full body text for assertions

def ok(name: str, passed: bool, detail: str = ""):
    """Record a test result. Use for non-fatal assertions."""
    # Don't fail the whole suite on one page — record and continue
```

### Journey Rules
- Always use `go(page, path)` — it handles settling and error recovery
- Use `ok()` for non-fatal checks, `assert` for fatal ones
- Wait 1-2s after navigation for content to load
- If a page fails, log it and continue — never stop on first failure
- Check `page.inner_text("body")` for content, not just status codes

---

## Coverage Improvement

### Measure
```bash
# Python
cd packages/core-py
.venv/bin/python -m pytest tests/test_<module>*.py \
  --cov=domains/<area>/<module> \
  --cov-report=term-missing -q

# Frontend
cd apps/web && npm test -- --coverage
```

### Workflow
1. **Measure** — find uncovered lines
2. **Read source** — understand uncovered code
3. **Write test** — target the specific path
4. **Verify** — test passes + coverage increases
5. **Find bugs** — watch for KeyError, TypeError, AttributeError
6. **Pragma** — only for provably unreachable code (optional imports)

### Bug Patterns to Watch For

| Symptom | Likely Bug | Fix |
|---------|-----------|-----|
| `KeyError` | Missing `.get()` with default | Add fallback |
| `AttributeError` after `hasattr` | Value is `None` | Check for None |
| `TypeError` | Parameter name mismatch | Check signature |
| Dead code after return | Unreachable branch | Pragma or remove |

---

## Verification Checklist

Before marking complete:

- [ ] Frontend: `npm test -- --run <file>` passes
- [ ] Backend: `python -m pytest <file> -x -v` passes
- [ ] No `console.log` or `print` debug statements
- [ ] All external dependencies mocked
- [ ] Tests isolated (no shared state)
- [ ] Both success and error paths tested
- [ ] Journey tests: `python -m pytest tests/test_user_journeys.py -x -v` passes
