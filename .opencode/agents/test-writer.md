---
description: >
  Unified testing agent for sloughGPT. Writes unit tests, component tests,
  Playwright journey tests, and improves coverage. Use when the user says
  "test", "add tests", "write tests", "coverage", "journey tests", or
  needs testing work done.
mode: subagent
---

# Test Writer Agent

You write tests for sloughGPT. This file contains every pattern, mock, and
convention used in the codebase. Do not search for examples — they are here.

## Decision Tree

```
What are you testing?
├── .tsx page/component → Frontend Component Test (vitest + jsdom)
├── .py module → Backend Unit Test (pytest)
├── Page flow / "journey" → Playwright Journey Test
└── "coverage" → Measure → Write → Verify
```

---

## 1. Frontend Component Tests

### Run
```bash
cd apps/web && npm test -- --run <relative-path-to-test>
```

### File Location
Next to source: `app/(app)/my-page/page.test.tsx`

### Environment
Vitest auto-selects `jsdom` for files in `app/`, `components/`, `hooks/`, `features/` via `vitest.config.ts` globs. No `@vitest-environment` comment needed.

### How Mocking Works in This Codebase

Mocks MUST be declared BEFORE imports. The order is:

```
1. vi.mock('next/navigation', ...)     ← always
2. vi.mock('@/hooks/useLocale', ...)   ← always
3. vi.mock('@/lib/toast-store', ...)   ← if page uses toasts
4. vi.mock('@/lib/X-controller', ...)  ← whatever the page imports
5. import Component from './page'      ← AFTER mocks
```

This is not optional. vitest hoists `vi.mock()` calls, but the import
must come after the mock declaration for the mock to be in place.

### Pattern A: Page That Calls an API Controller

Used for: tools pages (brainstorm, decide, explain, rewrite, translate, wellness, writing), training pages, knowledge pages.

```tsx
// apps/web/app/(app)/brainstorm/page.test.tsx
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import React from 'react'

const mockAddToast = vi.fn()

vi.mock('next/navigation', () => ({
  usePathname: () => '/brainstorm',
  useRouter: () => ({ push: vi.fn() }),
}))

vi.mock('@/hooks/useLocale', () => ({
  useLocale: () => ({ t: (k: string) => k }),
  LOCALES: [],
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (sel: any) => sel({ addToast: mockAddToast }),
}))

vi.mock('@/lib/tools-controller', () => ({
  generateTool: vi.fn(),
}))

import BrainstormPage from './page'
import { generateTool } from '@/lib/tools-controller'

describe('BrainstormPage', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('renders the page title', () => {
    render(<BrainstormPage />)
    expect(screen.getByText('Brainstorm')).toBeDefined()
  })

  it('calls generateTool when sending', async () => {
    vi.mocked(generateTool).mockResolvedValue(undefined)
    render(<BrainstormPage />)

    fireEvent.change(screen.getByPlaceholderText(/What/i), { target: { value: 'Test idea' } })
    fireEvent.click(screen.getByRole('button', { name: /send/i }))

    await waitFor(() => {
      expect(generateTool).toHaveBeenCalledWith(
        'brainstorm',
        expect.objectContaining({ history: expect.any(Array) }),
        expect.any(Object),
        expect.any(Object)
      )
    })
  })
})
```

**Key details:**
- `useToastStore: (sel: any) => sel({ addToast: mockAddToast })` — the `(sel: any) => sel({...})` pattern is required because Zustand stores use selector functions
- `vi.mocked(generateTool).mockResolvedValue(undefined)` — cast the mock for TypeScript
- `expect.objectContaining({...})` — match partial payload, not exact shape
- `expect.any(Object)` — for callbacks and options objects

### Pattern B: Page With settingsController (Data Fetching)

Used for: training/analytics, training/presets, training/runs, settings, monitoring.

```tsx
vi.mock('@/lib/settings-controller', () => ({
  settingsController: {
    get: vi.fn().mockResolvedValue({ training_analytics: { total_runs: 5 } }),
    listTrainingPresets: vi.fn().mockResolvedValue({ presets: [] }),
    applyTrainingPreset: vi.fn().mockResolvedValue({}),
  },
}))

vi.mock('@/lib/download-utils', () => ({
  downloadJson: vi.fn(),
}))
```

**Test pattern for data-fetching pages:**

```tsx
it('renders after loading', async () => {
  render(<MyPage />)
  await waitFor(() => {
    expect(screen.getAllByText(/Analytics/i).length).toBeGreaterThan(0)
  }, { timeout: 5000 })
})
```

Use `getAllByText(...).length` instead of `getByText` because StrictMode
double-renders. Use `timeout: 5000` because async data fetching may take time.

### Pattern C: Static Content Page (No API)

Used for: shortcuts, phoneme (with mocked children).

```tsx
vi.mock('next/navigation', () => ({
  usePathname: () => '/shortcuts',
  useRouter: () => ({ push: vi.fn() }),
}))

vi.mock('@/hooks/useLocale', () => ({
  useLocale: () => ({ t: (k: string) => k }),
  LOCALES: [],
}))

import ShortcutsPage from './page'

describe('ShortcutsPage', () => {
  it('renders the page title', () => {
    render(<ShortcutsPage />)
    expect(screen.getAllByText(/Keyboard Shortcuts/i).length).toBeGreaterThan(0)
  })

  it('shows content', () => {
    render(<ShortcutsPage />)
    expect(screen.getByText('Command palette')).toBeDefined()
  })
})
```

### Pattern D: Page With Many Child Components

Used for: phoneme, settings, any page that imports 10+ cards.

Mock each child to return a testable placeholder:

```tsx
vi.mock('@/components/phoneme/EncodingCard', () => ({
  default: () => <div data-testid="encoding-card">EncodingCard</div>,
}))

vi.mock('@/components/phoneme/ScoringCard', () => ({
  default: () => <div data-testid="scoring-card">ScoringCard</div>,
}))
```

Then assert the child rendered:

```tsx
it('shows encoding tab', () => {
  render(<PhonemePage />)
  expect(screen.getByTestId('encoding-card')).toBeDefined()
})
```

### Pattern E: Complex Component With Controller (MemoryTab)

Used for: panels, complex components with multiple API calls.

```tsx
// Use vi.hoisted to create mock functions accessible in vi.mock factories
const hoisted = vi.hoisted(() => ({
  memoryController: {
    list: vi.fn(),
    stats: vi.fn(),
    delete: vi.fn(),
    clear: vi.fn(),
  },
}))

vi.mock('@/lib/memory-controller', () => ({
  memoryController: hoisted.memoryController,
}))

// Set up return values in beforeEach
beforeEach(() => {
  vi.clearAllMocks()
  hoisted.memoryController.list.mockResolvedValue({ items: [item()], total: 1 })
  hoisted.memoryController.stats.mockResolvedValue({ enabled: true, total_facts: 1 })
})

// Test async rendering with findByText
it('renders after load', async () => {
  render(<MemoryTab />)
  expect(await screen.findByText('1 fact')).toBeDefined()
})
```

**Key: `screen.findByText` (async) vs `screen.getByText` (sync)**
- Use `findByText` when data loads asynchronously (API calls, useEffect)
- Use `getByText` when content is static (rendered immediately)

### Assertion Cheat Sheet

| What to check | Code |
|--------------|------|
| Element exists | `screen.getByText('X')` |
| Element absent | `expect(screen.queryByText('X')).toBeNull()` |
| Multiple elements | `screen.getAllByText('X').length` |
| Button disabled | `expect(btn).toBeDisabled()` |
| Button enabled | `expect(btn).not.toBeDisabled()` |
| Input value | `expect(input).toHaveValue('text')` |
| Placeholder | `screen.getByPlaceholderText(/pattern/i)` |
| Role | `screen.getByRole('button', { name: /x/i })` |
| Label | `screen.getByLabelText('X')` |
| API called | `expect(mockFn).toHaveBeenCalledWith(...)` |
| API not called | `expect(mockFn).not.toHaveBeenCalled()` |
| Toast shown | `expect(mockAddToast).toHaveBeenCalledWith('msg', 'error')` |
| Loading state | `document.querySelectorAll('[class*="animate-pulse"]').length` |
| CSS class | `element.className.toContain('primary')` |
| Aria attribute | `expect(el.getAttribute('aria-checked')).toBe('true')` |

### Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `useToastStore.mockReturnValue is not a function` | Wrong mock pattern | Use `(sel: any) => sel({...})` |
| `Found multiple elements` | StrictMode double-render | Use `getAllByText(...).length` |
| `Unable to find role="button"` | Button text wrong | Check actual button label in source |
| `Expected mock to have been called` | Missing `await waitFor` | Wrap assertion in `waitFor(() => ...)` |
| `ReferenceError: ... is not defined` | Import before mock | Move `vi.mock()` above `import` |

---

## 2. Backend Unit Tests (Python)

### Run
```bash
cd packages/core-py && .venv/bin/python -m pytest tests/test_<module>.py -x -v
```

### File Location
`packages/core-py/tests/test_<module>.py`

### Pattern

```python
"""Tests for domains.<area>.<module>."""
import pytest
import tempfile
from pathlib import Path

DATA_TEXT = "The quick brown fox jumps over the lazy dog. " * 50

FAST_CONFIG = {
    "method": "sft",
    "epochs": 1,
    "batch_size": 8,
    "block_size": 32,
    "max_steps": 2,
    "n_embed": 32,
    "n_layer": 2,
    "n_head": 2,
}

class TestMyModule:
    def test_basic(self):
        from domains.my_module import my_function
        result = my_function("input")
        assert result == "expected"

    def test_error(self):
        from domains.my_module import my_function
        with pytest.raises(ValueError, match="not found"):
            my_function(None)

    def test_with_temp_file(self, tmp_path):
        from domains.my_module import process_file
        f = tmp_path / "test.txt"
        f.write_text(DATA_TEXT)
        result = process_file(str(f))
        assert result is not None
```

**Rules:**
- Import inside test methods (lazy imports)
- Use `tmp_path` for file I/O
- Use `DATA_TEXT` constant for test data
- Use `FAST_CONFIG` for training config
- Mark slow tests: `@pytest.mark.slow`

---

## 3. Playwright Journey Tests

### Run
```bash
cd packages/core-py && .venv/bin/python -m pytest tests/test_user_journeys.py -x -v
```

**Requires:** API on `:8000`, web on `:3000`.

### Adding a Route Test

Add to `ROUTES` list in `test_user_journeys.py`:

```python
ROUTES = [
    # ... existing ...
    ("/brainstorm", "brainstorm"),
    ("/consciousness/dashboard", "consciousness_dashboard"),
]
```

Auto-generates: navigate → wait → check `len(body) > 50`.

### Adding an Interactive Journey Test

```python
class TestToolsFlows:
    def test_brainstorm_has_input(self, page: Page):
        body = go(page, "/brainstorm")
        has_input = page.locator("textarea:visible").count() > 0
        has_suggestions = "Name ideas" in body
        ok("brainstorm_input", has_input and has_suggestions)
        assert has_input and has_suggestions

    def test_decide_has_two_options(self, page: Page):
        body = go(page, "/decide")
        inputs = page.locator("input:visible")
        ok("decide_options", inputs.count() >= 2)
        assert inputs.count() >= 2
```

**Helpers:**
- `go(page, path)` — navigate, wait for load, settle SSE streams, retry on error boundary, return body text
- `ok(name, passed, detail)` — record non-fatal test result (won't stop suite)

**Rules:**
- Use `go()` not `page.goto()` — it handles settling
- Use `ok()` for non-fatal checks, `assert` for fatal
- If a page fails, log and continue — never stop on first failure
- Check `page.inner_text("body")` for content presence
- Use `page.locator("element:visible")` not `page.locator("element")` to avoid invisible elements

---

## 4. Coverage Workflow

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

### Process
1. Read source → find uncovered lines
2. Write test targeting that specific path
3. Verify test passes + coverage increases
4. Watch for bugs: KeyError → missing `.get()`, TypeError → param mismatch
5. Pragma only genuinely unreachable code (optional imports)

---

## 5. Adding to ROUTES

The `ROUTES` list in `test_user_journeys.py` accepts tuples of `(path, name)`:

```python
("/page-path", "page_name"),
```

All routes in the list get a parametrized navigation test that:
1. Navigates to `http://localhost:3000{path}`
2. Waits for load + "Connecting..." to disappear
3. Asserts `len(body) > 50`
4. Records pass/fail via `ok()`
