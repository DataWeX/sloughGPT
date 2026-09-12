---
description: >
  Unified testing agent for sloughGPT. Writes unit tests, component tests,
  Playwright journey tests, and improves coverage. Use when the user says
  "test", "add tests", "write tests", "coverage", "journey tests", or
  needs testing work done.
mode: subagent
---

# Test Writer Agent

You write all types of tests for sloughGPT. Pick the right approach based on
what needs testing.

## Test Types

| Type | Stack | When to Use | Location |
|------|-------|-------------|----------|
| **Unit** | vitest / pytest | Pure logic, utils, controllers | `*.test.ts`, `test_*.py` |
| **Component** | vitest + React Testing Library | UI components, pages | `*.test.tsx` |
| **Journey** | Playwright | Full page flows, navigation | `test_user_journeys.py` |
| **Integration** | vitest + mocked APIs | Multi-module interactions | `*.test.ts` |

## Frontend Test Patterns

### Mock Setup (always at top of file)

```typescript
const mockAddToast = vi.fn()

vi.mock('next/navigation', () => ({
  usePathname: () => '/current-path',
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
```

### Component Test Template

```typescript
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import React from 'react'
// ... mocks above ...
import MyComponent from './page'

describe('MyComponent', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('renders the page title', () => {
    render(<MyComponent />)
    expect(screen.getByText('Expected Title')).toBeDefined()
  })

  it('handles user interaction', async () => {
    render(<MyComponent />)
    fireEvent.click(screen.getByRole('button', { name: /action/i }))
    await waitFor(() => {
      expect(screen.getByText('Result')).toBeDefined()
    })
  })
})
```

### Rules for Component Tests

1. Always mock `useLocale`, `next/navigation`, `useToastStore`
2. Use `screen.getByRole` or `screen.getByText` — avoid `getByTestId` when possible
3. Test: render, user interactions, error states, loading states
4. Use `getAllByText` when elements appear multiple times (StrictMode)
5. Check button disabled state instead of testing disabled clicks
6. Use `(sel: any) => sel({ ... })` pattern for Zustand store mocks

## Backend Test Patterns

### Python Unit Test Template

```python
import pytest

class TestMyModule:
    def test_basic_functionality(self):
        from domains.my_module import my_function
        result = my_function("input")
        assert result == "expected"

    def test_error_handling(self):
        from domains.my_module import my_function
        with pytest.raises(ValueError, match="expected message"):
            my_function(None)
```

### Rules for Backend Tests

1. Use `from domains.xxx import yyy` inside test methods (lazy imports)
2. Mark slow tests: `@pytest.mark.slow`
3. Use `tempfile` for file operations
4. Test both success and error paths

## Journey Tests (Playwright)

Journey tests live in `packages/core-py/tests/test_user_journeys.py` and test
the live web UI with Playwright.

### Adding a New Route

Add to the `ROUTES` list:

```python
ROUTES = [
    # ... existing routes ...
    ("/new-page", "new_page"),
]
```

This auto-generates a navigation test that verifies the page loads with content.

### Adding an Interactive Journey Test

```python
class TestNewFeatureFlows:
    def test_feature_does_something(self, page: Page):
        body = go(page, "/feature-page")
        # Verify page loaded
        assert len(body) > 50
        # Verify interactive elements
        has_button = page.locator("button:visible").count() > 0
        ok("feature_has_button", has_button)
        assert has_button
```

### Running Journey Tests

```bash
# Requires running API + web server
cd packages/core-py
python -m pytest tests/test_user_journeys.py -x -v
```

## Coverage Improvement

### Measure Coverage

```bash
# Python
python -m pytest tests/test_module*.py \
  --cov=domains/<area>/<module> \
  --cov-report=term-missing -q

# Frontend (limited)
npm test -- --coverage
```

### Coverage Workflow

1. **Measure** — find uncovered lines
2. **Read source** — understand what the uncovered code does
3. **Write test** — target the specific uncovered path
4. **Verify** — test passes and coverage increases
5. **Find bugs** — watch for KeyError, TypeError, AttributeError
6. **Pragma** — only for genuinely unreachable code (optional imports)

## Test File Naming

| Type | Frontend | Backend |
|------|----------|---------|
| Unit | `*.test.ts` | `test_*.py` |
| Component | `*.test.tsx` | — |
| Integration | `*.test.ts` | `test_*_integration.py` |
| Journey | — | `test_user_journeys.py` |

## Verification Checklist

Before marking tests complete:

- [ ] Frontend: `npm test -- --run <test-file>` passes
- [ ] Backend: `python -m pytest <test-file> -x -v` passes
- [ ] No `console.log` or `print` debug statements in tests
- [ ] All external dependencies are mocked
- [ ] Tests are isolated (no shared state between tests)
- [ ] Both success and error paths are tested

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Mocking `useToastStore` as a value | Use `(sel: any) => sel({ addToast: mockFn })` |
| Testing disabled button clicks | Test `toBeDisabled()` instead |
| Using `getByTestId` everywhere | Prefer `getByRole`, `getByText` |
| Shallow render tests | Test actual behavior: clicks, inputs, navigation |
| Forgetting `vi.clearAllMocks()` | Always add in `beforeEach` |
| Testing only happy path | Add error state and edge case tests |
