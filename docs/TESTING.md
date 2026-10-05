# Testing Guide

This document covers the testing infrastructure for sloughGPT across both the Python backend and Vite frontend.

## Test Overview

| Stack    | Framework                      | Location                  | Run Command                                             |
| -------- | ------------------------------ | ------------------------- | ------------------------------------------------------- |
| Frontend | Vitest + React Testing Library | `apps/web/`               | `cd apps/web && npm test`                               |
| Backend  | pytest                         | `packages/core-py/tests/` | `cd packages/core-py && python -m pytest -n auto -x -q` |

## Frontend Testing

### Framework

- **Vitest** with jsdom environment
- **React Testing Library** for component tests
- **@testing-library/user-event** for user interactions
- **@testing-library/jest-dom** for DOM matchers

### Test File Convention

Tests live alongside source files as `*.test.tsx`:

```
apps/web/app/(app)/brainstorm/
├── page.tsx
└── page.test.tsx
```

### Running Tests

```bash
cd apps/web

# Run all tests
npm test

# Run specific test file
npm test -- consciousness/page.test.tsx

# Run with coverage
npm test -- --coverage

# Run in watch mode
npm test -- --watch
```

### Test Helper

Use `apps/web/lib/__test-helper.ts` for consistent mocking:

```typescript
import { createMockController } from '@/lib/__test-helper'

const consciousnessController = createMockController()
vi.mock('@/lib/consciousness-controller', () => ({
  consciousnessController,
}))
```

### Common Mock Patterns

**LocaleProvider:**

```tsx
import { LocaleProvider } from '@/hooks/useLocale'

render(
  <LocaleProvider>
    <Component />
  </LocaleProvider>,
)
```

**Navigation (next/navigation compat):**

```typescript
vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: vi.fn(), back: vi.fn() }),
  usePathname: () => '/current/path',
  useSearchParams: () => new URLSearchParams(),
}))
```

**Toast Store:**

```typescript
vi.mock('@/lib/toast-store', () => ({
  useToastStore: () => ({ addToast: vi.fn() }),
}))
```

**Strui Components:**

```typescript
vi.mock('@sloughgpt/strui', () => ({
  Button: ({ children, ...props }) => <button {...props}>{children}</button>,
  Card: ({ children }) => <div data-testid="card">{children}</div>,
  // ... other components
}))
```

### Component Test Structure

```tsx
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { LocaleProvider } from '@/hooks/useLocale'
import MyComponent from './page'

const renderWithProviders = (ui: React.ReactElement) => {
  return render(<LocaleProvider>{ui}</LocaleProvider>)
}

describe('MyComponent', () => {
  it('renders correctly', () => {
    renderWithProviders(<MyComponent />)
    expect(screen.getByText('Expected Text')).toBeInTheDocument()
  })

  it('handles user interaction', async () => {
    const user = userEvent.setup()
    renderWithProviders(<MyComponent />)
    await user.click(screen.getByRole('button', { name: /click me/i }))
    expect(screen.getByText('Result')).toBeInTheDocument()
  })
})
```

## Backend Testing

### Framework

- **pytest** with parallel execution (`-n auto`)
- **pytest-asyncio** for async tests

### Running Tests

```bash
cd packages/core-py

# Run all tests (parallel)
python -m pytest -n auto -x -q

# Run specific test file
python -m pytest tests/test_training_infrastructure.py

# Run with coverage
python -m pytest --cov=domains tests/

# Verbose output
python -m pytest -v tests/
```

### Test File Convention

Tests live in `packages/core-py/tests/`:

```
packages/core-py/tests/
├── test_training_infrastructure.py
├── test_training_export_presets.py
├── test_inference.py
└── ...
```

### Patch Target Drift (test-doctor --mock-drift)

A test patches the **definition** site while production reads the **re-export**:

```python
@patch("domain.feedback._internal.per_user_lora.get_per_user_lora")   # test
# but routers/user_adapters.py:54 does:
from domain.feedback import get_per_user_lora                          # reader
```

If `domain/feedback/__init__.py` binds that name **eagerly** (plain
`from ... import ...`, no `__getattr__`), the package attribute is frozen at
import time — the patch never reaches the reader and the test silently runs
against **live state** instead of the fake. A lazy re-export
(`__getattr__` that re-imports per access) does follow the patch.

The same symbol can be read both ways, so **patch where it is read** — the
target depends on the code under test, not on the symbol:

```python
# routers/memory.py imports _internal at call time -> this target is correct
@patch("domain.memory._internal.task_memory.list_archive")
# routers/user_adapters.py imports the package -> this one is required
@patch("domain.feedback.get_per_user_lora")
```

```bash
python scripts/test-doctor.py --mock-drift        # confirmed (exit 1 if any)
python scripts/test-doctor.py --mock-drift -v     # + latent readers
```

Reports a target only when **both** hold: (1) patching it provably fails to
move the reader's attribute, and (2) a test that patches it imports the module
doing the non-patched read. Condition (2) keeps false positives down — a test
whose subject imports `_internal` directly is fine even though the package
binding does not follow.

It also reports **unresolvable** targets, where `patch()` itself raises and the
test errors at setup:

```text
✖ domain.learner._internal.get_learner
      AttributeError: module 'domain.learner._internal' (namespace) … has no
      attribute 'get_learner'
```

These are loud, but easy to miss for exactly that reason: if the file is
`slow`-marked it is deselected and the ERRORs never reach a normal run. Call
sites that pass `create=True` are skipped — that is an intentional mock of a
name which does not exist yet, not a broken target.

## Test Coverage

### Current Status

| Component                | Coverage | Tests             |
| ------------------------ | -------- | ----------------- |
| Consciousness pages      | 100%     | 26 page tests     |
| Consciousness components | 100%     | 5 component tests |
| Consciousness hooks      | 100%     | 6 hook tests      |
| Consciousness lib        | 100%     | 3 lib tests       |
| Navigation               | 100%     | 8 tests           |
| Hooks/Lib                | 95%+     | 1793 tests        |
| Features/Chat            | 95%+     | 1467 tests        |
| Tools pages              | 100%     | 7 page tests      |

### Achieving High Coverage

1. **Test all states:** Render, loading, error, empty, success
2. **Test user interactions:** Clicks, form submissions, navigation
3. **Mock external dependencies:** API calls, navigation, stores
4. **Use shared helpers:** `createMockController()` for consistency

## CI/CD Integration

Tests run automatically on:

- Pull request creation
- Push to main branch
- Manual trigger via GitHub Actions

### Pre-commit Hooks

```bash
# Run lint and typecheck before commit
npm run lint && npm run typecheck
```

## Writing New Tests

### Checklist

- [ ] Test file follows `*.test.tsx` convention
- [ ] Component renders without errors
- [ ] All interactive elements have click handlers tested
- [ ] Loading states are handled
- [ ] Error states are handled
- [ ] Empty states are handled
- [ ] External dependencies are mocked
- [ ] LocaleProvider wraps test component
- [ ] Tests are isolated (no shared state)

### Example Template

```tsx
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { LocaleProvider } from '@/hooks/useLocale'
import { YourComponent } from './page'

// Mock external dependencies
vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: vi.fn() }),
}))

const renderWithProviders = (ui: React.ReactElement) => {
  return render(<LocaleProvider>{ui}</LocaleProvider>)
}

describe('YourComponent', () => {
  it('renders correctly', () => {
    renderWithProviders(<YourComponent />)
    expect(screen.getByText('Expected')).toBeInTheDocument()
  })

  it('handles click', async () => {
    const user = userEvent.setup()
    renderWithProviders(<YourComponent />)
    await user.click(screen.getByRole('button'))
    // Assert behavior
  })
})
```
