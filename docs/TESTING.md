# Testing Guide

This document covers the testing infrastructure for sloughGPT across both the Python backend and Next.js frontend.

## Test Overview

| Stack | Framework | Location | Run Command |
|-------|-----------|----------|-------------|
| Frontend | Vitest + React Testing Library | `apps/web/` | `cd apps/web && npm test` |
| Backend | pytest | `packages/core-py/tests/` | `cd packages/core-py && python -m pytest -n auto -x -q` |

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
  consciousnessController
}))
```

### Common Mock Patterns

**LocaleProvider:**
```tsx
import { LocaleProvider } from '@/hooks/useLocale'

render(
  <LocaleProvider>
    <Component />
  </LocaleProvider>
)
```

**Next.js Navigation:**
```typescript
vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: vi.fn(), back: vi.fn() }),
  usePathname: () => '/current/path',
  useSearchParams: () => new URLSearchParams()
}))
```

**Toast Store:**
```typescript
vi.mock('@/lib/toast-store', () => ({
  useToastStore: () => ({ addToast: vi.fn() })
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
  return render(
    <LocaleProvider>
      {ui}
    </LocaleProvider>
  )
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
python -m pytest --cov=domain tests/

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

## Test Coverage

### Current Status

| Component | Coverage | Tests |
|-----------|----------|-------|
| Consciousness pages | 100% | 26 page tests |
| Consciousness components | 100% | 5 component tests |
| Consciousness hooks | 100% | 6 hook tests |
| Consciousness lib | 100% | 3 lib tests |
| Navigation | 100% | 8 tests |
| Hooks/Lib | 95%+ | 1793 tests |
| Features/Chat | 95%+ | 1467 tests |
| Tools pages | 100% | 7 page tests |

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
  useRouter: () => ({ push: vi.fn() })
}))

const renderWithProviders = (ui: React.ReactElement) => {
  return render(
    <LocaleProvider>
      {ui}
    </LocaleProvider>
  )
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

## Site Doctor (runtime monitor — not `scripts/test-doctor.py`)

`scripts/test-doctor.py` diagnoses the **test suite** (why a pytest run
failed). The **site doctor** diagnoses the **live site**: a read-only
monitor that probes the running stack and triages what it finds. Phase A
is report-only — it never remediates.

- **What it probes:** API health/errors (`/health*`, `/errors/*`),
  `/health/stream` cadence + payload size (flags >256 KB frames and >8s
  stalls), and the UX journey sweep report (`domain/journeys`).
- **Placement:** wiring lives in `domain/core/_internal/doctor/` (the
  "body of the system" — system-level ops); probes are seams onto the
  core components.

```bash
# Full summary (~15s, no browser):                        exit 0/1/2
.venv/bin/python -m domain.core._internal.doctor
# Read existing journey report instead of sweeping:
.venv/bin/python -m domain.core._internal.doctor --no-sweep
# Machine-readable report on stdout; also written to the report path:
.venv/bin/python -m domain.core._internal.doctor --json
# --window N (SSE seconds), --skip sse,http,journey, --report PATH, --strict
```

**Exit codes:** `0` ok/info · `1` warn (with `--strict`: info too) ·
`2` critical (e.g. API unreachable, oversized SSE frames).

**Report path:** `${SLO_DOCTOR_REPORT:-~/.cache/slog-doctor/findings-report.json}`
(JSON, `schema_version: 1`, findings ranked worst-first).

### Surfaced: `/doctor` API + `/doctor` page

| Endpoint | Data | Notes |
|----------|------|-------|
| `GET /doctor/report` | `{report, path, age_s}` | reads `$SLO_DOCTOR_REPORT`; missing/corrupt file → `report: null` + `age_s: null` (empty state, **not** an error); `age_s` = seconds since `report.ts` |
| `POST /doctor/run` | `{report}` | **light run**: `http` + `sse` (6s window) + journey findings from disk, written atomically, then returned; the browser sweep is **never** triggered here (`run_sweep=False`) |

The `/doctor` page (`apps/web/app/(app)/doctor/page.tsx`) fetches the report
on mount and after every run, then shows the overall severity pill + severity
counts + report age, a live component strip (inference / engines-system from
`useLiveStatus`), and the findings grouped worst-first with each detail folded
behind a disclosure.

```bash
# Router + doctor package tests
.venv/bin/python -m pytest tests/server/test_doctor_router.py packages/core-py/tests/test_doctor.py -q
# Page + component tests (worktree: add --config /tmp/opencode/vitest-worktree.config.ts)
node_modules/.bin/vitest run "app/(app)/doctor/page.test.tsx" components/doctor/
```

