# Testing Guide

This document covers the testing infrastructure for sloughGPT across the Next.js frontend and the Python backend. The root-level pytest suite layout is documented separately in `tests/README.md` — this guide references it instead of duplicating it.

## Test Overview

| Stack | Framework | Location | Run Command |
|-------|-----------|----------|-------------|
| Frontend | Vitest + React Testing Library | `apps/web/` | `cd apps/web && npm test` |
| Backend core | pytest + pytest-asyncio | `packages/core-py/tests/` | `cd packages/core-py && python -m pytest -q` |
| Root suite | pytest | `tests/` | `python -m pytest tests/ -q` (repo root) |
| Contract suite | pytest | `tests/contract/` | `python -m pytest tests/contract/ -q` (repo root) |

> **Always use the project venv** (`.venv/`). Plain `python`/`pip` outside it violates project rules. Server/contract suites need `PYTHONPATH` (see Backend Testing).

## Frontend Testing

### Framework

- **Vitest** with jsdom environment
- **React Testing Library** for component tests
- **@testing-library/user-event** for user interactions
- **@testing-library/jest-dom** for DOM matchers

### Test File Convention

Tests live alongside source files as `*.test.tsx`:

```
apps/web/app/(app)/consciousness/help/
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

# lib/ only (matches CI: npm run test:lib in ci.yml)
npm run test:lib

# Run with coverage
npm test -- --coverage

# Run in watch mode
npm test -- --watch
```

Other scripts: `test:components`, `test:hooks` (subtree runs), `test:changed` (git-changed only), `lint` (`next lint`), `typecheck` (`tsc --noEmit`).

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

- **pytest** with **pytest-asyncio** (`asyncio_mode = auto` in `pytest.ini`)
- **pytest-cov** installed for coverage
- **pytest-xdist** is installed **in CI only** (`.github/workflows/reusable-ci-core.yml` installs it); the local venv does not have it — do **not** pass `-n` to local pytest runs, they fail with `unrecognized arguments: -n`.

### Configuration

Two `pytest.ini` files apply:

- **Root `pytest.ini`** — `testpaths = packages/core-py/tests tests apps/cli/tests`, `addopts = -v --tb=short -m "not slow" --import-mode=importlib`, markers `unit`, `integration`, `e2e`, `slow`.
- **`packages/core-py/pytest.ini`** — `addopts = -q -m "not slow" --ignore=tests/test_neural_e2e.py`, markers `slow`, `integration`.

The journey suites add the `browser` and `live` markers (queued with the `feat/journey-library` branch, card 067); browser tests are excluded from default runs by the `-m` filter.

### Running Tests

```bash
# Core backend suite (from packages/core-py)
cd packages/core-py
python -m pytest -q                                    # applies -m "not slow"
python -m pytest tests/test_training_infrastructure.py # one file
python -m pytest --cov=domain tests/                   # coverage (pytest-cov)

# Root suite (from repo root)
python -m pytest tests/ -q

# Server + contract suites import the FastAPI app — set PYTHONPATH first
export PYTHONPATH=".:packages/core-py:apps/api/server"
python -m pytest tests/server -q
python -m pytest tests/contract/ -q                    # contract suite: 42 tests
```

### Test File Convention

Core tests live in `packages/core-py/tests/`:

```
packages/core-py/tests/
├── test_training_infrastructure.py
├── test_training_export_presets.py
├── test_agent_core.py
└── ...
```

Root-level suites (`tests/`, `tests/server/`, `tests/contract/`, journey/browser suites) are described in `tests/README.md`.

## Test Coverage

### Measuring Coverage

No self-reported percentages are kept in this document — generate them:

```bash
# Frontend
cd apps/web && npm test -- --coverage

# Backend core (venv active)
python -m pytest --cov=domain packages/core-py/tests
```

CI enforces a floor on the core subset: `--cov-fail-under=30` in `.github/workflows/reusable-ci-core.yml`.

## CI/CD Integration

Tests run automatically on:

- `.github/workflows/ci.yml` — frontend job: `npm ci` + `npm run test:lib` in `apps/web` (i.e. `vitest run lib/`).
- `.github/workflows/ci_cd.yml` — `npx vitest run` for the web app (twice: unit + another stage), `sdk-test-py` runs `tests/test_sdk.py`, `standards-schemas` validates example manifests.
- `.github/workflows/reusable-ci-core.yml` — shared lint + core pytest subset; installs `pytest pytest-cov pytest-asyncio pytest-xdist` before running.

### Pre-commit Hooks

Real hooks are configured (not just manual commands):

- `.pre-commit-config.yaml` — pre-commit framework: trailing-whitespace, end-of-file-fixer, check-yaml, check-added-large-files (500KB), check-merge-conflict, `no-commit-to-branch main`, TypeScript check (`tsc --noEmit` in `apps/web`), Python syntax check (`py_compile`).
- `.husky/pre-commit` — runs `npx lint-staged`; the root `lint-staged` config runs `eslint --fix` + `prettier --write` on TS/TSX, `.venv/bin/ruff check --fix` + `ruff format` on Python files, `prettier --write` on JSON/MD.
- `.husky/pre-push` — advisory warning against pushing straight to `main`.
- Manual gate before committing: `npm run lint && npm run typecheck` (root scripts).

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
- [ ] Backend contract changes extend `tests/contract/` (and this guide stays truthful — it is guarded by `tests/contract/test_testing_doc.py`)

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
