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

A third class, **suspects**, is advisory and never sets the exit code. Condition
(2) above is a _direct_ import, so it under-reports: a test may import router A
while router B — one hop away — is the module that reads the un-patched path.
`--mock-drift` therefore follows the test's imports through a static
module → module import graph out to **two hops**:

```text
  Suspects  reader reachable within 2 imports — verify, not verdicts
  ✖ domain.infrastructure._internal.model_registry.get_model_registry
      read via domain.infrastructure.model_registry
      controllers.models
      tests/test_training_distill.py
```

These are leads, not verdicts: reaching a module by import is weaker than using
it at runtime, so the exit code stays `0` while confirmed and unresolvable still
exit `1`. The point is that without this hop the latent count reads as "all
clear" while a second router is quietly on live state.

#### Verifying a suspect

Reach cannot tell you whether the reader is actually _used_, so make it say so:
replace the reader's binding with a sentinel that raises, and rerun the file.

```python
# /tmp/md_probe_plugin.py
# PYTHONPATH="$PYTHONPATH:/tmp" MD_PROBE_SPEC='[{"mod":"domain.knowledge",
#   "attr":"get_knowledge_ingestor","reader_files":["domain.knowledge.engine"]}]' \
#   python -m pytest tests/test_learner_pipeline.py -q -p md_probe_plugin
import importlib
import json
import os
from unittest.mock import patch


def _boom(mod, attr):
    def f(*_a, **_k):
        raise AssertionError(f"PROBE-CALLED {mod}.{attr}")

    return f


def pytest_configure(config):
    for it in json.loads(os.environ["MD_PROBE_SPEC"]):
        for name in {it["mod"], *it.get("reader_files", [])}:
            try:
                m = importlib.import_module(name)
            except Exception:
                continue
            if hasattr(m, it["attr"]):
                patch.object(m, it["attr"], _boom(name, it["attr"])).start()
```

Two details matter. Patch **both** `reader_mod` and the reader file module: a
function-level `from shim import x` re-reads the shim at call time, while an
already-bound module-level read only moves via its own attribute. And install in
`pytest_configure`, before collection binds the name.

- Tests still pass → the reader is never exercised on that path: **harmless**,
  a trap for whoever later extends the test down it, not a defect today.
- Tests fail with `PROBE-CALLED` → the reader **is** used, so the `_internal`
  patch never fed it and the test has been running on live state: **retarget**.

The 9 suspects found on 2026-10-05 were triaged this way and all 9 came back
harmless — which is why the class does not set the exit code.

## The `slow` blind spot

Both `pytest.ini` files carry `addopts = … -m "not slow" …`, so **every default
run — including CI and every bare `python -m pytest` — deselects the `slow`-marked
files entirely.** As of 2026-10-07 that is **24 files / 715 tests** — and the split
matters: **693 sit behind a module-level `pytestmark = pytest.mark.slow`,** so
those files collect _zero_ tests under a default run, while only **22 are
individual `@pytest.mark.slow` decorators across 7 files.** Anything wrong in
them is invisible to every regression gate; `test_chat_loop_e2e.py` hid 6 ERRORs
that way, and the march below cleared 38 more failures.

Run them explicitly:

```bash
# one file
python -m pytest tests/test_e2e_smoke.py -m "slow or not slow" -q

# everything a default run skips
python -m pytest -m "slow" -q
```

Three traps when you do:

- **Do not pass `--timeout=N`.** Neither `pytest.ini` declares a timeout, so the
  flag _creates_ failures: `test_continual_learner.py` "fails" at `--timeout=120`
  and passes at 289 s with no flag, because `train_now()` really trains — which is
  why those tests are marked `slow` in the first place.
- **`pytest.importorskip(...)` exits `5`** (zero items collected) when the
  capability is absent. `tests/test_optimized_pipeline.py` does this for `torch`,
  so it is both invisible _and_ non-zero. The gate itself is correct.
- **Treat exit `5` as _not tested_, never as passed.** A per-file runner that
  logs "processed" rather than the _outcome_ reports success over an empty set:
  gate 1 counted **11 files as processed** while every one recorded
  `NO-SUMMARY exit=5`. A `skipped` count also understates its own blast radius —
  `test_export.py` gated on `gguf` at module level _mid-file_ and reported
  **`1 skipped`** while actually discarding **32 tests**, none of which needed
  `gguf`. When a file comes back `exit=5`, open it before believing it.

Walking these files in 2026-10-07 sorted every failure into seven classes —
worth knowing by name, because each one is invisible to a different check:

| class                          | what broke                                                                        | caught by                            |
| ------------------------------ | --------------------------------------------------------------------------------- | ------------------------------------ |
| **envelope drift**             | reading top level where `success_response()` wrapped it                           | run the file                         |
| **contract drift**             | an assertion predating schema validation                                          | run the file                         |
| **dead routes**                | production route deleted, callers still live (cards `e205b15c`, `db9c4e70`)       | run the file                         |
| **test-harness defect**        | fixture app missing exception handlers; unwired `phase` param                     | run the file                         |
| **config move + rename**       | params folded into `ModelConfig`/`feedback_dir`; 2 also renamed                   | run the file                         |
| **dead seam**                  | patch installs on the _correct_ symbol, but production stopped routing through it | **neither run nor binding probe**    |
| **over-broad capability gate** | module-level `importorskip` mid-file discards unrelated tests                     | count what ran, not what was skipped |

Only the first four were reachable by static analysis; the last three are why
**fail-first beats reason-first** here. In particular the _dead seam_ defeats
both tools at once — the binding probe sees a sound patch, and a boom-probe sees
the same failure either way — so `--mock-drift`'s `exit=0` is a statement about
bindings, not about tests passing.

## Flake Ledger

Six tests flaked in full-suite runs around 2026-10-06 and the per-test analysis
was lost to a server reboot. The record now lives here — **append to it instead
of re-deriving from memory** (cards `297573d9`, `b015795b`).

| Flake | Root cause | Fix |
| ----- | ---------- | --- |
| `vm/page`, `TokenTreeMergesCard` | racy post-`waitFor` assertion | `f964dfe53` |
| `TokenizerPage` (found later, 4/10 isolated) | two "Prev" pagers on one page; the unmocked `TokenTreeVocabCard` reaches the live backend → network race | `f964dfe53` |
| `TokenTreePage` | racy post-`waitFor` assertion — the detector missed it: `getAllBy*` was never matched | `f964dfe53` |
| `ConsciousnessComparePage > shows qualia dimensions` | same blind spot; it had been **mis-triaged** as "a different root cause" | `f964dfe53` |
| `useChatModelSettings` (`0.8 != 0.5`) | async `initStore()` hydration lands *after* the `beforeEach` reset; the hook's store-sync effect applies the backend's `0.8` over the value the test just fetched | `19e922561` |
| `step-components` | **nothing to race**: a call-through fetch probe records 0 requests, the test never uses `fireEvent`, and none of the components it renders (`DataStep`/`TrainStep`/`ResultsStep`) has an effect. 25/25 in every surviving log; the failure text is gone | — |

### Why one test can see another run's value

Three facts, each verified rather than assumed:

1. **`chatDB` is HTTP, not local storage.** `lib/db.ts` implements
   `getKV`/`setKV` with `apiGet`/`apiPut` against `/docstore/kv/*`, so every
   test file shares *one* backend KV — the live one (`localhost:8000` was up).
   Probe: file A does `await chatDB.setKV('app-settings', { defaultTemp: 999 })`;
   file B reads `999`.
2. **`initStore()` self-starts** at `lib/store.ts:168` on module import and
   `setState`s whenever that read resolves — which can be *after* a test's
   `beforeEach` reset. Deterministic repro (throwaway file, KV restored
   afterwards): write `0.8` → `_resetInitGuard()` → reset the store →
   `await initStore()` → the store reads `0.8`; gate the other way
   (`await initStore()` **then** reset) → `0.7` and stays there.
3. **The store writes back.** `updateSettings` → `_pushToBackend` →
   `settingsController.updateGeneration` PATCHes `/settings/generation`.
   Before the fix that file issued 6 such calls per run — mutating whatever
   backend happened to be running.

**Fix pattern** — all three channels, test-only, no product code touched:

```ts
beforeEach(async () => {
  await initStore() // hydration finishes first → the reset below is final
  vi.clearAllMocks()
  useAppStore.setState({ settings: { ...DEFAULT_SETTINGS } })
})
vi.mock('@/lib/settings-controller', () => ({ settingsController: {/* … */} }))
afterEach(() => expect(fetchUrls, `not hermetic: ${fetchUrls}`).toEqual([]))
```

The guard must be a hard assertion, not a log: after the fix the window records
0 calls, while the file's only remaining requests are `initStore`'s two
hydration GETs — issued at import, before any spy exists, read-only, and now
awaited by `beforeEach`, so they can no longer race anything.

Still open: any test using the real store can poison the shared KV for every
file that runs after it, so the *reader-side* fix above does not close the class
— see card `a8d41abd` for the 4 files still writing unmocked.

### When a run fails wholesale

A run with many **file-level** failures and **0 failing tests** is an
environment fault, not a code fault: on 2026-10-06 ENOSPC at 97% disk failed 189
files at collection while 5530/5530 tests still passed, and that exhaustion also
killed shell output so it looked like a tool outage. Check `df` before triaging
assertions (card `dddd954a`).

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
