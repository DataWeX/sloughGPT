---
name: test-driven-development
description: Use when implementing any feature or bugfix in sloughGPT, before writing implementation code. Covers both Python (pytest) and TypeScript (vitest + React Testing Library) stacks.
---

# Test-Driven Development — sloughGPT

## Overview

Write the test first. Watch it fail. Write minimal code to pass.

**Core principle:** If you didn't watch the test fail, you don't know if it tests the right thing.

## Stack

| Layer | Tool | Command | Speed |
|-------|------|---------|-------|
| **Frontend type check** | tsc | `npx tsc --noEmit` | 5-10s |
| **Frontend unit (lib/)** | vitest (node) | `npm run test:lib` | 10-20s |
| **Frontend components** | vitest (jsdom) | `npm run test:components` | 40-60s |
| **Frontend hooks** | vitest (jsdom) | `npm run test:hooks` | 15-30s |
| **Frontend changed** | vitest | `npm run test:changed` | 20-40s |
| **Frontend full** | vitest | `npm run test` | 150-200s |
| **Python syntax** | py_compile | `python3 -m py_compile <file>` | instant |
| **Python unit** | pytest | `make test-py ARGS="tests/test_file.py -x -q"` | varies |
| **Python full** | pytest | `make test-py` (parallel) | 2-3 min |

**Flow:** `tsc` → `test:changed` → commit → `test` before push.

## The Iron Law

```
NO PRODUCTION CODE WITHOUT A FAILING TEST FIRST
```

Write code before the test? Delete it. Start over. No exceptions.

## Red-Green-Refactor

### RED — Write Failing Test

Write one minimal test showing what should happen.

**TypeScript (vitest):**
```typescript
describe('retryOperation', () => {
  it('retries failed operations 3 times', async () => {
    let attempts = 0;
    const operation = () => {
      attempts++;
      if (attempts < 3) throw new Error('fail');
      return 'success';
    };
    const result = await retryOperation(operation);
    expect(result).toBe('success');
    expect(attempts).toBe(3);
  });
});
```

**Python (pytest):**
```python
def test_retries_failed_operations_3_times():
    attempts = 0
    def operation():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ValueError("fail")
        return "success"
    result = retry_operation(operation)
    assert result == "success"
    assert attempts == 3
```

### Verify RED — Watch It Fail

**MANDATORY. Never skip.**

```bash
# TypeScript
npx vitest run path/to/test.test.ts

# Python
python3 -m pytest tests/test_file.py -x -q
```

Confirm: test fails (not errors), failure message is expected, fails because feature missing (not typos).

### GREEN — Minimal Code

Write simplest code to pass. Don't add features, don't refactor, don't "improve" beyond the test.

### Verify GREEN — Watch It Pass

```bash
npx vitest run path/to/test.test.ts  # TypeScript
python3 -m pytest tests/test_file.py -x -q  # Python
```

### REFACTOR — Clean Up

After green only: remove duplication, improve names, extract helpers. Keep tests green.

## Frontend Test Patterns

### Component Tests

```typescript
// @vitest-environment jsdom
import { render, screen, fireEvent } from '@testing-library/react';
import { MyComponent } from './MyComponent';

describe('MyComponent', () => {
  it('renders with default state', () => {
    render(<MyComponent />);
    expect(screen.getByText('Expected text')).toBeInTheDocument();
  });

  it('calls onClick when button clicked', () => {
    const onClick = vi.fn();
    render(<MyComponent onClick={onClick} />);
    fireEvent.click(screen.getByRole('button', { name: /click me/i }));
    expect(onClick).toHaveBeenCalledOnce();
  });
});
```

### Hook Tests

```typescript
// @vitest-environment jsdom
import { renderHook, act } from '@testing-library/react';
import { useMyHook } from './useMyHook';

describe('useMyHook', () => {
  it('returns initial state', () => {
    const { result } = renderHook(() => useMyHook());
    expect(result.current.value).toBe(0);
  });

  it('increments on call', () => {
    const { result } = renderHook(() => useMyHook());
    act(() => result.current.increment());
    expect(result.current.value).toBe(1);
  });
});
```

### Controller/Lib Tests (node environment)

```typescript
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { myFunction } from './my-module';

describe('myFunction', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('returns expected result', async () => {
    const result = await myFunction('input');
    expect(result).toEqual({ status: 'ok' });
  });
});
```

### Mocking Patterns

```typescript
// Mock a module
vi.mock('@/lib/model-controller', () => ({
  modelController: {
    list: vi.fn().mockResolvedValue([]),
    load: vi.fn().mockResolvedValue(undefined),
  },
}));

// Mock fetch globally
vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
  ok: true,
  json: () => Promise.resolve({}),
}));

// Use vi.hoisted for mock declarations at module level
const { mockFn } = vi.hoisted(() => ({
  mockFn: vi.fn(),
}));
vi.mock('./module', () => ({ fn: mockFn }));
```

### StrictMode Double-Mount

React 18 StrictMode double-mounts in jsdom. Use `getAllBy*` for potentially duplicated elements:

```typescript
// Instead of:
expect(screen.getByText('Ready')).toBeInTheDocument();

// Use:
const elements = screen.getAllByText('Ready');
expect(elements.length).toBeGreaterThanOrEqual(1);
```

## Python Test Patterns

### Unit Tests

```python
import pytest
from domains.training.trainer_protocol import TrainResult

def test_train_result_backward_compat():
    result = TrainResult(success=True, final_loss=0.5)
    assert result["checkpoint"] == result.model_path  # dict-like access
    assert result.success is True
```

### Fixture Patterns

```python
@pytest.fixture
def mock_model_server():
    server = ModelServer(model=None, tokenizer=None)
    server._generate_sync = lambda *a, **kw: "mocked response"
    return server

def test_generate_returns_response(mock_model_server):
    result = mock_model_server.generate("test")
    assert result == "mocked response"
```

### Async Tests

```python
import pytest

@pytest.mark.asyncio
async def test_async_generate():
    result = await async_generate("hello")
    assert len(result) > 0
```

## Common Rationalizations

| Excuse | Reality |
|--------|---------|
| "Too simple to test" | Simple code breaks. Test takes 30 seconds. |
| "I'll test after" | Tests written after pass immediately — proves nothing. |
| "Need to explore first" | Fine. Throw away exploration, start with TDD. |
| "Test hard = design unclear" | Listen to test. Hard to test = hard to use. |
| "TDD will slow me down" | Catches bugs before commit, prevents regressions. |
| "Already manually tested" | Manual testing is ad-hoc, not reproducible. |

## Red Flags — STOP and Start Over

- Code before test
- Test after implementation
- Test passes immediately
- Can't explain why test failed
- Rationalizing "just this once"

**All mean: Delete code. Start over with TDD.**

## Verification Checklist

Before marking work complete:
- [ ] Every new function/method has a test
- [ ] Watched each test fail before implementing
- [ ] Each test failed for expected reason
- [ ] Wrote minimal code to pass
- [ ] `npx tsc --noEmit` passes (frontend)
- [ ] `npx vitest run` passes (frontend)
- [ ] `python3 -m py_compile <file>` passes (Python)
- [ ] `make test-py` passes (Python)
