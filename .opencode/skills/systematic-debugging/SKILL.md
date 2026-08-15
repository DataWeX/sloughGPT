---
name: systematic-debugging
description: Use when encountering any bug, test failure, or unexpected behavior in sloughGPT, before proposing fixes. Covers the full stack: Next.js frontend, FastAPI backend, SloNet core, and cross-component integration.
---

# Systematic Debugging — sloughGPT

## Overview

**Core principle:** ALWAYS find root cause before attempting fixes. Symptom fixes are failure.

## The Iron Law

```
NO FIXES WITHOUT ROOT CAUSE INVESTIGATION FIRST
```

## Stack Context

| Layer | Tech | Common failure modes |
|-------|------|---------------------|
| **Frontend** | Next.js, React, vitest | JSDOM StrictMode double-mount, `vi.mock` TDZ, portal rendering, async state |
| **API** | FastAPI, Python 3.9 | Event loop blocking, sync-in-async, missing imports, schema validation |
| **Core** | SloNet (NumPy autograd) | Gradient explosion, shape mismatch, Metal accelerator non-determinism |
| **Integration** | SSE streaming, Provider chain | Cancel-on-disconnect, provider priority, KV cache state, session binding |
| **Infra** | ModelServer, CircuitBreaker | MPS OOM, subprocess crash, memory leak, warmup thread binding |

## The Four Phases

### Phase 1: Root Cause Investigation

**BEFORE attempting ANY fix:**

1. **Read Error Messages Completely**
   - Python stack traces: read from bottom up
   - TypeScript errors: read the full `file:line:col` path
   - Frontend console: check both client and server logs
   - SSE errors: check network tab for partial responses

2. **Reproduce Consistently**
   - Exact steps to trigger
   - Does it happen every time?
   - If intermittent: gather data (log frequency, timing, input patterns)

3. **Check Recent Changes**
   - `git diff` for last 3 commits
   - New dependencies, config changes
   - Import order changes (Python), barrel re-exports (TypeScript)

4. **Trace the Full Pipeline**

   **Frontend → Backend:**
   ```
   Browser fetch → middleware → FastAPI route → controller → core domain → provider → model
   ```
   At each boundary: what goes in? What comes out? Where does it diverge?

   **SSE streaming specifically:**
   ```
   AsyncGenerator → StreamingResponse → SSE parse → frontend handler → state update
   ```
   Common breaks: sync function blocking event loop, `request.is_disconnected()` not checked, envelope format mismatch.

5. **Add Diagnostic Logging**

   Don't guess — instrument. Add targeted prints/logs at component boundaries:
   ```python
   # Python
   import logging
   logger = logging.getLogger(__name__)
   logger.debug("entering phase X, input=%s", type(input_data))
   ```

   ```typescript
   // TypeScript — use console.debug, never console.log in prod
   console.debug('[ComponentName] state:', { key: value });
   ```

### Phase 2: Pattern Analysis

1. **Find Working Examples** — locate similar code that works
2. **Compare** — what's different between working and broken?
3. **Identify Dependencies** — what does this code path need to be true?

### Phase 3: Hypothesis and Testing

1. **Form Single Hypothesis** — "I think X is the root cause because Y"
2. **Test Minimally** — SMALLEST possible change, one variable at a time
3. **Verify** — did it work? If no, NEW hypothesis. Don't add more fixes on top.

### Phase 4: Implementation

1. **Create Failing Test Case** — simplest reproduction
2. **Implement Single Fix** — root cause only, one change, no "while I'm here"
3. **Verify Fix** — test passes, no other tests broken
4. **If Fix Doesn't Work:**
   - < 3 attempts: return to Phase 1
   - ≥ 3 attempts: STOP. Question the architecture.

## Common sloughGPT Bugs and Their Root Causes

| Symptom | Likely root cause | Fix |
|---------|------------------|-----|
| SSE streaming hangs | Sync function in async generator blocking event loop | `await asyncio.to_thread(sync_fn, ...)` |
| Server crash after first request | MPS false positive on Intel Mac | Force CPU: `_resolve_device()` returns `"cpu"` |
| Frontend test: `element not found` | StrictMode double-mount or portal | `getAllBy*` + `.length >= 1`, or `waitFor` for portals |
| `vi.mock` TDZ error | Mock declaration at module level without `vi.hoisted` | Wrap in `vi.hoisted(() => ({ fn: vi.fn() }))` |
| Gradient explosion in SloNet | `_layernorm` backward not reducing batch/seq dims | `.sum(axis=sum_axes)` on weight gradients |
| `generate()` non-determinism | Metal GPU accelerator floating-point variance | Disable accelerator during `generate()` |
| Provider never selected | `text_provider_name` already set by earlier registration | Check `or text_provider_name == "hf-default"` |
| Model load doesn't affect chat | Providers not re-registered after load | Call `setup_providers()` after `_load_hf_model_core()` |
| Settings crash in production | Zustand persist shallow-merge overwrites missing fields | Custom `merge` that deep-merges over defaults |
| Event loop `RuntimeError` | `asyncio.run()` inside already-running loop | Use `asyncio.to_thread()` or `asyncio.create_task()` |
| Test timeout (120s) | `setInterval` in component keeping vitest fork alive | Mock timers or extract testable logic |

## Red Flags — STOP and Follow Process

If you catch yourself thinking:
- "Quick fix for now, investigate later"
- "Just try changing X and see if it works"
- "It's probably X, let me fix that"
- "I don't fully understand but this might work"
- "One more fix attempt" (when already tried 2+)

**ALL mean: STOP. Return to Phase 1.**

## Verification After Fix

- [ ] Test that exposed the bug now passes
- [ ] No other tests broken: `npx vitest run` (frontend), `make test-py` (Python)
- [ ] `npx tsc --noEmit` passes (frontend changes)
- [ ] `python3 -m py_compile <file>` passes (Python changes)
- [ ] Edge cases handled (empty input, null, error paths)
- [ ] Fix doesn't introduce new coupling or architectural debt
