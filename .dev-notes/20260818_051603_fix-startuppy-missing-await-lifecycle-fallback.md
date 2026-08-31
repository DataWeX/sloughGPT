---
id: 20260818_051603_fix-startuppy-missing-await-lifecycle-fallback
title: Fix startup.py missing await + lifecycle fallback
status: done
tags: api,startup,bugfix,status:done
created: 2026-08-18T05:16:03.710858+00:00
---

Fix startup.py missing await + lifecycle fallback

Fixed two bugs in apps/api/server/infrastructure/startup.py that caused feature routers (agents, meta-weights, user-adapters, etc.) to never register:

1. Missing await: _phase5_model_registry() and _phase6_routers() are async functions but were called without await in the fallback path (line 297-298). Coroutines were created but never executed.

2. Lifecycle failure not handled: When lifecycle.start() returned False (e.g., model load failed), code logged a warning and continued without running fallback phases. Feature routers were never registered.

Fix: Both paths now properly await the phase functions, and lifecycle failure triggers the fallback path.

Verified: py_compile passes, git diff clean.