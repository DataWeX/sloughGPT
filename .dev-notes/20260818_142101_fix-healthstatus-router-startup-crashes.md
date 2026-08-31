---
id: 20260818_142101_fix-healthstatus-router-startup-crashes
title: Fix health/status router startup crashes
status: done
tags: router,health,bugfix
created: 2026-08-18T14:21:01.621688+00:00
---

Fix health/status router startup crashes

Fixed two router startup crashes:

1. health.py: Added missing liveness() and model_health() methods, return type annotations, docstrings
2. status.py: Added missing router export, completed live() method body, return types, docstrings

Root cause: stale .pyc bytecode cache caused 27 false router import warnings. Clearing __pycache__ resolved them.

Verified: 47 routes registered, zero warnings, all health endpoints responding.