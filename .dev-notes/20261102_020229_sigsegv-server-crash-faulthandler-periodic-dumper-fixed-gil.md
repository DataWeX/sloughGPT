---
id: 20261102_020229_sigsegv-server-crash-faulthandler-periodic-dumper-fixed-gil
status: done
board: 20261102_020229_sigsegv-server-crash-faulthandler-periodic-dumper-fixed-gil
title: "SIGSEGV server crash: faulthandler periodic dumper — FIXED (GIL-safe thread, 392fa637a), verify clean window"
tags: ["crash", "sigsegv", "faulthandler", "api"]
sprint: ""
gh: null
---

# SIGSEGV server crash: faulthandler periodic dumper — FIXED (GIL-safe thread, 392fa637a), verify clean window

apps/api/server/main.py _install_stack_dump_timer (line ~234) / _cancel_stack_dump_timer (~259). faulthandler.dump_traceback_later periodic dumper was root-caused as the SIGSEGV trigger. User ruling at time: keep the dumper, dump CURRENT THREAD only — then shelved after it was shown dump_traceback_later has no all_threads parameter, and a current-thread-only dump from the timer thread would just print the timer's own useless stack. Real options to record: (a) Python-level fix: schedule the dumper on a timer but have it walk sys._current_frames() and write each thread's traceback via faulthandler.dump_traceback(file) — per-thread dumps without faulthandler's C-level all-threads walk, (b) drop the periodic timer entirely and dump only synchronously from signal handlers / on-watchdog, (c) keep as-is if a later repro disproves the causality. Evidence + repro steps from the original diagnosis session should be re-checked before choosing (this was shelved, not verified to root). DoD: reproduce or disprove, land the chosen mechanism with a test, remove the shelved state.

## Resolution 2026-11-02 — proven, fixed, landed (option a)

**Root cause CONFIRMED by core dumps** (crash reproduced live, no need to re-check old notes):
- 23 SIGSEGVs over 2026-10-31..11-02; crash cadence every 4-14 min (tonight 01:30:48, 01:39:21, 01:43:01, 01:53:13); one hit during IDLE STARTUP — no traffic required. systemd auto-restarts, so the service "looks up" while requests hang.
- 3 retained cores (pids 419307, 84130, 106351) all die at the IDENTICAL instruction: frame#0 `python3.12 + 0x1a3c7b` → `dump_frame`, `Python/traceback.c:1259` (`code->co_filename` off a foreign thread's already-torn-down frame).
- Mechanism: the periodic timer's C-level walk of raw `_PyInterpreterFrame` runs in signal/timer context WITHOUT the GIL and races normal frame teardown (GIL-governed frame lifetime).
- numba ruled out: installed Nov 1 13:10, first SEGVs Oct 31.

**Fix landed: `392fa637a`** (branch fix/fault-periodic-dumper-gil-safety) — periodic timer replaced with a daemon thread snapshotting `sys._current_frames()` (taken under GIL; frame objects pin the chain). On-crash faulthandler in `infrastructure/startup.py` untouched. 8 new contract tests (`tests/server/test_stack_dump.py`); gate: ruff clean, `tests/server/` 2037 passed / 1 skipped.

**Deployed:** slough-api restarted 02:53:46 WAT on 392fa637a; log line `stack dump active every 30.0s (SLO_DUMP_STACKS=1, gil-safe)`; 30s dump ticks present in journal.

**DoD remaining:** clean journal window — baseline is a crash every 4-14 min, so 30-60 min without `status=11/SEGV` is meaningful evidence. Then close.
