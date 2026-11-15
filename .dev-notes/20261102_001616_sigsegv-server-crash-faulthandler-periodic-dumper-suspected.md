---
id: 20261102_001616_sigsegv-server-crash-faulthandler-periodic-dumper-suspected
status: done
board: 20261102_001616_sigsegv-server-crash-faulthandler-periodic-dumper-suspected
title: "SIGSEGV server crash: faulthandler periodic dumper suspected — fix shelved, pick up (sys._current_frames walk vs drop timer)"
tags: ["crash", "sigsegv", "faulthandler", "api"]
sprint: ""
gh: null
---

# SIGSEGV server crash: faulthandler periodic dumper suspected — fix shelved, pick up (sys._current_frames walk vs drop timer)

apps/api/server/main.py _install_stack_dump_timer (line ~234) / _cancel_stack_dump_timer (~259). faulthandler.dump_traceback_later periodic dumper was root-caused as the SIGSEGV trigger. User ruling at time: keep the dumper, dump CURRENT THREAD only — then shelved after it was shown dump_traceback_later has no all_threads parameter, and a current-thread-only dump from the timer thread would just print the timer's own useless stack. Real options to record: (a) Python-level fix: schedule the dumper on a timer but have it walk sys._current_frames() and write each thread's traceback via faulthandler.dump_traceback(file) — per-thread dumps without faulthandler's C-level all-threads walk, (b) drop the periodic timer entirely and dump only synchronously from signal handlers / on-watchdog, (c) keep as-is if a later repro disproves the causality. Evidence + repro steps from the original diagnosis session should be re-checked before choosing (this was shelved, not verified to root). DoD: reproduce or disprove, land the chosen mechanism with a test, remove the shelved state.

CLOSED 2026-11-03: superseded — option (a) implemented in 392fa637a (GIL-safe thread + sys._current_frames), ancestor of main; done-card 20261102_020229 cites it.
