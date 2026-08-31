---
id: 20260803_175719_shell-coverage-simulationsurfaceruntimedevices-to-100
title: Shell coverage: simulation/surface/runtime/devices to 100%
status: done
tags: coverage,shell
created: 2026-08-03T17:57:19.724955+00:00
---

Shell coverage: simulation/surface/runtime/devices to 100%

Raised to 100%: simulation.py (was 99), surface.py (97), runtime.py (83), devices.py (81). New tests: surface base/py partial-render; sim dead-baby skip, verbose log, negative-radius empty, run-break, empty summary; runtime boot/shutdown/api/status; devices LLM/knowledge/vision/embedding/proc/manager fallback branches. 25 shell modules at 100%. Remaining: tui_repl.py 45% (_main curses loop), kernel_npu.py 87% (needs psutil/torch), repl.py, vm.py.