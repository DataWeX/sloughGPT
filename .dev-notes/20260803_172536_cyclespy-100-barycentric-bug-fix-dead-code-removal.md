---
id: 20260803_172536_cyclespy-100-barycentric-bug-fix-dead-code-removal
title: cycles.py 100% — barycentric bug fix + dead code removal
status: done
tags: shell,cycles,coverage,testing
created: 2026-08-03T17:25:36.690263+00:00
---

cycles.py 100% — barycentric bug fix + dead code removal

cycles.py + cycles_device.py at 100% coverage (570 stmts, 0 missed). Barycentric fix: _triangle_barycentric replaced cross-product formula with proper d00/d01/d11/d20/d21 barycentric coordinates (w3=1-w1-w2), degenerate-triangle fallback 1/3,1/3,1/3. Dead code removed: _direct_light() + _single_intersect() (0 references). Verified: 61 cycles tests + lifecycle + tui_repl + tui_live all pass; py_compile OK.