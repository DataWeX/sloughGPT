---
id: 20261101_140448_test-core-py-green-baseline-fix-or-honestly-quarantine-the-c
status: done
board: 20261101_140448_test-core-py-green-baseline-fix-or-honestly-quarantine-the-c
title: "test-core-py green baseline: fix or honestly quarantine the chronic reds (quantization e2e, shell TUI, VM benchmarks)"
tags: ["roadmap", "platform"]
sprint: ""
gh: null
---

# test-core-py green baseline: fix or honestly quarantine the chronic reds (quantization e2e, shell TUI, VM benchmarks)

ROADMAP.md Next/Platform. PRs currently merge only after proving failures pre-existed. Fix or quarantine (pytest.mark.slow precedent PR #174). A permanently red baseline stops being visible debt the day someone starts believing it.

CLOSED 2026-11-03: green baseline achieved — CI test-core-py success on consecutive runs; quarantine evidence in /tmp/opencode/quarantine-run.txt (shell TUI 18 passed, quant e2e -m slow 17 passed, vm benchmark -m slow 71 passed).
