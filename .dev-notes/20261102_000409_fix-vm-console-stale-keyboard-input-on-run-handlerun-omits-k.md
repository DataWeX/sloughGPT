---
id: 20261102_000409_fix-vm-console-stale-keyboard-input-on-run-handlerun-omits-k
status: done
board: 20261102_000409_fix-vm-console-stale-keyboard-input-on-run-handlerun-omits-k
title: "Fix VM console stale keyboard input on Run — handleRun omits keyboardInput from useCallback deps"
tags: []
sprint: ""
gh: null
---

# Fix VM console stale keyboard input on Run — handleRun omits keyboardInput from useCallback deps

Real pre-existing bug flagged during the vm page decomposition (AGENT_SYNC 20261101_052821, 'follow-up candidate'): apps/web/app/(app)/vm/useVmPage.ts handleRun reads keyboardInput inside the callback but its useCallback deps are [source, maxSteps, role, debug] — keyboardInput missing. Consequence: type into the Keyboard input field, press Run without touching anything else -> vmController.run receives the STALE keyboardInput from when the callback was last rebuilt (typically undefined/empty), so guest programs reading INT 0x80 keyboard input silently get old/empty input. The eslint react-hooks/exhaustive-deps warning (pre-existing, recorded through the decomposition) points at exactly this. FIX: add keyboardInput to the deps array (behavior: callback rebuilds on each keystroke — cheap, it already does for source). TDD: colocated page.test.tsx test typing into the keyboard input then running asserts run() receives the typed value; expect RED pre-fix (stale/undefined) -> GREEN post-fix. Gates: tsc, eslint (warning count should DROP by 1), vitest vm closure, audit greps. benchmark n/a (bugfix). SOP: branch fix/vm-keyboard-input-stale-closure from origin/main, push HEAD:main with AGENT_SYNC entry, card done with evidence.
