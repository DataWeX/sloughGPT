---
id: 20261102_014255_vm-pre-run-showcase-show-what-the-12-programs-do-before-run
status: done
board: 20261102_014255_vm-pre-run-showcase-show-what-the-12-programs-do-before-run
title: "VM pre-run showcase — show what the 12 programs do before Run (audit finding 1, user-decided direction)"
tags: []
sprint: ""
gh: null
---

# VM pre-run showcase — show what the 12 programs do before Run (audit finding 1, user-decided direction)

Audit finding 1 (card 20261101_104216): VM assembly lands on an empty 'No result yet' state — '12 programs, nothing to play with'. User decision 2026-11-02: PRE-RUN SHOWCASE ON LOAD (chosen over Browser-VM promotion and playable demos). Build: showcase panel in apps/web/app/(app)/vm/ rendered in the pre-run state — sample outputs/previews for each of the 12 programs so first paint is legible and invites interaction. Rules: real captured outputs where available, clearly-labeled samples otherwise (NEVER fake live results); Noir Violet design system locked (no new hues/fonts/spacing); post-run behavior unchanged (Run still replaces showcase with real output). TDD: pre-run showcase renders samples; run replaces showcase with real result. Gates: tsc, eslint, vitest vm closure.
