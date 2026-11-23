---
id: 20261112_124359_fixweb-sanitize-training-hp-inputs-via-sanitizets-nan-safe-c
status: done
board: 20261112_124359_fixweb-sanitize-training-hp-inputs-via-sanitizets-nan-safe-c
title: "fix(web): sanitize training HP inputs via sanitize.ts (NaN-safe, clamp, LR=0.001)"
tags: ["web", "training", "sanitization"]
sprint: ""
gh: null
---

# fix(web): sanitize training HP inputs via sanitize.ts (NaN-safe, clamp, LR=0.001)

User-reported (screenshot 2026-11-12): batch size / learning rate inputs have a sanitization issue; learning rate showed 1, expected 0.001. APPROVED SCOPE (user picked 'Clamp + VM form too'): (1) ConfigureStep.tsx + useTrainingForm.ts use clampTrainingField/TRAINING_LIMITS from apps/web/lib/sanitize.ts as single source of truth (epochs 1-1000, batch 1-512, lr 1e-5-1.0 — hand-written 1-500/1-256/(0,1] drift), NaN-safe parsing, no Number(x)||1e-3 silent fallback; clamp saved-config load in useTrainingForm (unclamped LR=1 persisted and reloaded = the reported bug); (2) VMTrainingConfigForm.tsx local num() helper migrates to parseAndClamp. LR input must be typeable (0.001 entry) — draft-string + commit/clamp on blur pattern, not clamp-on-every-keystroke. Update ConfigureStep.test.tsx limit strings. Gates: vitest affected files + tsc.
