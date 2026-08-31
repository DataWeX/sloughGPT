---
id: 20260815_050246_fix-turbophase-type-error-in-usetrainingsession
title: Fix turboPhase type error in useTrainingSession
status: done
tags: frontend,web,typescript,training
created: 2026-08-15T05:02:46.182982+00:00
---

Fix turboPhase type error in useTrainingSession

tsc failed on hooks/useTrainingSession.ts:494. The turboPhase ternary used 'phase === TRAINING && method === turbo' as the first condition, so the negated conjunction in the final branch prevented TS from narrowing TRAINING out of training.phase, producing a union-type assignability error. Restructured the discriminant as the outer condition: phase === TRAINING ? (method === turbo ? training : idle) : phase. Behavior identical (verified truth table), TS narrows correctly. tsc --noEmit exit 0. useTrainingSession tests 15/15, training page tests 11/11 pass.