---
id: 20261112_060632_consolidate-tests-into-features-co-locate-suites-with-the-fe
status: done
board: 20261112_060632_consolidate-tests-into-features-co-locate-suites-with-the-fe
title: "Consolidate tests into features: co-locate suites with the features they gate"
tags: []
sprint: ""
gh: null
---

# Consolidate tests into features: co-locate suites with the features they gate

User directive (2026-11-11, verbatim): 'consolidate some of our tests into our features — that's how to really learn some tests; [our] tests [are] powerful enough to use their features in our codebase to work and introduce and integrate with and adapt new features, systems and processes.'

Context: split off from card 20261101_120913 (training-adapter-gate) per user ruling — finish that card first, then card the consolidation separately (that card landed the ADR-002 + BatchArray seam + composition gate in 59b650334).

Interpretations to resolve at intake (user to pick):
1. Co-locate tests with feature code (test files beside the features they cover, e.g. under domain/training/).
2. Fold overlapping suites into per-feature gates (test_command_parity / composition-gate shape applied across scattered suites — related to Test Rule audit card 299d8d38).
3. Expose gates as product features (CLI / Site Doctor runnable checks).

Constraints when built: card-first, branch, TDD, Test Rule clauses 1-5 (owned-code only, no redundancy, change-scoped locally + full suite in CI, seam gates ride their surface, trimming is a card). Note: tests/core-py/ placement is what CI's test-core-py job picks up — any move must keep CI coverage intact.
