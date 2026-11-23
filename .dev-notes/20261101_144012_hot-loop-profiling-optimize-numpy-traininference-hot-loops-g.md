---
id: 20261101_144012_hot-loop-profiling-optimize-numpy-traininference-hot-loops-g
status: todo
board: 20261101_144012_hot-loop-profiling-optimize-numpy-traininference-hot-loops-g
title: "Hot-loop profiling: optimize NumPy train/inference hot loops (gates the PyO3 kernel decision)"
tags: ["roadmap", "decision", "model", "area:training", "track:next"]
sprint: ""
gh: null
---

# Hot-loop profiling: optimize NumPy train/inference hot loops (gates the PyO3 kernel decision)

Committed in the model-candidates decision session 2026-11-01 (card 20261101_140449). Goal 1's stated remnant: profile and optimize the pure-NumPy train/inference hot loops. Gates the deferred PyO3 kernel rewrite (ROADMAP deferred table: only if profiling shows Python loop overhead is the bottleneck; 10-50x potential). Known lead: prefill attention O(seq^2) numpy fallback (card 20261101_110410). Close with: before/after profile evidence + benchmark_results record --kind training.
