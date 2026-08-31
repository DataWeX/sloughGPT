---
id: 20260809_111850_surprise-gated-learning-over-episodic-memory
title: Surprise-gated learning over episodic memory
status: done
tags: shell,world-realm,memory,learning
created: 2026-08-09T11:18:50.364810+00:00
---

Surprise-gated learning over episodic memory

Wired surprise-based learning (predictive coding) into SimBaby.learn() using the episodic-reward baseline: lr scale = 0.5 + min(|delta - mean_recent_reward|, 1.0), 0.5x..1.5x. Sign rule unchanged (gain reinforce / loss weaken) so energy conservation stays intact. info() now exposes a learning block {baseline, surprise, scale}. Added 5 tests (empty-memory max surprise, at-baseline floor scale, weight-update ratio 3x, learning block exposure). WORLD_REALM.md updated (learning rule, memory section, component table, test count 22). Verified: 434/434 scope tests pass (test_memory 22 + test_simulation 117 + test_evolution 20 + test_shell_repl 275); test_simulation stress 5/5 stable. Surprise was previously reverted pre-conservation for causing farming flakiness; conservation fix now makes it safe.