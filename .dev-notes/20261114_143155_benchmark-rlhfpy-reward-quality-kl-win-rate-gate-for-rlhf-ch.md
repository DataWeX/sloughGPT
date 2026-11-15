---
id: 20261114_143155_benchmark-rlhfpy-reward-quality-kl-win-rate-gate-for-rlhf-ch
status: todo
board: 20261114_143155_benchmark-rlhfpy-reward-quality-kl-win-rate-gate-for-rlhf-ch
title: "benchmark_rlhf.py: reward quality / KL / win-rate gate for RLHF changes"
tags: [training, rlhf, benchmark]
sprint: ""
gh: null
---

# benchmark_rlhf.py: reward quality / KL / win-rate gate for RLHF changes

No RLHF benchmark exists (`scripts/benchmark_*` has none) — every rlhf.py /
RM change would be unmeasured, violating the run-benchmarks rule. Create
`scripts/benchmark_rlhf.py` measuring:

1. **RM held-out pairwise accuracy** on preference pairs (the sanity gate
   the RM trainer must pass before PPO consumes it);
2. **PPO run health** — mean reward trend up, approx KL bounded under
   `target_kl` (KL anchor not blown), entropy not collapsed, value loss
   decreasing;
3. **win-rate**: frozen RM or held-out pair ranking of policy-before vs
   policy-after samples.

Small fixed prompt set, deterministic seeds, comparable to other
`benchmark_*` scripts via `benchmark_results.py record/compare`. Depends:
rm-trainer card (needs a trained RM to measure); also covers existing
rlhf.py fixes (real rollout collection, reward conditioned on prompt+response)
once those land.

Board card `20261114_143155` (created 2026-11-14T14:31:55Z, chain 217).
