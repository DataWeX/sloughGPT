---
id: 20260924_037_goal12-owned-arch
status: done
title: Goal 12 owned architecture e2e
tags: roadmap,training,owned-model
---
# Goal 12 — Owned architecture

## Plan
1. Explore .soul train → .slnc serve gap; find HF on default path
2. Ship smallest e2e: train tiny → .soul → .slnc → serve without HF
3. Tests + ruff + scatter gate; ROADMAP 12 Done if acceptance met
4. Clean worktree feat/goal12-owned-arch-e2e off main

## Result
- Main already shipped goal 12 bridge (2af86021 / f009377c): soul_to_slnc, priority flip, SLO_BOOTSTRAP_HF, compile_slnc, 7/7 owned_path tests.
- Remaining bug fixed on main (d1068247): from_soul rebuilt abs-pos + LayerNorm instead of RoPE + RMSNorm.
  - from_soul uses load_soul, infers intermediate/use_rope/norm_type, rebuilds native arch
  - build_model_from_config honors use_rope/norm_type/max_seq_len
  - train_pipeline persists arch flags + intermediate_size in soul.config
  - soul_to_slnc emits explicit use_rope/norm_type
  - new tests/test_goal12_owned_arch_e2e.py (5 tests) green
- Gate: 178 passed (e2e + owned_path + scatter + train_pipeline); ruff clean.
