---
title: Fix pre-existing test debt (session_core, chat_trainer, weight_loader, memory_pressure)
status: done
created: 2026-09-24
tags: [testing, debt]
sprint: current
gh: null
body: |
  Fixed 4 pre-existing test clusters on feat/training-resume-ux (no commit unless asked).
  (1) test_session_core: retargeted fixture patch domain.infrastructure.session_core.get_message_feedback → domain.infrastructure._internal.session_core.get_message_feedback (SessionCore lives on _internal).
  (2) test_training_queue: retargeted 30 monkeypatch strings domain.training.chat_trainer.* → domain.training._internal.chat_trainer.* (matches training_queue late-function import).
  (3) test_memory_pressure: removed contradictory assert that _classify(85)==NORMAL after configure(warning=82); kept correct WARNING assert.
  (4) weight_loader impl + test: (a) infer_arch_from_state_dict now also matches blocks.*.ff.w1.weight (native SloNet naming), not only mlp.w1/gate_proj; (b) fused QKV splits original axis-0 then transposes each chunk (was np.split(arr.T) which fails for (12,4) weights); same fix in DirectWeightLoader mmap path; (c) register_loader lowercases suffix + getattr(__name__, str) so case-insensitive get_loader and MagicMock registry work; (d) test_fused_qkv_bias mock bias params resized to 1-D (4,) to match real bias layout.
  Gates: 158 passed on the 4 files, ruff clean. Kanban 20260924_044 → done.
status_detail: done
---

## Regression fix (post-gate)
First fused-split edit broke `test_direct_loader::test_gpt2_direct_load_matches_fused`
(GPT-2 stores `(n, 3n)`; needs transpose-then-split). Replaced both call sites with
shape-aware `_split_fused_qkv_weight()`:
- `(3n, n)` → split rows, transpose each chunk
- else (HF `(n, 3n)`) → transpose, split rows

Final gate: **239 passed** (4 target files + test_direct_loader), ruff clean.
