---
id: 20260810_235346_incremental-feedback-training-loop
title: Incremental feedback training loop
status: done
tags: feedback,training
created: 2026-08-10T23:53:46.568797+00:00
---

Incremental feedback training loop

Wired OnlineLoRAUpdater/PerUserLORAStore into a continuous background loop.

Root cause of the loop never running: _run_background_training read the tokenizer from lora_updater._tokenizer (an attribute OnlineLoRAUpdater does not have) -> tok always None -> early return. Fixed to read self._tokenizer (set via set_model()).

Wiring gaps closed:
- controllers/feedback.py _wire_model() now falls back to server_state.model/tokenizer when no auto-train student_net is set (student_net is never populated in production, so set_model was previously never called).
- main.py _start_feedback_workflow() wires the active server model into the workflow at startup.

Tests: 4 new core-py workflow tests (background training uses workflow tokenizer, skips without model/tokenizer, needs 2 recent items) + 4 new server tests (test_feedback_controller_wiring.py). Verified: test_workflow 23 pass, test_online_lora + test_feedback_router pass, server feedback router 16 pass.