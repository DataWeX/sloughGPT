---
id: 20260808_021206_task-queue-training-handler-tests-executor-bugfix
title: Task-queue training handler tests + executor bugfix
status: done
tags: 
created: 2026-08-08T02:12:06.551228+00:00
---

Task-queue training handler tests + executor bugfix

Added packages/core-py/tests/test_training_queue.py (9 tests) covering training_handler + training_sessions_handler. Tests exposed 3 real bugs in training_queue.py: (1) pause_event was initialized SET, and train_pipeline.train() treats set==paused, so every native task-queue training hung forever at the first step; (2) blocking trainer.train()/train_from_sessions() starved the cancel/pause bridge coroutines (no awaits during training); (3) cancelled training returned a success TrainResult because _emit_progress swallows on_progress exceptions. Fixed: clear pause_event at init, run training via asyncio.to_thread so bridges pump the loop, post-check cancel_event to emit sse_complete(cancelled)+status cancelled, and moved SloughGPTTrainer construction inside try so prepare_data failures surface as failed results. Verified: 9/9 new tests, 299 targeted core-py, 44 task_queue, 14/14 auto-train e2e.