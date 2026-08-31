---
id: 20260812_012043_slonet-native-training-memory-leak-fix-train-native-cli
title: SloNet native training: memory-leak fix + train native CLI
status: done
tags: training,cli,core
created: 2026-08-12T01:20:43.590944+00:00
---

SloNet native training: memory-leak fix + train native CLI

  ## Goal
  Torch-free SloNet training must run from the CLI (core -> CLI -> API build order). Fix the per-step memory leak that blocks long training, surface native training as a first-class CLI command, verify regression, commit+push.
  
  ## Root cause
  backward() never cleared node._consumers. Persistent model parameters pinned every step's computation graph (~94MB/step leak). 250 steps went 11GB+ before the fix.
  
  ## Fix
  - slonet.py: Tensor.backward() clears node._consumers after the reverse pass.
  - Memory verified bounded: sawtooth RSS 0.8-2.1GB across 2450-step run with periodic drops to baseline after checkpoint saves; no monotonic growth.
  - Regression tests added: test_backward_clears_consumers + test_backward_then_forward_grad_still_works (33 DAG tests pass).
  
  ## CLI
  - cli.py: 'train native' click command; commands/train.py cmd_train_native() (config-driven, auto-saves .soul + final sloughgpt-native.soul).
  - 5 new CLI tests in apps/cli/tests/test_train_commands.py; 182 total regression green.
  - CLI smoke run --steps 25 produced + loaded a .soul end-to-end.
  
  ## Verification
  - Real 2450-step run (PID 225601, epochs=5) at ~1.5-2.5 it/s, final_loss trend 4.76 -> 2.7, eval_loss periodic.
  - run_real_train.py TrainResult JSON serialization fixed (_jsonable); epochs=5.
  - API training router already a thin wrapper (start_training drives SloughGPTTrainer, saves .soul) - no changes needed.
  
  ## Commits
  - b7007f4 fix(training): eliminate per-step memory leak; add 'train native' CLI
  - 5739d89 test(slonet): consumer-graph release regression tests
  - 509ec172 docs(agents): 'train native' CLI + memory discipline