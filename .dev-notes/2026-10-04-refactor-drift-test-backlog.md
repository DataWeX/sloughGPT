---
id: 20261004_refactor_drift_test_backlog
title: "Refactor-drift backlog: tests still importing pre-refactor APIs (~300 of 619 failures) — sweep table inside"
status: todo
date: 2026-10-04
tags: [tests, refactor, debt]
board_id: 33773fce-b94e-410d-99c4-0007f96363a2
board: 33773fce-b94e-410d-99c4-0007f96363a2
hash: c36419bd5ff6e32f2ea6e03a9e58af233c7952b9c947b5e8de66ae7a4c909e46
---

Evidence: verification sweep 2026-10-04 on worktree /home/mana/Documents/sloughgpt-api-std @ 5bfbb8e08 (full suite 619F/45211P/180E in 1h56m; logs in ~/.cache/sweep-logs/: backend5.log full run, detail.log+detail2b.log exception detail, full-summary.txt, failed-by-file.txt).

The largest failure cluster is tests referencing APIs that recent refactors renamed/removed (each line: failing count — missing symbol — presumed refactor):

- 83F test_shell_repl_more — `ShellREPL._interpret_natural` gone (+15x NoneType.shutdown, _CaptureOutput.getvalue) — shell lazy-loading refactor 89e01b58c era
- 24F+14F+4F kernel_devices/neural/devices — device `open()/info/read_fd/write_fd` API changes
- 16F+2F mobile/multimodal routers — `domain.mobile.notifications`, `domain.training.mobile_training_store/video_trainer/auto_trainer` gone (knowledge/memory + plural-tree rehomes: 5bf0d7400, 8e1a0c75c)
- 7F kb_router + 1F files_router — stale `@patch("domain.learner.knowledge...")` (runtime OK: kb.py imports domain.knowledge.engine)
- 11F+6F+ vm family — `FakeNPU.load_file`, `domain.shell.vm` module path, response shape Keys programs/isa/registers
- 5E+1F logging/cli — `domain.logging.cli_logger` → `CLILogger` rename (stale consumers)
- 4F models_init — `domain.models` lost RMSNorm/SloughGPTAttention/SloughGPTBlock/SwiGLU exports
- 5F companion + 3F io — mocks missing `respond/traits/_interactive`
- 4F lora_eval + 5F meta_weights + 4F memory + 12F self_train + 7F user_adapters + 3F security — route registration/response-shape drift
- misc: logging_factory 10F (record schema), pair_extractor 8F, type-as-string 3F (`assert 'float' is float`), wandb 5F, training_stream 5F, truth_context 4F, quantization 5F (`QuantizedLinear.make_torch_forward`)

DECISION NEEDED per group: fix the test (if refactor intentional) or restore the API (if other consumers still need it). Suggested: process alphabetically, one commit per group, rerun file until green. NOT env-caused (verified deterministic across 3 runs at different loads).

COORDINATION: each symbol maps to a named refactor commit above — check whether that session already planned test updates (kanban) before duplicating.
