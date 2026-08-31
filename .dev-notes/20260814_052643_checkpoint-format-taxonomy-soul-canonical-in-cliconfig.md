---
id: 20260814_052643_checkpoint-format-taxonomy-soul-canonical-in-cliconfig
title: Checkpoint format taxonomy: .soul canonical in CLI/config
status: done
tags: training,checkpoint,format,cli
created: 2026-08-14T05:26:43.931850+00:00
---

Checkpoint format taxonomy: .soul canonical in CLI/config

Aligned the code-level checkpoint format taxonomy so .soul is canonical and sou is a legacy alias only.

Code fixes:
- config.yaml: checkpoint.export_format sou -> soul; comment updated.
- config_loader.py: CheckpointConfig.export_format default sou -> soul.
- apps/cli/src/commands/train.py cmd_train: save loop now reports the deterministic <save_path>.soul output (trainer.save() always writes .soul regardless of format param, which is an alias gate). Previously printed .sou paths that never existed on disk.
- cmd_train_native: --save-format default/fallback sou -> soul; removed the never-real npz option (trainer.save() warns and writes .soul for it).
- apps/cli/src/cli.py: --save-format Choice [sou, npz] -> [soul, sou], default soul.
- tests updated: test_config.py default assertion -> soul; test_train_commands.py native save-format assertions -> soul; smoke test dropped bogus use_mixed_precision kwarg.
- Docs: apps/cli/README.md line 31 and docs/FEATURES.md line 95 updated (soul canonical, no npz).

Verified: 49 passed CLI/config group + 125 passed trainer/checkpoint/eval group; pycache cleared.