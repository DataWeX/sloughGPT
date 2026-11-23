---
id: 20261103_121721_slonet-dataset-scale-loop-feed-our-own-prepared-inputs-via-a
status: done
board: 20261103_121721_slonet-dataset-scale-loop-feed-our-own-prepared-inputs-via-a
title: "SloNet dataset scale loop — feed our own prepared inputs via ADAPTER into SloughGPTTrainer (blocked on gap benchmark)"
tags: ["sloNet", "dataset", "training"]
sprint: ""
gh: null
---

# SloNet dataset scale loop — feed our own prepared inputs via ADAPTER into SloughGPTTrainer (blocked on gap benchmark)

Self-sufficiency thread (user Q 2026-11-03): own inputs -> own dataset -> own stack. Existing plumbing: TrainingDataset, dataset_manifest.py, data_import.py, pair_extractor.py, experience_adapter.py, quality_scorer.py — this card feeds them AT VOLUME from our own inputs, through a dataset ADAPTER into SloughGPTTrainer (NO new training loop — known-debt rule; train_pipeline.py/chat_trainer.py/consciousness copies must not grow). Quality gate via quality_scorer. BLOCKED on card 20261103_121700 (gap benchmark) — sizing depends on its throughput number.
