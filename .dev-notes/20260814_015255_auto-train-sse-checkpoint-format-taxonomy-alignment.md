---
id: 20260814_015255_auto-train-sse-checkpoint-format-taxonomy-alignment
title: Auto-train SSE + checkpoint format taxonomy alignment
status: done
tags: training,checkpoint,sse,format
created: 2026-08-14T01:52:55.454136+00:00
---

Auto-train SSE + checkpoint format taxonomy alignment

Auto-train SSE streaming hardened after task-queue migration: terminal-event backstop on all completion/error/cancel paths, idempotent queue/worker (34/34 green).

Checkpoint format taxonomy (user-defined): .slo = personality metadata (SouParser.parse -> SloProfile), .soul = checkpoint file, .slnc = compiled mmap model.
- auto_train.py: .pt removed; catalog handles .soul + .slo; _load_soul_meta parses .slo via SouParser; _load_soul blanks self-name only for .soul; export_checkpoint_mobile restricted to .soul (was calling import_from_sou on .slo metadata) + test.
- CLI models cmd: 'PyTorch Checkpoints (.pt)' section -> 'Compiled Models (.slnc)' via rglob.
- train_pipeline.save(): default format='soul' (was 'sou'), legacy 'sou' alias kept; call sites updated.
- model_catalog format set = 'slnc' | 'safetensors' | 'soul'.
- SloTextEmbedder: default checkpoint path text-embedder.sou -> text-embedder.soul; vocab/bpe sidecars now extension-agnostic via os.path.splitext (fixes .soul -> '-vocab.jsonl' corruption); legacy explicit .sou paths still load. 3 new tests; embedder/embedding-service/kb/CLI all green.

Remaining intentional legacy: export_to_sou/import_from_sou API names (30+ import sites), pt_loader.py (torch-free .pt reader), export.py .pt/.gguf output formats, models/__init__ .sou suffix read-compat.