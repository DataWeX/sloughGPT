---
id: 20261104_144058_featbigdata-compose-bundle-clean-seal-train-data-pipeline-li
status: todo
board: 20261104_144058_featbigdata-compose-bundle-clean-seal-train-data-pipeline-li
title: "feat(bigdata): compose bundle->clean->seal->train data pipeline + live-feed seam ruling"
tags: ["training", "data", "pipeline", "design", "area:training"]
sprint: ""
gh: null
---

# feat(bigdata): compose bundle->clean->seal->train data pipeline + live-feed seam ruling

STATUS: card only, no code yet (user ruling 2026-11-04: "play it safe first" -> file the card, keep discussing).

GOAL: a four-stage big-data training pipeline — bundle -> clean -> prepare(seal) -> harvest for training, across large and multiple formats — plus a live-data feed feeding the loop. Big data volume first, GPU/torch NOT in scope (see 20261104_111333: resurrection needs volume AND a measured GPU-side win; measured 1.0-1.2x today per 20261104_100637).

KEY FINDING: all four stages already EXIST as pieces — this is composition, not greenfield (port, don't rebuild):
- bundle: domain/training/_internal/data_import.py (GitHub/HF/URL/local/pack import)
- clean: domain/training/_internal/cruncher/clean.py (scrub_pii, quality_ok, DedupIndex against ALL persisted history)
- prepare/seal: domain/training/_internal/cruncher/pipeline.py::seal -> immutable v<N> snapshot + dataset_manifest.py
- harvest: dataset adapter -> SloughGPTTrainer (pattern proven by FeedBatchSampler in training_feed.py, BatchSampler protocol)
- live feed: training_feed.py (offset-cursor JSONL/HTTP, "feed:<source>" as data_path)

INVARIANT: mutable ingest on one side, immutable training input on the other, sealed between them. Dedup index travels inside each snapshot, so history resumes across runs.

DESIGN REFINEMENTS AGREDED IN REVIEW (2026-11-04):
1. Mechanism/placement — the live feed terminates at the FEED seam, never at the snapshot seam. Batch harvest = sealed snapshot (bytes immutable -> dataset_signature computable at seal, satisfies 20261104_081626); live harvest = BatchSampler over the cursor (append-only, unbounded, NO stable signature by construction). The fork already lives at train_pipeline.py ~L1490 ("feed path owns streaming"). KNOWN GAP to rule on explicitly: the live feed bypasses DedupIndex, so live records can duplicate sealed-corpus content — accept or add light dedup first; decide, don't inherit silently.
2. Scope — multi-format is a bundling-stage concern ONLY: normalize to one record schema {text, source, meta} at stage 1; nothing downstream of bundling knows what a PDF/HTML/JSONL is, and the trainer never learns about formats or shards (that is the second-loop birth canal per AGENTS.md one-loop rule).
3. The trainer's real wall is its input contract, not ingest: dataset_manifest.py says "the char-level trainer expects a single UTF-8 text file". Volume dies at that seam -> fix lives in ONE sharded/streamed dataset adapter feeding SloughGPTTrainer, never in teaching the trainer about shards.

PLAN (steps, in order):
1. This card (done).
2. MEASURE FIRST: bundle a corpus 10-100x tinyshakespeare through crunch_stream -> seal, record where time/memory goes (tokenize vs dedup vs read). Blocked on input: real dataset on disk (path TBD) vs synthesized corpus.
3. Branch feat/bigdata-pipeline, TDD: RED (a) composed pipeline end-to-end on a multi-format bundle, (b) seal twice over identical bytes -> identical dataset_signature, (c) the live-feed fork never touches snapshot state.
4. Compose, don't rewrite: one thin orchestrator over data_import + cruncher + adapter. Delete nothing.
5. Gates: scripts/run_affected.sh closure, ruff, ingest->train tokens/s benchmark number (new scripts/benchmark_*.py since this is a training implementation).
6. Land green, then re-evaluate 20261104_111333 with the measured number.

RELATED: 20261104_081626 (dataset_signature at seal — this card's seal-time signature mechanism closes it), 20261104_111333 (torch resurrection trigger), 20261104_100637 (GPU 1.19x ruling), 6b44763fd (own-inputs dataset adapter precedent).

OBSERVATION (for 20261104_100954): the installed console script /home/mana/.local/bin/planner is broken — ModuleNotFoundError: No module named 'planner.cli' (stale entry point; the grib rename's pip reinstall is the owed work). app_planner runs fine as a module with PYTHONPATH=packages/app-planner/src.
