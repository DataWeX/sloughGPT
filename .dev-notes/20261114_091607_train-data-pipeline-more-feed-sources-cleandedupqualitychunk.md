---
id: 20261114_091607_train-data-pipeline-more-feed-sources-cleandedupqualitychunk
status: todo
board: 20261114_091607_train-data-pipeline-more-feed-sources-cleandedupqualitychunk
title: "train-data pipeline: more feed sources + clean/dedup/quality/chunking + feed hardening"
tags: ["training", "data-pipeline", "feeds"]
sprint: ""
gh: null
---

# train-data pipeline: more feed sources + clean/dedup/quality/chunking + feed hardening

GOAL: improve the training data pipeline in three areas (brief from coordinator session):
1. MORE DATASET SOURCES — extend _SOURCES registry in apps/api/server/training/feeds.py + TrainingFeedClient candidate lookup: generic file-directory source (file-drops: drop *.jsonl in data/file_drops/), url-list source (data/feeds/urls.txt: fetch+parse a list of URLs), and runs/imports/<name>/corpus.jsonl candidates so data_import.py outputs resolve as feeds.
2. BETTER DATA PIPELINE — (a) cleaning: strip HTML, normalize whitespace, remove boilerplate in domain/training/_internal/cruncher/clean.py wired into Cruncher._ingest_record; (b) dedup: add exact content-hash dedup alongside existing MinHash near-dup in DedupIndex; (c) quality: add detect_language() to quality_scorer.py + dominant language in compute_data_quality; (d) chunking: TrainingDataset gains strategy param routing to existing domain/memory/_internal/knowledge_store.chunk_text (auto: heading/paragraph/semantic), default fixed keeps contract.
3. PRODUCTION HARDENING — feeds.py + training_feed.py: streaming pagination (_source_lines reads ALL lines today → O(1)-memory paged reads), corrupted-JSONL repair (repair_json_line: balanced-prefix / close-truncated / cut-back salvage), cursor recovery (persisted offset past EOF resets to 0), feed health fields (kind/bytes/freshness_seconds/healthy) on existing /training/feeds rows — registry entries, NO new endpoints (descriptor rule).

PORT-DON'T-REBUILD CHECK: cruncher/ (clean.py MinHash dedup, quality gate, PII scrub) already exists — extend it, do not build a parallel cleaner. knowledge_store.chunk_text already has auto/fixed/paragraph/heading/semantic — reuse, do not copy. No TrainerProtocol/TrainResult changes.

OWNED PATHS: apps/api/server/training/feeds.py, domain/training/_internal/{training_feed,quality_scorer,dataset}.py, domain/training/_internal/cruncher/{clean,pipeline,__init__}.py, domain/memory/_internal/knowledge_store.py (chunk_text kwarg mapping only), tests: tests/core-py/test_training_feed.py, apps/api/server/tests/test_training_feed_router.py, tests/test_cruncher.py, tests/core-py/test_training_quality_methods.py, tests/core-py/test_dataset.py, scripts/benchmark_feed_read.py (new), docs/AGENT_SYNC.md.

DoD: ruff check+format clean on owned paths; ./scripts/run_affected.sh green (incl. existing feed/cruncher/dataset/quality suites — no regressions: total/offset semantics preserved, TrainingDataset default chunking byte-identical); benchmark script documents streaming vs read_text baseline; commit pathspec-limited on feat/training-data-pipeline; AGENT_SYNC entry.
