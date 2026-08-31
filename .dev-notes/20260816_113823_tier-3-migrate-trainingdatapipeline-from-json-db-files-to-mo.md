---
id: 20260816_113823_tier-3-migrate-trainingdatapipeline-from-json-db-files-to-mo
title: Tier 3: migrate TrainingDataPipeline from JSON .db files to MogDB collections
status: done
tags: mogdb,migration,training-pipeline
created: 2026-08-16T11:38:23.991304+00:00
---

Tier 3: migrate TrainingDataPipeline from JSON .db files to MogDB collections

Fifth code-quality pass (robustness/portability): export_training_data writes via temp file + os.replace (atomic, no corrupt partial exports on crash); create_backup now holds the pipeline lock and calls MogDB compact_all() first so backups are consistent snapshots (.mogdb) with no torn journals (docstring/side effects updated); get_pipeline raises ValueError if called with a different data_dir instead of silently returning the wrong pipeline; scripts/demo_training_pipeline.py no longer hardcodes /Users/mac/... path (resolves REPO_ROOT, normal imports) and clears its /tmp store so re-runs are deterministic. Tests updated: backup tests assert .mogdb snapshots + restore round-trip; singleton test asserts rebind raises. Suite: 50 pipeline + 10 root + 54 dependent all pass; demo verified deterministic across runs.