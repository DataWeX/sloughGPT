---
id: 20261104_060023_dataset-listing-structured-real-datasets-vs-system-dirs-hone
status: done
board: 20261104_060023_dataset-listing-structured-real-datasets-vs-system-dirs-hone
title: "Dataset listing structured — real datasets vs system dirs, honest sizes/sample counts"
tags: ["datasets", "listing", "ui"]
sprint: ""
gh: null
---

# Dataset listing structured — real datasets vs system dirs, honest sizes/sample counts

User 2026-11-04: 'some datasets don't show in this size' + 'make it more structured'. Measured defects in list_datasets (apps/api/server/controllers/datasets.py): (1) num_samples only counted when size < 1_000_000 — tinyshakespeare (32,777 records) and imported_dataset (18MB) report samples=0; (2) 24 kind=system dirs (auth_mogdb, errors_json, rag_store...) pollute the 37-row list — frontend isTrainingCorpus already hides them for training but the raw list shows them as datasets; (3) entries with no corpus (nbaallelo_csv has only an empty versions/, test_ds empty) show size=Empty as kind=dataset. Ruling: structured listing ONLY (user rejected model involvement). Fix: honest sample counts for large corpora, group/segregate kind=system+adapter+media from the dataset list, empty entries labeled not misclassified.
