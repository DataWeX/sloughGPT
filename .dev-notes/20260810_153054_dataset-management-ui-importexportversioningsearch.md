---
id: 20260810_153054_dataset-management-ui-importexportversioningsearch
title: Dataset management UI — import/export/versioning/search
status: done
tags: web,datasets
created: 2026-08-10T15:30:54.202944+00:00
---

Dataset management UI — import/export/versioning/search

Dataset management UI: wired convert-to-chat-format into dataset detail page (new card with optional system prompt, success banner + Open converted dataset link, error toast). Added 5 backend tests for the convert endpoint (text wrapping, custom system prompt, existing system kept, 404s) and 4 frontend tests for the new card. Verified: tsc 0 errors, 103 dataset frontend tests pass, 68 backend tests pass. ROADMAP item #6 marked done.