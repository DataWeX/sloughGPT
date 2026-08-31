---
id: 20260808_033147_20260808-knowledge-enrichment-fix-regression-verification
title: 20260808 Knowledge-enrichment fix + regression verification
status: done
tags: 
created: 2026-08-08T03:31:47.389475+00:00
---

20260808 Knowledge-enrichment fix + regression verification

Fixed /chat greeting garbage: knowledge enrichment injected facts with no relevance gate. _is_casual_small_talk guard + MIN_RELEVANCE_SCORE=0.15 (min_score param) in knowledge_augmenter.py; verified live on :8000 (Hello! -> clean, France query -> correct). Also moved stray workspace-root data/ to /tmp/opencode backup. Regression sweep: training infra 393 pass, learner/knowledge/feedback 707 pass (1 skip/1 xpass), server targeted 125 pass. Full 286-file core-py suite too slow to run in one shot (interrupted); running targeted groups instead. No regressions found in touched areas.