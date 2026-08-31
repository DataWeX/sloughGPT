---
id: 20260826_070607_turbo-training-and-scroll-fixes-continuation
title: Turbo training and scroll fixes continuation
status: done
tags: area,training
created: 2026-08-26T07:06:07.625886+00:00
---

Turbo training and scroll fixes continuation

Added tests for all scroll-overflow fixes: TrainingLogCard (MAX_VISIBLE_LINES=500), TrainingHistoryView (paginate at 20), TrainingBuildsCard (paginate at 10), auto-train checkpoints (paginate at 10). All 286/291 pass, 5 pre-existing LossChart failures.