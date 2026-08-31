---
id: 20260826_074642_training-flow-edge-cases-and-polish
title: Training flow edge cases and polish
status: done
tags: area,training
created: 2026-08-26T07:46:42.735863+00:00
---

Training flow edge cases and polish

Final round: fixed stale optimistic job on training start (auto-clear after 5s), fixed dismiss buttons clearing error state in FeedbackTrainCard and APILogsCard, added custom- prefix to preset keys preventing name collisions, added Notification API guard in training/page.tsx. 113/113 tests pass.