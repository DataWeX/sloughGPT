---
id: 20260829_145531_update-sdk-test-files-to-match-new-training-urls
title: Update SDK test files to match new training URLs
status: done
tags: area,subarea
created: 2026-08-29T14:55:31.020578+00:00
---

Update SDK test files to match new training URLs

Fixed 3 CLI URL mismatches in train.py: GET /training -> GET /training/jobs, /training/from-sessions/start -> /training/from-sessions-start, /training/from-sessions/stream -> /training/from-sessions-stream. Updated docstrings. All 270 CLI tests pass.