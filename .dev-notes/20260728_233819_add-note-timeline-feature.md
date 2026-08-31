---
id: 20260728_233819_add-note-timeline-feature
title: Add note timeline feature
status: done
tags: notes,timeline
created: 2026-07-28T23:38:19.261390+00:00
---

Add note timeline feature


Added NoteStore.timeline(days, tag, status) returning (date, notes) groups. Added CLI subcommand notes timeline --days/--tag/--status. Added REPL note timeline dispatch with tab completion. Fixed indentation bug in status CLI handler. 6 new timeline tests. All 40 tests pass.