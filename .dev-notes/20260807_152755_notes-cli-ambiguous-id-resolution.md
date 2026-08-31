---
id: 20260807_152755_notes-cli-ambiguous-id-resolution
title: notes CLI ambiguous id resolution
status: done
tags: planner,cli
created: 2026-08-07T15:27:55.402394+00:00
---

notes CLI ambiguous id resolution

Fixed three notes/kanban CLI issues (packages/planner/).

1. Ambiguous id resolution (core.py NoteStore.get + kanban.py KanbanStore._find_one):
   - Root cause: find_by_prefix(); len(matches)>1 logged a warning and returned None -> any bare date-prefix (8-char short id shared by all notes/cards that day) resolved to 'not found'.
   - Fix: resolve ambiguous prefixes to the most recently updated match (sorted by updated_at/created_at/id desc), logging the full match count + chosen id.
   - show handler prints '(matched <full id>)' on stdout so the resolution is visible even with stderr logging off.
   - delete/rm stays strict (requires exactly 1 match) - destructive ops must not guess.
   - Found & fixed a mislabeled kanban test: 'get_card("alpha")' never matched anything (ids start with timestamps, not slugs) - rewritten to use the shared date prefix.

2. notes list --today (was documented in AGENTS.md but rejected: 'unrecognized arguments: --today'):
   - Added --today flag to the list subcommand; list_notes() gained today: bool param filtering date_str == date.today().
   - The standalone 'today' subcommand remains for the same purpose.

Tests (+4, parameterized over file/mogdb backends where applicable): test_show_ambiguous_prefix_resolves_to_most_recent, test_edit_ambiguous_prefix_updates_most_recent, test_delete_ambiguous_prefix_refuses, test_list_today_flag_filters_to_today, plus kanban test_find_one_ambiguous_prefix_resolves_to_most_recent. Full planner suite 138 passed.

Verified live: 'notes show 20260806' resolves to 20260806_143527_session-core-py-test-sweep-datasetspage-flake-fix; 'notes list --today' lists today's 2 notes.