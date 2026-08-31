---
id: 20260801_084436_planner-notes-cli-test-suite-core-fixes-34-tests
title: planner notes CLI test suite — core fixes + 34 tests
status: done
tags: planner,tests,core
created: 2026-08-01T08:44:36.033863+00:00
---

planner notes CLI test suite — core fixes + 34 tests

Added packages/planner/tests/test_core.py (34 tests, file+mogdb backends). Fixed 3 real bugs surfaced: (1) get_note_store now caches by backend so sequential in-process CLI calls share store state (mogdb clients were isolated); (2) NoteStore coerces str notes_dir to Path (mogdb backend crashed on str); (3) Note.from_markdown stripped leading blank line so round-tripped body no longer gains a leading newline. Tests are CLI-driven (create/edit/verify via cli_main). Full planner suite: 96 passed.