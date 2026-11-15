---
id: 20261113_064113_grib-gui-standalone-writes-the-md-store-while-clisync-read-t
status: todo
board: 20261113_064113_grib-gui-standalone-writes-the-md-store-while-clisync-read-t
title: "grib gui standalone writes the md store while CLI/sync read the journal — same-command, two stores (docstring claims otherwise)"
tags: ["area:tooling"]
sprint: ""
gh: null
---

# grib gui standalone writes the md store while CLI/sync read the journal — same-command, two stores (docstring claims otherwise)

Evidence (session 2026-11-13): grib/gui.py:483 builds core.NoteStore(backend=default_backend()='file') → _FileBackend globs ~/.config/dev-notes/*.md (248 files); the CLI path (_note_ctl → get_store) is PlannerStore → notes.journal.jsonl (63 entries); gui.py docstring claims 'reads and writes the same data as the notes and kanban CLI commands' — FALSE for the default backend. Consequence: a GUI drag updates the md note only; the journal stays stale; the next `flock … grib sync` restores card.column from the journal → GUI moves revert for the ~64 journal-bound cards (same defect class as card 20261112_063528, other store — that card fixed journal-side writes only, 1060548c6). Fix options: (1) route GUI note ops through NotesController(PlannerStore)/journal so there is ONE store (preferred — matches 'CLI = journal' reality); (2) minimal: correct the docstring + --backend help to state file=md vs journal split. If the server-mount card lands first, standalone `grib gui` is the only remaining md writer — this card still applies to it. TDD RED first; grib suite 376 baseline.
