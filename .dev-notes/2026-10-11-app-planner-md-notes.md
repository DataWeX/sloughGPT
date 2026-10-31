---
title: "app-planner could not read the notes AGENTS.md calls the source of truth"
date: 2026-10-11
status: done
branch: fix/app-planner-md-notes
landed: 59b15ef36
tags: [planner, kanban, notes, sync, markdown, data-integrity]
---

# Outcome

`AGENTS.md` says notes live at `~/.config/dev-notes/*.md` and that the board is
derived from them by `sync`. `PlannerStore` read `<notes_dir>/notes.journal.jsonl`
and nothing else — so the documented store was invisible, **even when you pointed
`--notes-dir` at it**: the directory is accepted, `list_notes()` returns empty,
`sync()` adds no cards. Silent.

Landed as **`59b15ef36`** on `fix/app-planner-md-notes`. Card `59b7a45f`.

# What the card got right, and what it got wrong

The card said *"app-planner reads notes.journal.jsonl, so sync is a silent no-op
for the 169 md notes."* Directionally right, mechanism wrong — and the difference
changed the fix.

`PlannerStore._notes_dir` resolves to **`<root>/.dev-notes`**, which holds a
journal of 122 notes. Sync works fine *there*. It was never a no-op in general;
app-planner was simply pointed at a **different directory** than the one the
docs designate.

Measured, the two stores are **disjoint**, not duplicated:

| | journal (`<root>/.dev-notes`) | markdown (`~/.config/dev-notes`) |
|---|---|---|
| notes | 122 | 170 |
| id form | uuid | date-stem |
| **id overlap** | | **0** |
| **title overlap** | | **4** |
| statuses | done 66, review 21, todo 14, wip 11 | done 98, todo 15, doing 12 |

So this is not consolidation. Two separate note sets, ~292 notes, and only the
second one is documented.

# The trap that decided the design

`update_note()` and `delete_note()` do `notes = self._read_notes()` then
`self._write_notes(notes)`. The obvious fix — make `_read_notes()` also read
markdown — would have **copied all 170 `.md` notes into the journal on the next
write** and silently merged two stores that were deliberately separate. Caught
by reading the write paths before touching the read path; pinned afterwards by
`test_a_markdown_note_is_never_folded_into_the_journal`.

# What was done

- **Read-only markdown source** (`_read_markdown_notes`), merged by
  `_read_all_notes()` into `list_notes`/`get_note`/`search_notes` — and therefore
  into `sync()`, which reads via `list_notes()`.
- **`_read_notes` untouched.** The three write round-trips still see only the
  journal, so the stores can never merge by accident.
- **No second parser.** `core.Note.from_markdown` already parsed frontmatter;
  it is a *different* dataclass from `store.Note` (it carries `author`), so the
  parser is reused and only the carrier adapted.
- **Tag parse fixed** — it is the path this change activates. `from_markdown`
  split on commas alone, so the dominant real format `tags: [a, b, c]` produced
  `["[a", "b", "c]"]`, and those bracket artefacts travelled into board card
  tags on the next sync. Bare `tags: a,b` still parses.

# Proven, not asserted

- **Default store unchanged:** `<root>/.dev-notes` has no `*.md`; `sync()` on a
  copy of the real 1117-card board reported `added=0, updated=0`. Byte-identical.
- **Opt-in works:** the same run pointed at `~/.config/dev-notes` found all 170
  notes and would add **151** cards. **Not applied.**

# Left open on purpose

Reconciling the two stores is a **migration, not a fix** — 292 notes, and 151 of
them would become new cards (70 already `done`) flooding a shared queue that 83
sessions read. That needs the user's decision, so it is a separate card rather
than a side effect of a read-path change.

# Gates

`12 passed` (9 new + the atomic-write guard) · `packages/app-planner` own suite
**206 passed** · `tests/contract` 9 passed · `ruff check` + `format --check`
clean · mypy `0 -> 0`.

# Note for whoever reads this next

The write round-trip is the thing to check before extending any read path. A
store that reads a superset and writes the whole list back will duplicate what
it newly learned, one write later, with no error.
