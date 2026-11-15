---
id: b3435a2f
title: "Which notes store is canonical: journal vs markdown (needs user decision)"
status: review
board_id: Main
date: 2026-10-16
tags: [decision, app-planner, notes, sync, docs-debt]
---

# Card b3435a2f — decision memo: journal vs markdown canonical store

## Verdict up front

The **code already decided**: the journal (`.dev-notes/notes.journal.jsonl`,
planner dialect) is the canonical store. The split-brain is **documentation
debt plus a hand-authoring habit**, not architecture debt. Recommendation:
**(A) finish the migration** — correct the docs, stop hand-authoring
markdown, keep markdown files as the read surface they already are.

## How it actually works today (mapped from code)

1. **One controller, two dialects.** `notes_controller.py` (`NotesController`)
   abstracts `core.NoteStore` — the **legacy markdown** dialect
   (`~/.config/dev-notes/*.md`, read+write) and `store.PlannerStore` — the
   **planner** dialect (journal jsonl). Fields a dialect cannot store fail
   loudly by design.
2. **The journal is the inferred default for this workspace.**
   `config.default_backend()`: env override, else infer from the notes dir —
   a journal jsonl in it selects the journal ("mogdb") backend. This repo
   has `.dev-notes/notes.journal.jsonl` (182 entries) → journal active.
3. **Sync reads only the journal** (`store.py`); the board is derived from
   it. Journal schema carries everything markdown does —
   `assignee, body, created_at, gh, id, sprint, status, tags, title,
   updated_at` — plus uuid identity; `app_planner note update --body`
   exists. No content capability is lost by canonicalising there.
4. **Markdown is still a live READ surface** — `domain/search`
   (`adapters/filesources.py`) indexes `~/.config/dev-notes/*.md` for
   search, and `slot_chain.py` hashes linked note frontmatter into card
   slot hashes. The files must stay; hand-authoring them is what must stop.

## The documented claim disagrees

AGENTS.md "Architecture": *"Notes (`~/.config/dev-notes/*.md`) are the
user's journal — source of truth for task metadata (sprint, gh, status,
body)"* — describes the **legacy dialect only** and never mentions the
journal or the dialect split. Agents following the doc write markdown files
that sync cannot see. Counts corroborate a ~complete migration with drift:
183 markdown files vs 182 journal entries, the markdown surplus being
hand-authored notes.

## Live failure evidence (this session, all three modes)

- **85e15cc4**: done-markdown + todo-journal twin → card stuck in todo; only
  `note update <journal-uuid> --status done` closed it (note status wins).
- **0d077078**: manual `board move → review` was **reverted by sync** (twin
  said todo); fixed via the journal twin, same mechanism.
- **af8ec05b / 66bf07ef / 6cba481c / 0d077078 memos**: hand-written markdown,
  entirely invisible to `list_notes` (journal-only, 182) — manual board
  moves are their only sync mechanism.

## Options

| | Cost | Consequence |
|---|---|---|
| **(A) journal canonical** (recommended) | one AGENTS.md correction + agent convention (notes via `note new/update`, never hand-written .md); optional small follow-up: `note import <file>` for stragglers | matches the code that already runs; markdown survives as search/hash surface |
| **(B) markdown canonical** | rewrite PlannerStore note backend to parse frontmatter; uuid↔short-id mapping; CLI + sync re-pointed | high risk, loses journal-only fields, buys nothing the user uses today |
| **(C) dual-write + reconcile** | conflict rules and repair paths forever | keeps the 85e15cc4 stuck-card failure mode alive whenever writes diverge |

## Recommendation: (A)

Finish the migration the backend inference already completed. (B) reverses
working code against the grain of the CLI; (C) institutionalises the exact
bug class this card was filed after. If the user still hand-writes markdown
sometimes, the one-shot `note import` (small follow-up card) covers it
without reopening the canonical question.

## Decision needed (user)

- **(A)** — I correct AGENTS.md's Architecture section (journal canonical,
  markdown = search/archive surface) and add the note-authoring convention;
  optionally file the `note import` follow-up, or
- **(B)** / **(C)** as costed above.
