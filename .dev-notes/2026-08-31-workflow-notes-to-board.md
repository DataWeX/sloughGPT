---
title: Notes-to-Board Workflow
date: 2026-08-31
tags: [workflow, planner, kanban, notes]
status: open
---

# Notes-to-Board Workflow

## Philosophy

- **Notes** = scratch pad for in-progress work (markdown files in `~/.config/dev-notes/`)
- **Board** = finished/committed work (JSONL in `.kanban/board.jsonl`)
- **Sync** bridges them when work is complete

## Flow

1. Create a note when starting work (markdown, editable, working state)
2. Do the work (note is the live document)
3. When done, sync transfers the note to the board card (JSONL, permanent record)

## Rules

- Notes are for todos **not yet done**
- Board is for **done** work
- Sync runs: note status changes to `done` → card created/moved to `done` column
- Note body, tags, and metadata transfer to the card
- After sync, the note is archived (status `done`), the board card is the source of truth

## Why

- Markdown is easy to edit, diff, version
- JSONL is structured, queryable, displayable
- Don't pollute the board with half-finished work
- Board stays clean as a record of what's been accomplished
