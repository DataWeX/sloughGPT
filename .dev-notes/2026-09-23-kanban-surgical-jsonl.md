# Kanban board surgical JSONL updates

Status: done
Sprint: 2026-09
Tags: kanban, board, app-planner, jsonl

## Why

`PlannerStore` rewrites the entire `board.jsonl` on every change (`add_card`,
`update_card`, `move_card`, `delete_card`, `archive_done`, sync). The board file
on disk is a mix of note-derived lines (`{"body","id","priority","status","title"}`)
and card-schema lines, so a full re-serialize re-keys every line → whole-file
git diffs (1300+ lines seen during worker churn).

## Plan

- Canonical emit: card lines via `json.dumps(..., sort_keys=True, ensure_ascii=False)`.
- Surgical single-line updates: patch only the touched card's line by id,
  preserve all other lines byte-for-byte, write atomically (temp + rename).
- add_card appends one line (+ schema header if missing), never touching
  existing lines.
- Keep `_write_board`/`save_board` (full write) for init/GUI whole-board save.

## Verification

- Sibling note-schema lines preserved byte-identical across update/move/delete.
- Sync only touches the matching card.
- Board diff stays ~1 line per operation.