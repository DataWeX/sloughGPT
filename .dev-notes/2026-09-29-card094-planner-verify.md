---
status: done
priority: medium
tags: [tooling, kanban, planner]
board: 8ba740dc-c80d-4f22-ba33-f8d6d560cb81
landed: 2c58146a2 + bbf46e7c6 (main/api-std), 73127ebbd + 637b72dcb (feat/logs-unification)
hash: 097a72f5ca918b2248f2d0bbdbcd2e7233c8fbad7b491ce848adae31d569437b
---

# Card 094 — `app-planner verify`: three-way diff (note ↔ board card ↔ landed shas)

## What

New CLI command + module `packages/app-planner/src/app_planner/verify.py`:

```bash
app-planner verify <note-path|title-substring> [--repo DIR] [--board-file F]
```

- **board**: note frontmatter `board:` uuid ↔ a real `board.jsonl` card
  (falls back to matching by the note's title when the key is absent).
- **status**: note `status:` ↔ card `column` via `config.STATUS_TO_COLUMN`.
- **git**: every `landed:` sha must **exist** in `--repo`; shas in
  `main*`/unlabeled groups must also be **ancestors of HEAD** (actually
  landed); `feat/*`-labeled groups are provenance only (squash landings
  never make branch shas ancestors).
- Exit `0` = consistent, `1` = drift (CI-gateable). Raw frontmatter parsing
  keeps `board:`/`landed:` keys that `core.Note.from_markdown` drops.

## Why

Notes are the source of truth, but nothing checked that a done note's card
column, its `board:` link, and its claimed `landed:` shas actually match
reality — drift was only discoverable by eye.

## Verification

- TDD: `packages/app-planner/tests/test_verify.py` — 11 tests RED→GREEN
  (mismatch/missing-uuid/non-ancestor/branch-provenance/CLI exit codes).
- app-planner suite: 166 passed; ruff clean.
- Real smoke on the card-092 note against `sloughgpt-api-std`: reports
  `all consistent` — 3 main shas landed, 3 branch shas provenance.

## Notes

- Commit on `feat/logs-unification` initially swallowed another session's
  199-file staged set (missing `-- <paths>` on commit) — recovered with
  `git reset --soft HEAD~1` + pathspec recommit; their staging verified
  byte-intact afterwards.
