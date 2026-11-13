---
id: 20261002_board_store_duplicate_card_ids
title: "Board store: duplicate card ids — 897 cards but 877 unique ids (dedupe, preserve note links)"
status: todo
date: 2026-10-02
tags: [kanban, data-integrity, store]
board_id: 89c58cfb-377d-434e-9352-5c4b69f705d7
board: 89c58cfb-377d-434e-9352-5c4b69f705d7
hash: 08fc2d871f1f60176a8d918cae593d17258c45e648cc3acd4832f4e90fb9f32a
---

# Board store: duplicate card ids — 897 cards but 877 unique ids (dedupe, preserve note links)

## 1. FINDING

FINDING (2026-10-02, during cards A/B done-flip): .kanban/board.jsonl contains 897
card lines but only 877 unique `id` values — 13 ids appear more than once (one appears
4x, five appear 3x, seven appear 2x; 33 lines carry a duplicated id = 20 surplus lines).

Observed distribution at creation time:
- x4: 8182563a (Logs unification, all todo, identical updated_at)
- x3: f74b9e75 (Loadable memory card), 2f7d91c4 (Structured palette),
  6109e26e (Magazine palette), 65ff978f (DESIGN_SYSTEM color tables),
  6a71e4db (tamagui.config palette)
- x2: 90839119 (Generic provider config), 44c63b91 (strui Checkbox+Radio),
  ec2ae5c1 (UI screenshot tests), 13aed9ad (Doc-vs-code gap audit),
  a25e490b (Tokens/sec investigation), 7e2af9fe (multi-provider web UI check),
  2e3f6ddc (Canonical local embedder)

Pre-existing: confirmed identical in a pre-edit snapshot taken before any 2026-10-02
edits, so it predates the cards A/B flip and the 2026-10-01 card batch. Discovered via
slot-chain apply run (verify ok: cards=897 ... slots=901).

## 2. WHY IT MATTERS

Any tool joining on card `id` (app_planner verify/sync, note<->card linking via
board_id, tray/slot tooling, dedupe in UI) will match the wrong twin or double-count;
note status <-> column checks may pass against one twin while another twin holds a
stale column; slot chain renumbering may operate on only one of the duplicates.

Concrete evidence gathered while creating this card:
- 8 of the 13 duplicated ids are byte-identical across twins (ignoring derived slot
  fields) -> safe drop of one after slot renumber.
- 5 differ in content. Of those, 4 disagree across twins on COLUMN:
  6109e26e, 65ff978f, 6a71e4db each span todo + in_progress + done; 2e3f6ddc spans
  in_progress + done; 2f7d91c4 spans all-done but content differs.
  These cannot be blindly dropped — the column disagreement is exactly the
  "status check passes against one twin while another holds a stale column" failure.
- One note already points at a duplicated id: 2026-09-30-card-44c63b91-strui-checkbox-radio.md
  -> board_id 44c63b91-363c-42fe-9a4d-eaa14de4e223 (appears x2). Dedupe must preserve
  this link (note resolution would otherwise silently detach).

## 3. BEFORE ANYTHING (file safety)

Back up .kanban/board.jsonl (cp to .kanban/board.jsonl.bak-<date>) AND confirm with the
user before deleting anything — board is derived+authoritative data with 1777 journal
nodes and 901 slots in play. This card = investigation first: identify the 13 repeated
ids, for each twin determine which line is canonical (newer updated_at? note that
resolves? both referenced by notes?), and report the dedupe plan (merge vs retire) —
DO NOT auto-delete.

## 4. INVESTIGATION STEPS

(a) list duplicated ids with both line numbers + column + updated_at + title of each twin.
(b) check whether any note frontmatter board_id/board points at a duplicated id, and
    whether the twins differ in content or are byte-identical (byte-identical = safe drop
    of one after slot renumber). — DONE for count/identity/notes; see FINDING + WHY IT MATTERS.
(c) determine root cause if feasible — e.g., append path writing without id-uniqueness
    guard (scripts/kanban_slot_chain.py apply, app_planner sync, or raw echo appends from
    concurrent sessions; the store also tolerated 3 unchained cards recently).
    Note: PlannerStore.add_card (store.py:283) mints id=str(uuid.uuid4()) with NO
    uniqueness check against existing ids — uuid4 collision is not the plausible cause;
    repeated appends of an already-present card (dup write from concurrent sessions or
    re-append of a restored/backup line) is.
(d) propose the fix: dedupe procedure + uniqueness guard (assert unique ids on write) +
    optional verify check `id unique` so this can't regress.

## 5. COORDINATION

Belongs to whoever owns the kanban store (grep dev-notes for kanban-sync/store cards;
note 2026-10-01-kanban-sync-hardening.md exists — link it; also line-reconciliation-plan
note). Concurrent sessions append to this file — coordinate through linked cards before
rewriting lines.

Two observed side notes from the 2026-10-02 flip: card b8c37898 notes[]/updated_at
changed mid-run by another session (benign concurrency), and 3 recently-appended cards
were unchained (74032dd0, 5c04bf42, 665042f9) — include chain-repair in the plan if
confirmed still unchained.
RECHECKED at card creation: all three ARE chained now (slot_position/slot_prev/
slot_hash/notes_hash/note_hashes all present, verify ok cards=897 journal_nodes=1777
slots=901) — an intervening apply repaired them, so no separate chain-repair work item
unless they regress.

SHARED WORKSPACE
- index updated: no index change required (store-internal fix — no new app/package/doc)
- shared copy reused: root .venv only, no installs
- coordination named: link notes 2026-10-01-kanban-sync-hardening.md and
  2026-10-01-line-reconciliation-plan.md + the b8c37898 card

## 6. FIX LANDED (2026-10-24, branch feat/kanban-dedupe-guard)

Implemented and proven end-to-end against the real board:

- `PlannerStore.dedupe_cards()` (store.py) — survivor = latest `updated_at`,
  first occurrence wins a strict-tie. Three passes: (1) pick survivor per id
  tracking file-order position; (2) delete surplus card lines RAW and
  byte-preserving, keyed by **line position among card lines** (not id, not
  an (id, updated_at) tuple — both would be wrong when twins collide on
  those keys); (3) `compute_chains()` reseals the hash chain over survivors
  via `_reorder_cards`.
- CRITICAL bug caught in review + fixed: the first draft called
  `save_board(board)` before `compute_chains`, and `save_board` →
  `_write_board` emits ONLY header + cards — it **destroyed the opaque
  slot_history/tray_roots trailer** (idx 956) that external tooling owns.
  Replaced with the raw position-keyed deletion pass, which never rewrites
  non-card lines. Verified byte-identical trailer after dedupe on a copy of
  the live board.
- `verify_board_integrity()` (verify.py) — read-only guard: flags any
  duplicate-id board as a `board-integrity` finding, **never raises** (a
  raising guard would crash whichever session next appends a card — the
  exact failure the plan warned about). Matches the true trailer by
  `slot_history` present AND `id` is None, so a card whose *title* mentions
  "slot_history" is not mistaken for one.
- Real-board proof (on a copy): 6 dup ids / 10 surplus lines → 0 dups after,
  chain reseals clean (`verify_chain()` returns []), both schema headers +
  the true trailer byte-identical.
- Tests: 15 new (test_dedupe_guard.py) — 270 app-planner tests green, ruff
  clean. Hot write path unaffected: benchmark_board_write.py green
  (688.8 adds/s, 8-process hammer integrity 200/8×25 ok) — dedupe is an
  offline repair path, verify guard is read-only.

Not yet applied to the live board (needs the quiet window + coordination
named in §5). Ready to run `store.dedupe_cards()` on `.kanban/board.jsonl`
once confirmed no concurrent writer.
