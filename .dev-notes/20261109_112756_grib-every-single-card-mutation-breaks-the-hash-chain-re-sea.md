---
id: 20261109_112756_grib-every-single-card-mutation-breaks-the-hash-chain-re-sea
status: done
board: 20261109_112756_grib-every-single-card-mutation-breaks-the-hash-chain-re-sea
title: "grib: every single-card mutation breaks the hash chain — re-seal at the write funnels (CLI + GUI), verify canonical in producer"
tags: ["grib", "kanban", "chain"]
sprint: ""
gh: null
---

# grib: every single-card mutation breaks the hash chain — re-seal at the write funnels (CLI + GUI), verify canonical in producer

Survey 2026-11-09 (grib quality pass, chosen next target F1). EVERY single-card mutation leaves verify_chain() non-empty: add_card -> [-1] (new card unchained), update/move/block/unblock -> [0] (edited card), delete_card -> [1..n] (successors chain_index no longer matches file position), KanbanStore.move_card and KanbanStore._bk.save (the GUI write path, gui.py:318/423) -> [0]. Only grib sync re-seals (sync() ends with unconditional compute_chains(), store.py:981) and export re-seals only the REPO copy (dest_store.compute_chains()) — so canonical stays broken after a GUI drag until some session runs grib sync; check_board_chain.py defaults to the repo board (always freshly re-sealed) so CI never sees canonical drift. Live canonical was VALID only because the 2026-11-09 closes ran grib sync. Empirical harness: temp copy of ~/.config/kanban/board.jsonl + verify_chain(). Fix design: seal at the three write funnels — _surgical_rewrite (after matched>0; covers update/move/block/unblock/delete), _append_card (covers add), grib.kanban._JSONBackend.save (covers all KanbanStore writers incl. gui direct _bk.save) — each calls compute_chains(), which is OCC-safe (_reorder_cards token retry) and preserves foreign/unparsable lines. Board is 173 cards: seal cost trivial; the surgical one-line-diff design was for the 1100-card era. TDD: assert chain VALID after each of the 7 mutations (CLI+GUI); two existing tests need tamper-via-raw-write instead of API (test_verify_flags_mutation uses update_card, test_verify_flags_unchained uses add_card — the tamper simulation must bypass the sanctioned path, never weaken the assertion). Optional add: producer-side canonical verify_chain gate. Env note: planner 0.2.0 (library-only, no console scripts) reinstalled 2026-11-09 — grib GUI imports planner.engine (gui.py:362/370, kanban.py:345/726); grib+planner are both --user installs; never add planner to grib pyproject deps (PyPI name collision).
