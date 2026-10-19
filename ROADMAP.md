# SloughGPT Roadmap

**This file moved.** The canonical roadmap is **[`docs/ROADMAP.md`](docs/ROADMAP.md)**:

- direction and current state,
- **Now / Next / Later** tracks (Product, Platform, Model),
- the v1 release definition (route map first, decided 2026-10-19),
- the resolved numbered goals 1–20 with their evidence.

See [`docs/INDEX.md`](docs/INDEX.md) for the full document map.

**History:** this stub replaces the old root phase ledger (Phases 1–8, "How
to Pick the Next Task", the velocity table whose last row was 2026-07-13).
Nothing was deleted — the pre-restructure content is one `git log` away:

```
git log --oneline -1 -- ROADMAP.md        # find the pre-stub revision
git show <sha>:ROADMAP.md                 # read the old phase ledger
```

That ledger was retired rather than maintained: every phase was marked ✅, its
"next task" pointer aimed at the fully-done Phase 6, its velocity table had
frozen three months earlier, and its Phase 7 (Mobile) claimed ✅ while
`docs/ROADMAP.md` said "not started" — two files, one name, opposite claims.
One canonical roadmap now lives in `docs/`; execution state lives on the
kanban board (`.kanban/board.jsonl`).
