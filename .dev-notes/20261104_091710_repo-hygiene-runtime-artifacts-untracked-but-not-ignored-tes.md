---
id: 20261104_091710_repo-hygiene-runtime-artifacts-untracked-but-not-ignored-tes
status: done
board: 20261104_091710_repo-hygiene-runtime-artifacts-untracked-but-not-ignored-tes
title: "repo hygiene: runtime artifacts untracked but not ignored — .testmondata-shm/-wal sidecars + config/flag_history.jsonl"
tags: ["tests", "flags"]
sprint: ""
gh: null
---

# repo hygiene: runtime artifacts untracked but not ignored — .testmondata-shm/-wal sidecars + config/flag_history.jsonl

Observed 2026-11-04: after any test run / API boot with load_config(), git status shows untracked .testmondata-shm, .testmondata-wal (SQLite WAL sidecars) and config/flag_history.jsonl (the flag audit ledger, written by FeatureFlags.load_config). Every session then re-discovers them and must avoid accidentally committing them.\n\nDecisions needed:\n- testmon sidecars: .gitignore them (transient; settle into .testmondata which is itself ignored? verify).\n- flag_history.jsonl: IGNORE vs COMMIT. It is an audit ledger — if the repo's stance is 'out-of-band flag changes are PR-visible history', committing it may be correct; if it is per-machine runtime state, ignore it. Pick one and write the reason in the .gitignore/comment.\nGates: git status clean after a test run + API boot.
