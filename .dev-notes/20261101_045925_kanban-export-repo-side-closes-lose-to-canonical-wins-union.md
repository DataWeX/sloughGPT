---
id: 20261101_045925_kanban-export-repo-side-closes-lose-to-canonical-wins-union
status: done
board: 20261101_045925_kanban-export-repo-side-closes-lose-to-canonical-wins-union
title: "kanban export: repo-side closes lose to canonical-wins union — done-monotonic merge for board columns"
tags: ["kanban", "pipeline", "export", "clobber"]
sprint: "kanban-sync"
gh: null
---

# kanban export: repo-side closes lose to canonical-wins union — done-monotonic merge for board columns

Found 2026-10-31: card 20261031_081502 was closed repo-side (jolly-wizard commit 0566d9f7f) with NO canonical note; subsequent canonical->repo exports (canonical wins on shared id) silently reverted column done -> todo. Fixed the instance by creating the canonical note (status done). Root cause: union_board() gives shared ids wholly to canonical, so repo-side closes cannot survive without a canonical note. Options: (a) done is monotonic in the union (either side done => done; reopen must clear BOTH sides), or (b) close always creates/updates the canonical note (process+CLI guard). Pick one in card, add a regression test next to the 4 union tests.
