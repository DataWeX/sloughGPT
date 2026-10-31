---
id: 8aa055b8
title: "Shell engine phase 1: WS /shell/term + VT100 emulator + <TermScreen>"
status: doing
board_id: Main
date: 2026-10-16
branch: feat/web-terminal-phase1
tags: [shell, terminal, web, ws, coordination]
---

# Card 8aa055b8 — implemented on held branch feat/web-terminal-phase1 (landable)

## Pickup verification (2026-10-16, session .wt-gpuops)

This card's scope is ALREADY BUILT on the held branch
`feat/web-terminal-phase1` (session .wt-gpuops) — recorded here so no other
session starts a second implementation. Branch commits map 1:1 onto the
card's design (one VT100 state machine, two projections):

- `05f23c7cd` feat(shell): WS /shell/term — live TUI over cell diffs, off by default
- `14a743cb5` fix(shell): warm child imports before forkpty (server deadlock)
- `df220a80d` feat(web): render the shell cell grid in the browser (ScreenGrid)
- `86ee0141a` feat(web): connect the shell terminal to WS /shell/term (useShellTerm)
- `79a2de7c8` feat(web): key encoding + ShellTermPanel — the journey assembles
- `d70350b79` feat(web): host ShellTermPanel on the developer Shell tab
- `27735c71d` feat(web): fit the terminal to its viewport instead of pinning 24x80
- `b10209d90` test(shell): prove the endpoint relays resize to the session
- `a9a69c914` fix(shell): show the cause a failing curses teardown was hiding

Status: awaiting landing (main checked out in the api-std tree; landing
queue also holds fix/seam-holes-main, feat/credential-authority,
test/planner-delete-finality). NOTE: `feat/uxjourneys-rehome` was REMOVED
from the queue — superseded by the harness deletion on card 92af1aa0, must
not land. CAVEAT at landing time: the branch base predates
the curses-removal TUI port `37e4c1053` on main — expect conflicts in
domain/shell and the TUI widget modules; rebase or re-gate the shell/TUI
zone after merge.
