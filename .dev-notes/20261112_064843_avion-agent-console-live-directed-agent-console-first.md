---
id: 20261112_064843_avion-agent-console-live-directed-agent-console-first
status: todo
board: 20261112_064843_avion-agent-console-live-directed-agent-console-first
title: "Avion agent console — live-directed agent (console first)"
tags: ["avion", "agent", "console", "area:agents"]
sprint: ""
gh: null
---

# Avion agent console — live-directed agent (console first)

GOAL: Live-directed browser agent control surface. A console (own window in the
window system, four panes: log / work area / message / terminal) where the user
chats to a running agent mid-run: set_instructions, set_params, pause, resume,
inject_message, stop — applied at step boundaries via a command inbox.

HARD CONSTRAINTS (user-confirmed):
- No browser extension, no site modification, no overlay. External browser
  control only (Playwright/CDP) — the site sees a normal user.
- Console = its own window, never buried by the agent's browser windows
  (dedicated app window; --pin best-effort always-on-top via X11).
- Flow data (screens/flows/sites) lives in OUR JSON (site packs), never in
  their site. Twitter is one site pack, idle until user specifies collection.
- Order: CONSOLE FIRST, flow schema second, Twitter pack last.

PLAN (this card = slice 1, console first):
1. avion/ai/control.py — CommandInbox + typed commands.
2. Agent loop drains inbox between steps; config/instructions apply at
   step boundaries; events flow to existing EventJournal (single-writer).
3. `avion console` verb — self-contained HTML (precedent: avion view),
   SSE tail + POST commands, --window/--pin.
4. TDD with fake backend + EchoModel; ruff + avion suite gates.

OWNED PATHS (may touch): packages/avion/src/avion/ai/control.py (new),
packages/avion/src/avion/ai/agent.py, packages/avion/src/avion/cli.py,
packages/avion/src/avion/__init__.py, packages/avion/tests/test_control.py (new),
packages/avion/tests/test_console.py (new).
MUST NOT TOUCH: avion/ai/choice.py, avion/ai/slo.py (other sessions' uncommitted
hunks), any site's code, apps/web.

VERIFY: cd <worktree> && PYTHONNOUSERSITE=1 /home/mana/miniconda3/envs/sloughgpt/bin/python -m pytest packages/avion/tests/test_control.py packages/avion/tests/test_console.py packages/avion/tests/test_agent_loop.py -q -p no:randomly
+ /home/mana/miniconda3/envs/sloughgpt/bin/python -m ruff check <paths> && ruff format --check <paths>

BRANCH: feat/avion-agent-console (worktree from origin/main)
