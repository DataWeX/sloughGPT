---
title: Browser shell renders raw ANSI — SGR renderer (M1)
date: 2026-10-05
status: in-progress
branch: feat/soul-path-grammar
tags: [web, shell, ansi, terminal, frontend]
---

# Browser shell: raw ANSI leak → SGR renderer (M1)

## Problem

`/shell` → `/developer` Terminal tab rendered escape codes as literal garbage.
`TerminalPanel.tsx` did `{line.text}` raw, while `POST /shell/exec` returns a
string carrying SGR sequences.

## Investigation (evidence, not assumption)

Probed the live endpoint (`http://127.0.0.1:8000/shell/exec`) across `help`,
`logs`, `ls`, `status`, `history`, `nonexistentcmd`, `cat /nope`, `ps`:

- **HTTP exec path emits SGR only** — observed set: `[0m [1m [2m [33m [36m`.
- **No cursor motion, no alternate screen, no mouse tracking** on that path
  (`[1A [2K [2J [H [?1049 [?1003` → zero hits).

Those control sequences exist only in the curses TUI
(`domain/shell/_internal/{tui_repl,interactive,console,realm_view}.py`), which
the browser does not render — that surface is served by
`graphics.Framebuffer` (`list[list[Cell]]` + `diff()`), not this endpoint.

## Decision: **no xterm.js**

Initial recommendation was xterm.js; the evidence reversed it.

- xterm.js is a VT100 emulator (~300KB + `@xterm/addon-*`) built to decode
  cursor addressing, alt-screen buffers and mouse reporting — **none of which
  cross this wire**. Shipping it here means installing a dependency to handle
  sequences we never send.
- The repo already owns the hard parts for the path that *does* need them:
  `graphics.Framebuffer` with `Cell(char, fg, bg, attr)`, `Color`/`Attr` enums,
  `diff()`/`snapshot()`/`restore()`, plus `surface.py`'s `_ANSI_RE`,
  `strip_ansi()` and CJK `_display_width()`.
- AGENTS.md: "build from scratch, import last" + reuse the shared copy.

**xterm.js (or `Framebuffer.diff()` → JSON cells) becomes relevant only for
M2/M3** — putting the real TUI (alt-screen, panes) or an *uncontrolled* host
byte stream (vim/htop) in the browser. That is a different stream than
`/shell/exec`.

## Implementation

- **`apps/web/lib/ansi.ts`** (new) — tokenizer + SGR state machine.
  - Drops every non-SGR CSI/OSC/two-char escape, so garbage can never reach
    the DOM even if the backend changes.
  - Handles: 0/1/2/3/4/7, resets 22/23/24/27, fg 30-37 + 90-97 + 39,
    bg 40-47 + 100-107 + 49, `38;5;n`/`48;5;n` (only n<16 → tokens; the 6x6x6
    cube has no Noir Violet equivalent so it falls back rather than guessing
    hex), `38;2;...` truecolor ignored.
  - Maps palette → **design tokens** (`text-destructive`, `text-success`,
    `text-warning`, `text-info`, `text-accent`, `text-primary`,
    `text-muted-foreground`, `text-foreground`; bg → `bg-*/10`), never hex.
  - Adjacent same-class segments merge → plain output is always exactly one
    unstyled segment.
- **`TerminalPanel.tsx`** — single tokenize pass yields segments *and* visible
  text; the `Error`/`error` prefix check now runs on visible text (a leading
  escape used to defeat it). Unstyled runs render as `Fragment`, not `span`,
  so no extra DOM node is created.

## Bug caught during the build

First `parseAnsi` stamped **every** segment with the final (reset) style: the
`flush` closure read `style` at *call* time, and it only ran after the loop, by
which point style had been reset to default. All 13 styling tests failed while
all text-extraction tests passed. Fixed by making `flush(atStyle)` explicit and
flushing *before* adopting each new SGR.

## Regression caught by benchmarking (the mandatory step)

Rendering parsed every visible line on every render. Streaming re-renders on
**each appended line**, so a 1000-line stream was O(n²) parse work:

| stream length | re-parse each render | with `WeakMap` cache |
|---------------|----------------------|----------------------|
| 100           | 210 ms               | 4 ms  (49.7x)        |
| 500           | 2276 ms              | 32 ms (72.3x)        |
| 1000          | **12004 ms**         | **76 ms (156.9x)**   |

Per-render cost at the cap was **30.3 µs/line → 30 ms**, already past the 16 ms
frame budget *before* the stream finished. Fix: cache keyed on the `ShellLine`
object in a `WeakMap`. This is sound because `capLines` spreads the *same* line
objects into each new array, so line identity is stable across renders — and
verified there is exactly one construction site (`useShell.ts:80`) with no
in-place mutation. `WeakMap` keys are released when `capLines` drops a line, so
the cache cannot grow unbounded. Scaling flipped quadratic → linear.

## Verification

- Targeted: **47/47 pass** (`lib/ansi.test.ts` 25,
  `components/shell/TerminalPanel.test.tsx` 22 — 21 pre-existing stayed green,
  4 new ANSI integration tests added).
- Whole affected surface: **55/55 pass** (`developer/page`, `shell/page`,
  `components/shell/*` — 6 files).
- Full `apps/web` suite: **8238 passed / 1 failed** of 8239. The single failure
  (`app/(app)/session/page.test.tsx` — `Sessions (2)`) is **not mine**: that
  file has zero references to my changes, and it passes in isolation both with
  my changes stashed *and* with them applied. The full run took 1510 s with
  3071 s of environment setup across 843 files — a `{ timeout: 5000 }`
  assertion lost the race under parallel load.
- `eslint` clean (exit 0); `tsc --noEmit` clean (exit 0).
- Design-system scanner: 0 hex, 0 `rgb()` without `var()`, 0 stock-palette
  utilities, 0 inline styles, 0 `console.log`. The `px-*` hits are all
  pre-existing lines outside my edit region (165–201).
- **Live end-to-end** (parser fed real `POST /shell/exec` responses):
  `help` → 17 segments/`text-primary`; `logs` → 15 segments/
  `font-bold|opacity-60|text-warning`; `status` → 25 segments/`opacity-60`;
  `nonexistentcmd` → 1 segment. Zero escape leaks, visible text preserved
  byte-for-byte on all four.
- Escape samples in tests mirror live `/shell/exec` responses.

## Not done / deferred

- **Kanban board sync skipped** — `.kanban/board.jsonl` is dirty from a
  concurrent session on this branch; read-modify-write race. Same call as the
  mogdb card.
- **Production rebuild pending** — `dist-vite/` is from 03:40, so `:8080`
  serves stale code without this change. `:5173` (Vite dev) serves source and
  reflects it.
- **Visual check not done** — no desktop browser connected to this session.
  Verification is test + live-data based only.
- xterm.js **not installed** — no dependency was needed for M1 (see decision).
- M2 = real TUI in browser via `Framebuffer.diff()` JSON; M3 = parity.
- Pre-existing, unrelated: `packages/core-py/tests/test_shell_integration.py::
  TestLocalCommands::test_help_shows_builtins` fails at HEAD (`health` no
  longer a builtin after `_cmd_help` rewrite).
