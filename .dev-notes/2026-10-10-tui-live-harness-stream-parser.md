---
title: "TUI live tests: harness stream fix, curses removal, render erase bug"
date: 2026-10-10
status: done
branch: fix/tui-live-harness-stream-parser
tags: [tui, curses, tests, harness, terminal, graphics, rendering]
---

# Outcome

`packages/core-py/tests/test_shell_tui_live.py` was **28 failed / 14 passed**.
It is now **42 passed / 0 failed**.

| Stage | Result | Commit |
|---|---|---|
| Harness stream-parser fix | 28 → 25 failed | `6b4d7f83f` |
| Production fix (sole screen owner + status contract + test encoding) | 25 → **0 failed** | `4222463a9` |

Both commits are on `fix/tui-live-harness-stream-parser`. Gates quoted in the
commit message: ruff clean, mypy 25 → 19 errors with none new, and
`scripts/benchmark_graphics.py` shows no regression (warm render −22%,
write+render −48%).

The design decision this card was blocked on was taken: **Option 1 — the
engine owns the terminal.** Curses is gone from `tui_repl.py` entirely, so
there is a single writer of the glass and the diff is permanently valid.

# Part 1 — harness defects (fixed, landed here)

`_Screen.feed()` treated each `os.read()` chunk as self-delimiting. A pty
stream is not frame-delimited. Three real defects, proven against `_Screen`
alone before touching production code:

| Probe | Before | After |
|---|---|---|
| `\x1b[?2` + `5h` in two feeds | `'» alph5hX'` | `'» alphX'` |
| `λ` split after `ce` | `''` (U+FFFD) | `'λ'` |
| `ab\x1b` + `[5D` | col 5 (lost) | col 0 |

1. **Split CSI** — `if final is None: break` dropped the truncated sequence and
   re-parsed the tail as literal text.
2. **Split UTF-8** — `decode("utf-8", "replace")` turned a split character into
   U+FFFD (`λ` is `ce bb 20`).
3. **Lone ESC at a chunk boundary** — dropped, losing the cursor move after it.

Plus a **pessimistic fixed settle**: `write()`/`keys()` slept a flat 0.15 s /
0.2 s, so under load assertions read a frame the TUI had not redrawn. Now
`settle()` waits for the **screen fingerprint to stop changing** (fingerprint,
not byte-quiet: the TUI repaints on a 100 ms `getch` tick, so bytes never stop
flowing). 18 whole-chunk controls verified byte-identical to the pre-fix parser.

**These were real, but they were not the cause of the 28 failures.** My
original attribution on this card was wrong and is corrected here.

# Part 2 — the real root cause (landed in `4222463a9`)

`GraphicsEngine.render()` (`domain/shell/_internal/graphics.py:975`) comments
say *"Diff against previous frame"*, but it diffs `_front_buffer` against
`_back_buffer`, which is the **base surface** and is never updated from the
front buffer (`open()` assigns it at :828; the only other writes are the
undo/redo paths at :1098/:1114). Consequences:

1. **No erase is ever emitted when a layer shrinks.** `_render_input` clears
   the input layer and repaints only the prompt; the vacated cells are blank in
   both buffers, so `diff()` sees no change and the terminal keeps the old
   glyphs.
2. **Every tick re-sends the whole screen** (~37 KB/tick; 24x80 cells).

`Framebuffer.composite()` (`:431`) additionally skips cells that are a space
with default bg, so a cleared layer cannot overwrite what it painted. Note the
two bugs **do not cancel** — neither alone reproduces nor fixes the symptom.

## Key handling is correct

Proved by the "typed X after Ctrl+U" fork against a live session:

| Step | Screen | Meaning |
|---|---|---|
| `alpha` | `'λ alpha'` | fine |
| `Ctrl+U` | `'λ alpha'` | buffer emptied, stale glyphs remain |
| `X` | **`'λ Xlpha'`** | X landed at col 3 = the emptied buffer's cursor |
| `bcd` | `'λ Xbcda'` | buffer `Xbcd`, stale `a` at col 7 |
| Backspace x2 | `'λ Xbcda'` | buffer `Xbcd`→`Xbc`→`Xb`; screen shows `λ Xb` + stale `cda` |

So `Ctrl+U`, `Ctrl+K` and Backspace all mutate `_input_buf` correctly. The bug
is purely that the render never erases vacated cells. All 25 remaining failures
are this one bug.

## Reproduction (drives the real engine, stdout captured into `_Screen`)

| Frame | Before fix | After `_prev_buffer` fix |
|---|---|---|
| 1 typed `alpha` | `λ alpha` | `λ alpha` |
| 2 `Ctrl+U` | `λ alpha` (stale) | `λ` |
| 3 static | 18 bytes re-sent | **0 bytes** |
| 4 typed `X` | **`λ Xlpha`** | `λ X` |

Fix verified in isolation, and `test_graphics.py` stayed green (80 passed).

## Why `_prev_buffer` alone regressed the suite (28 → 41 failed)

With `_prev_buffer` in place but curses still drawing, the live suite
regressed. A/B on two tests: 2 passed in 3.78 s without it; the same 2 failed
in 21.27 s with it; child stderr empty — no crash.

Cause, from byte-level A/B of the real stream (identical `ESC[2J` at offset
6376 in **both** versions, so not introduced by the fix):

- `stdscr.erase()` (`tui_repl.py:1320` at boot, `:1290` on resize) followed by
  `stdscr.refresh()` (`:1718`) makes **ncurses** emit `\x1b[2J`, wiping the
  frame the engine had just painted.
- The old diff-vs-blank engine repaired this *by accident* — it repainted the
  whole screen every tick.
- A correct diff engine stays silent (prev == front), so the screen stays blank
  forever: `screen non-blank cells: 0`.

So two independent renderers write to one terminal, and the engine's damage
tracking is only ever valid if it is the sole writer.

## Decision taken — Option 1, the engine owns the terminal

Option 1 was chosen and implemented in `4222463a9`; Option 2 (periodic
resync) was tried and **deleted** — it was a band-aid, proven unnecessary
once curses was gone (`clears` = 2 → **1**, a one-time boot event).

`domain/shell/_internal/tui_repl.py` is now curses-free end to end:

- `import curses` removed; the colour-pair layer (`_P_*`, `_STYLE_PAIRS`,
  `_init_pairs`) proved dead and was removed with it.
- `curses.newwin` ×8 → `_Win`, a plain geometry record.
- `curses.wrapper` → `_RawTerminal`, a context manager owning raw mode and
  its restore on every exit path.
- `_read_escape_remainder` now decodes CSI/SS3 itself, folding into local
  `KEY_*` constants that carry **ncurses' own `<curses.h>` numbers**
  (KEY_UP 259, KEY_DOWN 258, KEY_LEFT 260, KEY_RIGHT 261, KEY_HOME 262,
  KEY_BACKSPACE 263, KEY_DC 330, KEY_NPAGE 338, KEY_PPAGE 339, KEY_END 360,
  KEY_RESIZE 410, CTRL-LEFT 554, CTRL-RIGHT 569), so the dispatch table is a
  mechanical rename. int-form and `seq:*` string-form branches were verified
  to do identical work before folding.
- `_detect_resize` no longer takes `stdscr` and no longer calls
  `curses.resizeterm`; `stdscr.erase()`/`refresh()`/`move()` are gone.

**One non-obvious invariant that had to be preserved: ICRNL.** `tty.setraw()`
clears it, so the harness's `\r` (13) arrives as 13 while every submit branch
keys on `ch == ord("\n")` (10) — the shell could not run anything. Fixed in
the *reader* (`_RawTerminal.__enter__` re-ORs `termios.ICRNL`), not by editing
the dispatch, keeping the input contract byte-identical. ISIG stays off so
Ctrl+C is a character.

Measured effect of the fix: steady terminal traffic **~370 KB/s → ~52 B/s**
(full repaint per tick → damage-only), non-blank cells steady at 1539,
`clears` = 1, input row correctly `λ` after submit.

# Part 3 — two further root causes found after the render fix

Fixing the render defect exposed two more, both real:

1. **Status-bar contract lost in a mislabeled commit.** `_render_status`
   printed `OUTPUT` in *both* branches — `[LOG]` was unreachable — and emitted
   no `SCROLL` word at all. `git log -S` shows `68c427af5` added
   `f"SCROLL ↑{scroll}"` and `9f348d0c9`, whose message claims *"rename
   domain.cognitive → domain.cognition in source files"*, actually ported 459
   lines of the status bar from curses to GraphicsEngine and dropped both the
   `[OUTPUT]/[LOG]` switch and the `SCROLL` word. Restored, with the head
   width driving the columns that follow instead of the hardcoded `col = 7`.
   This alone accounts for 5 tests that failed **identically on HEAD and on
   the fixed build** (PgUp never reported scroll, Ctrl+O never reported log
   focus) — the clearest evidence that they were separate from the render bug.

2. **Test-side stale-artifact encoding.** Pane backgrounds are painted with
   `░` (`graphics.py:181`, `Pattern.SHADE_LIGHT`), so a command echoed into
   the output pane reads `λ░echo░alpha`. Assertions written as
   `"λ echo alpha" in text()` could therefore only ever match the *input*
   row — which the shell correctly clears on submit. They passed on HEAD only
   because erases were missing and stale text lingered there. Added
   `_Screen.flat_text()`/`flat_row()` (fill normalised to spaces) and switched
   the affected assertions to it; no assertion was weakened, and the input-row
   (`row(ROWS-1)`) and status-bar assertions were left alone because they
   never contained the fill glyph.

# Environment notes

- Live suite is slow and load-sensitive (311–419 s); this box ran at load
  12–30 and 96–97 °C. A/B runs must quote load or the comparison is invalid.
- Canonical pin: conda `sloughgpt`, `PYTHONNOUSERSITE=1`, full six-entry
  `PYTHONPATH`, `pytest -c pytest.ini <file> -p no:randomly --timeout=60`.
