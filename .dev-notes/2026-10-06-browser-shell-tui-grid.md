# Browser shell M2 — TUI cell grid (frontend half)

**Status:** frontend half complete and green; transport half deliberately not started.
**Depends on:** M1 (ANSI line renderer) — `2026-10-05-browser-shell-ansi-renderer.md`.
**Kanban:** board locked by concurrent session — this note is the source of truth
until `.kanban/board.jsonl` is writable again (same deferral as the mogdb card).

## Why M2 is a separate surface from M1

The shell reaches the browser two ways, and they are not the same wire:

| | Line mode | TUI mode |
| --- | --- | --- |
| Entry | `POST /shell/exec` | `TuiRepl.run()` (`tui_repl.py:186`) |
| Payload | flat `str` + SGR | `graphics.Framebuffer` cell grid |
| Browser half | M1 `lib/ansi.ts` | M2 `lib/screen-frame.ts` + `ScreenGrid` |
| Transport | exists | **does not exist** |

Confirmed live during M1: `/shell/exec` emits **SGR only** — no cursor motion, no
`?1049` alt-screen, no `?1003` mouse. Those escapes are written to `sys.stdout` by
`Renderer.open()` (`graphics.py:831-836`), never onto the HTTP wire. So M1's
"style parser, not an emulator" decision stands, and the TUI needs its own path.

## Contract — verified against Python, not assumed

Every fact below was executed against the pinned interpreter
(`/home/mana/miniconda3/envs/sloughgpt/bin/python`, `PYTHONNOUSERSITE=1`) before
any TypeScript was written.

1. **`Cell`** = `{char, fg, bg, attr}`; `fg`/`bg` are palette indices, `attr` is a
   bitmask (`Attr`: BOLD=1, DIM=2, ITALIC=4, UNDERLINE=8, BLINK=16, REVERSE=32,
   HIDDEN=64, STRIKETHROUGH=128).
2. **`Framebuffer.diff()` returns `self`'s cells.** Experiment:

   ```
   old.diff(new) -> [(1, 1, ' ')]   # stale value
   new.diff(old) -> [(1, 1, 'X')]   # the value a renderer wants
   ```

   The producer must therefore call `new.diff(old)`. Getting this backwards
   renders the previous screen.
3. **`Color.DEFAULT` is aliased to `Color.BRIGHT_RED` — both are 9.** The enum
   defines them with the same value, so Python aliases them. Usage decides the
   meaning: `Color.DEFAULT` is referenced **44 times**, `Color.BRIGHT_RED` **0
   times** (only ever defined, never used). Index 9 therefore means *unset*, and
   the renderer maps it to inherited rather than red. **Not fixed in the enum** —
   out of scope, and `BRIGHT_RED` being unreachable is a pre-existing defect worth
   its own card.
4. **The existing renderer already computes the right payload.**
   `Renderer._render_diff()` (`graphics.py:997`) does
   `changes = self._front_buffer.diff(self._back_buffer)` and iterates
   `(row, col, cell)` — exactly the `CellUpdate` shape. `render()` composites
   *into* `_front_buffer`, so `front` is the new frame and the direction is
   correct here. **The wire format already exists in Python; only the sink is
   missing** (it writes ANSI to `sys.stdout`).

## What landed (frontend half)

| File | Role |
| --- | --- |
| `apps/web/lib/terminal-palette.ts` | **new** — shared palette index → Noir Violet token map + `AttrBit` + `paletteClasses()` |
| `apps/web/lib/ansi.ts` | **refactored** — its duplicate 16-entry `FG_CLASS`/`BG_CLASS` removed; now flattens `Style` to an `AttrBit` mask and delegates to `paletteClasses()` |
| `apps/web/lib/screen-frame.ts` | **new** — `ScreenFrame`/`CellUpdate` wire types + `ScreenBuffer` (apply, seq ordering, resize, run grouping) |
| `apps/web/components/shell/ScreenGrid.tsx` | **new** — cell-grid renderer, exported from `components/shell/index.ts` |
| tests | `screen-frame.test.ts` (27), `ScreenGrid.test.tsx` (19), `ansi.test.ts` (25, unchanged) |

**Palette extraction, not duplication.** Both surfaces consume the same 0-15
indices, so mapping them in two places would guarantee eventual drift. One
module now owns the mapping; both import it.

### Two real bugs, both caught by tests

1. **2-D vs flat array.** `resize()` built
   `Array.from({length: rows}, () => blankRow(cols))` — an array whose *length*
   is `rows`, holding row-arrays — while `get`/`apply`/`runs` address cells flat
   as `row * cols + col`. Any index past `rows` read `undefined` and threw on
   `prev.char`. 25 tests failed; fixed by allocating
   `Array.from({length: rows * cols}, ...)`.
2. **`role="img"` hid the content.** The grid is text, not a picture of text.
   `img` marks children presentational, so a screen reader would announce
   "Shell screen" and none of the 1,920 cells. Changed to `role="group"` — label
   still exposed, cells still readable. Test asserts `role="group"` and
   `aria-hidden` absent.

### Performance

`runs()` rewritten: hoisted the loop-invariant `row * cols` (previously
recomputed per cell) and moved string building out of the per-cell path into a
per-run `joinChars()`.

Stress case — 1000 frames × 40 cell writes + 1000 × 24 row regenerations
(≈1.9M cell reads) on an 80×24 grid:

| | time |
| --- | --- |
| first implementation | 414 ms |
| after hoist + per-run join | **249 ms (1.66×)** |

Pinned as a test with a deliberately loose 1500 ms bound: it must fail on an
*algorithmic* change (the class of bug that made M1's first render 12,004 ms
instead of 76 ms), not on a loaded box.

**Caveat:** the box was running concurrent vitest sessions from another worktree
(`sloughgpt-chatui`) during both measurements, so absolute numbers are pessimistic;
the ratio is the signal.

### Verification

- targeted: 93 tests (27 + 19 + 25 + 22) green after every edit
- `eslint` exit 0, `prettier` clean, `tsc --noEmit` exit 0
- design-system scan: no hex/`rgb()`/`hsl()`, only semantic tokens; the
  `text-[12px] leading-[1.7]` ramp is byte-identical to `TerminalPanel.tsx:192`
- blast radius: **nothing imports `components/shell` barrel directly**;
  `lib/ansi` has exactly one consumer (`TerminalPanel.tsx:6`)

## Producer half (Python) — landed

| File | Role |
| --- | --- |
| `domain/shell/_internal/screen_wire.py` | **new** — `is_blank()`, `snapshot_frame()`, `diff_frame()` |
| `packages/core-py/tests/test_screen_wire.py` | **new** — 26 tests |

Pure, side-effect-free encoding: no endpoint, no transport, nothing the
endpoint rule covers. It exists so the contract is proven on *both* sides
before any route is drawn.

Design points worth keeping:

- **Public API only.** `snapshot_frame` diffs against a freshly allocated blank
  `Framebuffer` instead of reaching into `_cells`, so it can't break if the
  storage layout changes.
- **Blank elision, not omission.** The consumer resets its buffer *before*
  applying a `full` frame, so a blank cell is already correct and needs no
  bytes. An empty 24×80 screen costs **61 B** instead of ~126 KB. Blanks are
  defined by value (char *and* fg/bg/attr), so a space carrying a background —
  block cursors, selection bars — is still transmitted.
- **Resize falls back to a full frame.** `Framebuffer.diff()` intersects on
  `min(rows)`/`min(cols)`, so a diff across a resize would silently drop every
  cell outside the overlap and leave the browser showing a stale, wrongly-shaped
  screen. This is the same class of bug as the reversed-diff problem, so it gets
  its own guard.

Verification: `pytest` 26/26, `ruff check` clean, `py_compile` clean,
`tests/test_shared_indexes.py` 6/6.

### Measured cost (do not treat as precise)

| frame | cells | JSON | encode |
| --- | --- | --- | --- |
| full 24×80, solid fill | 1920 | 126 KB | 23.1 ms |
| 10-cell diff | 10 | 712 B | 10.4 ms |
| empty screen | 0 | 61 B | — |
| typical TUI screen (box + labels) | 98 | 6.9 KB | — |

**Read the diff row carefully: 10 cells cost nearly as much as 1920.** That is
structural, not noise — `Framebuffer.diff()` (`graphics.py:434`) walks the whole
grid on every call regardless of how many cells changed, so a frame's encoding
cost is O(rows×cols), independent of payload size.

**Caveat on every timing in this note:** measured while a concurrent vitest run
from `sloughgpt-chatui` held the box. Identical buffers clocked *slower* than
differing ones (8.96 ms vs 5.67 ms) — the expected ordering if the code were
doing what it appears to — and a `Cell.__eq__` micro-benchmark reported
8.3 µs/call, roughly 100× too slow for four integer compares. Absolute numbers
are unusable; only the structural claim (full-grid scan) is safe, and that comes
from reading the source.

**Next-feature candidate:** if the transport ever needs to hold a frame rate,
the fix is a dirty rect (the `Renderer` already tracks `self._dirty` at
`graphics.py:817`) rather than a faster scan — don't touch `graphics.py` to chase
these numbers.

## Wide characters — contract pinned, upstream bug found

`Framebuffer.write_str()` (`graphics.py:317`) handles a wide (CJK) glyph as
**two cells**: `Cell("你", ...)` followed by a continuation `Cell("", ...)` at
`graphics.py:337`.

Two consequences, both now pinned by tests on both sides
(`test_screen_wire.py::TestWideCharacters`, `screen-frame.test.ts` → *wide
characters*):

- The empty continuation is **not** blank by value, so it survives elision and
  is transmitted. It must be: it *claims a column*. Dropping it would let that
  column fall back to a space and render the row one column too wide.
- **Character count ≠ column count.** A 6-column row reading `ab你  ` holds only
  5 characters. Any future column arithmetic must not assume the
  "runs cover the full row width" invariant that holds for ASCII.

### Pre-existing bug in `write_str` (reported, NOT fixed)

```
write_str(0, 0, "hi你!") -> returns 5, fills cells: h i 你 !
                                                 0 1  2  3   <- only 4 cells
```

The column advances by *character index* (`c = col + i`, line 329) rather than
by **cumulative display width**, so a wide glyph's continuation cell lands at the
same index the next character writes to — and gets **overwritten**. Verified:

- `write_str(0, 0, "你X")` → cells `['你', 'X']` — `'你'` occupies visual
  columns 0–1, so `'X'` is drawn *underneath* it.
- The continuation survives only when the wide glyph is last in the string:
  `write_str(0, 2, "你")` then `write_str(0, 4, "Z")` → `['你', '', 'Z']`, correct.
- The return value (5) also disagrees with cells touched (4), so a caller doing
  `col += write_str(...)` leaves a spurious gap.

**Proposed fix** (needs approval — `graphics.py` is shared and untouched by me):

```python
c = col
for ch in text:
    ... write at c ...
    c += w          # advance by display width, not enumerate() index
```

plus returning `c - col`. Scope: **only** `write_str` — must not touch
`put`, `draw_box`, `composite`, `diff`, or the `Renderer`, and must not disturb
the existing 26 `test_graphics.py` cases or the new `test_screen_wire.py`.

Not blocking M2: ASCII output is unaffected, and the consumer already handles
the continuation cell when it does arrive.

## Not done — the transport (needs a card before code)

`ScreenGrid` is complete and tested but **not reachable from the app**: nothing
fills a `ScreenBuffer` yet. That is the honest state — building the renderer
first means the contract is proven on the consumer side before any endpoint
exists to satisfy it.

Blocking questions for the card:

1. **Where does the framebuffer come from?** `ShellREPL` (the singleton in
   `routers/shell.py:31`) has **no framebuffer attribute at all** — grep for
   `self.fb|self.screen|Framebuffer(` in `repl.py` returns nothing. `TuiRepl` is
   constructed ad hoc at `repl.py:2528` and `:5186`. There is no long-lived TUI
   session to read a screen *from*.
2. **Sink vs. core.** `Renderer._render_diff()` already holds `changes`; the
   minimal seam is a capture sink beside `sys.stdout.write` (sync, under the
   existing `self._lock`), **not** making the TUI async. Per AGENTS.md the core
   stays sync and the host crosses the boundary with
   `await asyncio.to_thread(...)` — exactly as `shell.exec` does at
   `routers/shell.py:104`.
3. **Must not disturb** `shell.exec` / `shell.exec_stream` line-mode semantics,
   and must not enter the chat/agent tool registry (external-utility tier).

Per AGENTS.md this endpoint is **discussed before code**. This note is that
discussion; awaiting the go-ahead.
