/**
 * TerminalEngine — line-oriented VT interpreter for the shell pane.
 *
 * Transport carries bytes verbatim; interpretation (CR overwrite, SGR styling,
 * cursor motion, scrollback) belongs here, at the view seam — the pane is a
 * terminal emulator, not an HTML line list.
 *
 * Input contract: each writeLine() receives ONE line event from
 * /shell/exec/stream — the server split on LF and stripped it — so the engine
 * applies the line end itself: content lands on the current row, the row is
 * committed, and the cursor moves to a fresh bottom row. Escape sequences are
 * never split across line events (no LF byte occurs inside CSI/OSC), so parser
 * state does not need to persist across calls.
 *
 * Styling maps SGR onto Noir Violet design tokens only — no hardcoded colors.
 * Truecolor/256-color specs (38/48;5;N / 38/48;2;R;G;B) are consumed but not
 * rendered (no token-safe mapping); text is never corrupted by them.
 */

export interface TerminalSpan {
  text: string
  className: string
}

export interface TerminalRow {
  /** Monotonic row id — stable React key, never reused after reset. */
  key: number
  /** Whole-row text (precomputed for render heuristics like the Error prefix). */
  text: string
  spans: TerminalSpan[]
}

export interface TerminalEngineOptions {
  /** Scrollback cap for snapshot() (default 1000). */
  maxLines?: number
}

interface Style {
  bold: boolean
  dim: boolean
  italic: boolean
  underline: boolean
  fg: string | null
}

interface Cell {
  ch: string
  cls: string
}

interface RowState {
  key: number
  cells: Cell[]
  committed: boolean
}

const DEFAULT_STYLE = (): Style => ({
  bold: false,
  dim: false,
  italic: false,
  underline: false,
  fg: null,
})

/** SGR foreground colors → design tokens (terminal palette only). */
const FG_TOKENS: Record<number, string> = {
  30: 'text-term-text-dim',
  31: 'text-term-close',
  32: 'text-term-zoom',
  33: 'text-term-minimize',
  34: 'text-term-blue',
  35: 'text-term-purple',
  36: 'text-term-cyan',
  37: 'text-term-text-strong',
}

function clsOf(style: Style): string {
  const parts: string[] = []
  if (style.fg) parts.push(style.fg)
  if (style.bold) parts.push('font-bold')
  if (style.dim) parts.push('opacity-60')
  if (style.italic) parts.push('italic')
  if (style.underline) parts.push('underline')
  return parts.join(' ')
}

function parseParams(paramStr: string): number[] {
  if (!paramStr) return []
  return paramStr.split(/[;:]/).map((p) => (p === '' ? 0 : parseInt(p, 10) || 0))
}

function compact(cells: Cell[]): TerminalSpan[] {
  const spans: TerminalSpan[] = []
  for (const cell of cells) {
    const last = spans[spans.length - 1]
    if (last && last.className === cell.cls) {
      last.text += cell.ch
    } else {
      spans.push({ text: cell.ch, className: cell.cls })
    }
  }
  return spans
}

export class TerminalEngine {
  private rows: RowState[] = [{ key: 1, cells: [], committed: false }]
  private nextKey = 2
  private cursorRow = 0
  private col = 0
  private style: Style = DEFAULT_STYLE()
  private readonly maxLines: number

  constructor(opts: TerminalEngineOptions = {}) {
    this.maxLines = Math.max(1, opts.maxLines ?? 1000)
  }

  /** Interpret one LF-terminated line of terminal output. */
  writeLine(text: string): void {
    let i = 0
    const n = text.length
    while (i < n) {
      const ch = text[i]
      if (ch === '\x1b') {
        i = this.parseEsc(text, i)
      } else if (ch === '\r') {
        this.col = 0
        i++
      } else if (ch === '\n') {
        // Server strips LF before emitting line events; tolerate one anyway.
        this.commitAndAdvance()
        i++
      } else if (ch === '\b') {
        this.col = Math.max(0, this.col - 1)
        i++
      } else if (ch === '\t') {
        this.col = (Math.floor(this.col / 8) + 1) * 8
        i++
      } else if (ch < ' ' || ch === '\x7f') {
        // remaining C0 controls (BEL, SO, SI, …) and DEL: no visual meaning
        i++
      } else {
        this.putChar(ch)
        i++
      }
    }
    this.commitAndAdvance()
  }

  /** Clear the screen: committed rows, cursor, style. */
  reset(): void {
    this.rows = [{ key: this.nextKey++, cells: [], committed: false }]
    this.cursorRow = 0
    this.col = 0
    this.style = DEFAULT_STYLE()
  }

  /** Rendered screen: committed rows only, capped at maxLines. */
  snapshot(): TerminalRow[] {
    const out: TerminalRow[] = []
    for (const row of this.rows) {
      if (!row.committed) continue
      const spans = compact(row.cells)
      out.push({
        key: row.key,
        text: spans.map((s) => s.text).join(''),
        spans,
      })
    }
    return out.slice(-this.maxLines)
  }

  // ── cursor helpers ────────────────────────────────────────────────────

  private curRow(): RowState {
    return this.rows[this.cursorRow]
  }

  private putChar(ch: string): void {
    const row = this.curRow()
    const cls = clsOf(this.style)
    while (row.cells.length < this.col) {
      row.cells.push({ ch: ' ', cls: '' })
    }
    if (this.col < row.cells.length) {
      row.cells[this.col] = { ch, cls }
    } else {
      row.cells.push({ ch, cls })
    }
    this.col++
  }

  /** End of a line event: commit the cursor row, cursor → fresh bottom row. */
  private commitAndAdvance(): void {
    this.curRow().committed = true
    if (this.cursorRow === this.rows.length - 1) {
      this.pushRow()
      this.cursorRow = this.rows.length - 1
    } else {
      // cursor was moved up (progress pattern): resume at the bottom row
      this.cursorRow = this.rows.length - 1
    }
    this.col = 0
  }

  private pushRow(): void {
    this.rows.push({ key: this.nextKey++, cells: [], committed: false })
    if (this.rows.length > this.maxLines + 2) {
      this.rows.shift()
      if (this.cursorRow > 0) this.cursorRow--
    }
  }

  // ── escape parsing ────────────────────────────────────────────────────

  /** Parse an escape sequence starting at ESC; returns the next index. */
  private parseEsc(text: string, i: number): number {
    const n = text.length
    if (i + 1 >= n) return n // dangling ESC at line end: consume
    const next = text[i + 1]
    if (next === '[') return this.parseCsi(text, i + 2)
    if (next === ']') return this.parseStringSeq(text, i + 2) // OSC
    if (next === 'P' || next === 'X' || next === '^' || next === '_') {
      return this.parseStringSeq(text, i + 2) // DCS / SOS / PM / APC
    }
    if (next >= ' ' && next <= '/') {
      // intermediate + final bytes, e.g. ESC ( B
      let j = i + 1
      while (j < n && text[j] >= ' ' && text[j] <= '/') j++
      if (j < n) j++
      return j
    }
    return i + 2 // two-char sequence (ESC 7, ESC =, …)
  }

  private parseCsi(text: string, start: number): number {
    const n = text.length
    let j = start
    let paramEnd = start
    while (j < n && text[j] >= '0' && text[j] <= '?') {
      j++
      paramEnd = j
    }
    while (j < n && text[j] >= ' ' && text[j] <= '/') j++
    if (j >= n) return n
    this.csi(text[j], text.slice(start, paramEnd))
    return j + 1
  }

  /** OSC / DCS / … string sequences: consume through BEL or ST (ESC \). */
  private parseStringSeq(text: string, start: number): number {
    const n = text.length
    for (let j = start; j < n; j++) {
      if (text[j] === '\x07') return j + 1
      if (text[j] === '\x1b' && text[j + 1] === '\\') return j + 2
    }
    return n
  }

  // ── CSI actions ───────────────────────────────────────────────────────

  private csi(final: string, paramStr: string): void {
    const nums = parseParams(paramStr)
    switch (final) {
      case 'm':
        this.applySgr(nums)
        break
      case 'A': // cursor up
        this.cursorRow = Math.max(0, this.cursorRow - (nums[0] || 1))
        break
      case 'B': // cursor down
        this.cursorRow = Math.min(this.rows.length - 1, this.cursorRow + (nums[0] || 1))
        break
      case 'C': // cursor forward
        this.col += nums[0] || 1
        break
      case 'D': // cursor back
        this.col = Math.max(0, this.col - (nums[0] || 1))
        break
      case 'E': // next line
        this.cursorRow = Math.min(this.rows.length - 1, this.cursorRow + (nums[0] || 1))
        this.col = 0
        break
      case 'F': // previous line
        this.cursorRow = Math.max(0, this.cursorRow - (nums[0] || 1))
        this.col = 0
        break
      case 'G': // cursor to column
        this.col = Math.max(0, (nums[0] || 1) - 1)
        break
      case 'H':
      case 'f': {
        // cursor position (1-based); pad rows with uncommitted blanks
        const target = Math.max(0, (nums[0] || 1) - 1)
        while (this.rows.length <= target) this.pushRow()
        this.cursorRow = Math.min(target, this.rows.length - 1)
        this.col = Math.max(0, (nums[1] || 1) - 1)
        break
      }
      case 'J':
        this.eraseDisplay(nums[0] ?? 0)
        break
      case 'K':
        this.eraseLine(nums[0] ?? 0)
        break
      default:
        // unsupported CSI (device modes, save/restore, …): consume silently
        break
    }
  }

  private applySgr(nums: number[]): void {
    const params = nums.length ? nums : [0]
    for (let i = 0; i < params.length; i++) {
      const p = params[i]
      if (p === 38 || p === 48) {
        // truecolor / 256-color spec: skip its arguments, render nothing
        if (params[i + 1] === 5) i += 2
        else if (params[i + 1] === 2) i += 4
        else i += 1
        continue
      }
      if (p === 0) this.style = DEFAULT_STYLE()
      else if (p === 1) this.style.bold = true
      else if (p === 2) this.style.dim = true
      else if (p === 3) this.style.italic = true
      else if (p === 4) this.style.underline = true
      else if (p === 22) {
        this.style.bold = false
        this.style.dim = false
      } else if (p === 23) this.style.italic = false
      else if (p === 24) this.style.underline = false
      else if (p >= 30 && p <= 37) this.style.fg = FG_TOKENS[p]
      else if (p >= 90 && p <= 97) {
        this.style.fg = FG_TOKENS[p - 60]
        this.style.bold = true
      } else if (p === 39) this.style.fg = null
      // 40–47 / 100–107 backgrounds and unknown params: consumed, no token
    }
  }

  private eraseDisplay(mode: number): void {
    if (mode === 2 || mode === 3) {
      // clear screen (watch -c): fresh cursor row, position home
      this.rows = [{ key: this.nextKey++, cells: [], committed: false }]
      this.cursorRow = 0
      this.col = 0
    } else if (mode === 1) {
      const row = this.curRow()
      const keep = row.cells.slice(this.col + 1)
      row.cells = Array.from({ length: Math.min(this.col + 1, row.cells.length) }, () => ({
        ch: ' ',
        cls: '',
      })).concat(keep)
    } else {
      // 0: cursor → end of screen
      const row = this.curRow()
      row.cells = row.cells.slice(0, this.col)
      this.rows.length = this.cursorRow + 1
    }
  }

  private eraseLine(mode: number): void {
    const row = this.curRow()
    if (mode === 1) {
      const head: Cell[] = Array.from({ length: this.col + 1 }, () => ({ ch: ' ', cls: '' }))
      row.cells = head.concat(row.cells.slice(this.col + 1))
    } else if (mode === 2) {
      row.cells = []
    } else {
      // 0: cursor → end of line
      row.cells = row.cells.slice(0, this.col)
    }
  }
}
