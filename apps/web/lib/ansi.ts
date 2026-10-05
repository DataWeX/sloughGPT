/**
 * Minimal SGR (colour/style) renderer for Dait shell output.
 *
 * ## Why this is a style parser and not a terminal emulator
 *
 * The browser shell talks to `POST /shell/exec`, which runs
 * `ShellREPL.execute()` and returns a plain string. Probed live against
 * `help`, `logs`, `ls`, `status`, `history` and the error paths: that boundary
 * carries **SGR sequences only** (reset, bold, dim, fg colours). Cursor motion
 * (`ESC[1A`, `ESC[2K`, `ESC[K`), screen control (`ESC[2J ESC[H`) and alternate
 * screen (`ESC[?1049`) are emitted only by the curses TUI (`tui_repl.py` /
 * `interactive.py`), which the browser does not render -- that surface is
 * served by `graphics.Framebuffer`, not by this endpoint.
 *
 * So a full VT100 emulator would be dead weight here: it would handle cursor
 * addressing, alternate-screen buffers and mouse reporting that never cross
 * this wire. Anything that is *not* an SGR sequence is dropped instead of
 * rendered, so raw escape garbage can never reach the DOM.
 *
 * Colours are mapped onto Noir Violet semantic tokens rather than hardcoded
 * hex, per the design-system rule.
 */

export interface AnsiSegment {
  /** Visible text for this run. Never contains escape characters. */
  text: string
  /** Design-system class names; empty string means "inherit the default". */
  className: string
}

interface Style {
  fg: number | null
  bg: number | null
  bold: boolean
  dim: boolean
  italic: boolean
  underline: boolean
  inverse: boolean
}

const DEFAULT_STYLE: Style = {
  fg: null,
  bg: null,
  bold: false,
  dim: false,
  italic: false,
  underline: false,
  inverse: false,
}

// ANSI palette indices 0-15 -> semantic tokens. Values are complete literal
// strings so the Tailwind scanner picks them up.
const FG_CLASS: Record<number, string> = {
  0: 'text-muted-foreground', // black
  1: 'text-destructive', // red
  2: 'text-success', // green
  3: 'text-warning', // yellow
  4: 'text-info', // blue
  5: 'text-accent', // magenta
  6: 'text-primary', // cyan
  7: 'text-foreground', // white
  8: 'text-muted-foreground', // bright black
  9: 'text-destructive', // bright red
  10: 'text-success', // bright green
  11: 'text-warning', // bright yellow
  12: 'text-info', // bright blue
  13: 'text-accent', // bright magenta
  14: 'text-primary', // bright cyan
  15: 'text-foreground', // bright white
}

const BG_CLASS: Record<number, string> = {
  0: 'bg-muted',
  1: 'bg-destructive/10',
  2: 'bg-success/10',
  3: 'bg-warning/10',
  4: 'bg-info/10',
  5: 'bg-accent/10',
  6: 'bg-primary/10',
  7: 'bg-background',
  8: 'bg-muted',
  9: 'bg-destructive/10',
  10: 'bg-success/10',
  11: 'bg-warning/10',
  12: 'bg-info/10',
  13: 'bg-accent/10',
  14: 'bg-primary/10',
  15: 'bg-background',
}

type Token = { kind: 'text'; value: string } | { kind: 'csi'; params: string; final: string }

/**
 * Split input into literal text runs and CSI escape sequences.
 *
 * Non-CSI escapes (OSC, two-character escapes) are swallowed -- they are never
 * displayable content, and emitting them would show raw ESC glyphs.
 */
function tokenize(input: string): Token[] {
  const out: Token[] = []
  let buf = ''
  let i = 0

  while (i < input.length) {
    const ch = input[i]
    if (ch !== '\u001b') {
      buf += ch
      i += 1
      continue
    }
    if (buf) {
      out.push({ kind: 'text', value: buf })
      buf = ''
    }

    const next = input[i + 1]
    if (next === '[') {
      // CSI: parameter bytes 0x30-0x3f, intermediate 0x20-0x2f, final 0x40-0x7e
      let j = i + 2
      while (j < input.length && input[j] < '@') j += 1
      if (j >= input.length) break // truncated -- drop the dangling escape
      out.push({ kind: 'csi', params: input.slice(i + 2, j), final: input[j] })
      i = j + 1
    } else if (next === ']') {
      // OSC: terminated by BEL or ST (ESC \)
      let j = i + 2
      while (j < input.length && input[j] !== '\u0007' && input[j] !== '\u001b') j += 1
      i = input[j] === '\u001b' ? j + 2 : j + 1
    } else if (next === undefined) {
      break
    } else {
      i += 2 // two-character escape, e.g. ESC ( B
    }
  }

  if (buf) out.push({ kind: 'text', value: buf })
  return out
}

/** Apply one SGR parameter list to *style*, returning a new style object. */
function applySgr(style: Style, params: string): Style {
  const parts = params === '' ? [''] : params.split(';')
  const nums = parts.map((p) => {
    const n = Number.parseInt(p, 10)
    return Number.isNaN(n) ? 0 : n
  })

  const next: Style = { ...style }
  let i = 0

  while (i < nums.length) {
    const code = nums[i]
    if (code === 0) {
      Object.assign(next, DEFAULT_STYLE)
    } else if (code === 1) {
      next.bold = true
    } else if (code === 2) {
      next.dim = true
    } else if (code === 3) {
      next.italic = true
    } else if (code === 4) {
      next.underline = true
    } else if (code === 7) {
      next.inverse = true
    } else if (code === 22) {
      next.bold = false
      next.dim = false
    } else if (code === 23) {
      next.italic = false
    } else if (code === 24) {
      next.underline = false
    } else if (code === 27) {
      next.inverse = false
    } else if (code >= 30 && code <= 37) {
      next.fg = code - 30
    } else if (code >= 90 && code <= 97) {
      next.fg = code - 90 + 8
    } else if (code === 39) {
      next.fg = null
    } else if (code >= 40 && code <= 47) {
      next.bg = code - 40
    } else if (code >= 100 && code <= 107) {
      next.bg = code - 100 + 8
    } else if (code === 49) {
      next.bg = null
    } else if (code === 38 || code === 48) {
      const mode = nums[i + 1]
      if (mode === 5) {
        // 256-colour: only the first 16 map onto design tokens; the 6x6x6
        // cube has no Noir Violet equivalent, so it falls back to default
        // rather than guessing a hex value.
        const colour = nums[i + 2]
        if (colour >= 0 && colour < 16) {
          if (code === 38) next.fg = colour
          else next.bg = colour
        }
        i += 2
      } else if (mode === 2) {
        i += 4 // truecolor has no token mapping -- ignore it
      }
    }
    i += 1
  }

  return next
}

function classNames(style: Style): string {
  // Inverse swaps the effective foreground/background.
  const fg = style.inverse ? style.bg : style.fg
  const bg = style.inverse ? style.fg : style.bg

  const parts: string[] = []
  if (fg !== null && FG_CLASS[fg]) parts.push(FG_CLASS[fg])
  if (bg !== null && BG_CLASS[bg]) parts.push(BG_CLASS[bg])
  if (style.bold) parts.push('font-bold')
  if (style.dim) parts.push('opacity-60')
  if (style.italic) parts.push('italic')
  if (style.underline) parts.push('underline')

  return parts.join(' ')
}

/**
 * Parse shell output containing SGR escapes into styled text segments.
 *
 * Adjacent segments that share a class are merged, so output with no escapes
 * always yields exactly one segment -- which lets callers render plain lines as
 * bare text instead of a wrapper element.
 */
export function parseAnsi(input: string): AnsiSegment[] {
  const segments: AnsiSegment[] = []
  let style: Style = { ...DEFAULT_STYLE }
  let plain = ''

  // NOTE: takes the style explicitly. A closure over `style` would read the
  // variable at call time, and since flush only runs after the loop it would
  // stamp every segment with the *final* (reset) style instead of the style
  // that was in effect while the text was emitted.
  const flush = (atStyle: Style) => {
    if (!plain) return
    const className = classNames(atStyle)
    const last = segments[segments.length - 1]
    if (last && last.className === className) last.text += plain
    else segments.push({ text: plain, className })
    plain = ''
  }

  for (const token of tokenize(input)) {
    if (token.kind === 'text') {
      plain += token.value
    } else if (token.final === 'm') {
      // Text before this point belongs to the style that was in effect, so
      // flush before adopting the new one.
      flush(style)
      style = applySgr(style, token.params)
    }
    // Every other CSI (cursor motion, erase, mode changes) is dropped: this
    // endpoint never sends them, and if it did we could not honour them in a
    // flow layout.
  }
  flush(style)

  return segments
}

/** Remove every escape sequence, returning the visible text only. */
export function stripAnsi(input: string): string {
  let out = ''
  for (const token of tokenize(input)) {
    if (token.kind === 'text') out += token.value
  }
  return out
}
