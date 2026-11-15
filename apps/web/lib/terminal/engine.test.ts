/**
 * TerminalEngine — line-oriented VT interpreter for the shell pane.
 *
 * Input contract: each writeLine() call receives ONE line event from
 * /shell/exec/stream (the server split on LF and stripped it), so the engine
 * applies the LF itself: content lands on the current row, the row is
 * committed, and the cursor moves to a fresh bottom row.
 *
 * These tests pin the interpretation semantics: CR overwrite, backspace,
 * tab stops, CSI cursor movement, screen erase, SGR styling (design tokens
 * only), and scrollback capping.
 */
import { describe, it, expect } from 'vitest'
import { TerminalEngine } from './engine'

function rowText(engine: TerminalEngine, i: number): string {
  const rows = engine.snapshot()
  return rows[i]?.spans.map((s) => s.text).join('') ?? ''
}

describe('TerminalEngine', () => {
  it('plain line renders as one committed row with a single span', () => {
    const e = new TerminalEngine()
    e.writeLine('hello')
    const rows = e.snapshot()
    expect(rows).toHaveLength(1)
    expect(rows[0].spans).toEqual([{ text: 'hello', className: '' }])
    expect(typeof rows[0].key).toBe('number')
  })

  it('keeps row order across lines and preserves blank lines', () => {
    const e = new TerminalEngine()
    e.writeLine('first')
    e.writeLine('')
    e.writeLine('third')
    const rows = e.snapshot()
    expect(rows).toHaveLength(3)
    expect(rowText(e, 0)).toBe('first')
    expect(rows[1].spans).toEqual([])
    expect(rowText(e, 2)).toBe('third')
  })

  it('row keys are stable identifiers (monotonic, never reused)', () => {
    const e = new TerminalEngine()
    e.writeLine('a')
    const k0 = e.snapshot()[0].key
    e.reset()
    e.writeLine('b')
    expect(e.snapshot()[0].key).not.toBe(k0)
    expect(k0).toBeGreaterThan(0)
  })

  it('CR inside a line overwrites from column 0', () => {
    const e = new TerminalEngine()
    e.writeLine('abcdef\rXY')
    expect(rowText(e, 0)).toBe('XYcdef')
  })

  it('CR with shorter overwrite keeps the untouched tail', () => {
    const e = new TerminalEngine()
    // 'total free space' (16 cells) overwritten by 5 chars → tail from col 5
    e.writeLine('total free space\r  42%')
    expect(rowText(e, 0)).toBe('  42% free space')
  })

  it('backspace moves the cursor back so the next write overwrites', () => {
    const e = new TerminalEngine()
    e.writeLine('abc\b!')
    expect(rowText(e, 0)).toBe('ab!')
  })

  it('tab advances to the next 8-column stop', () => {
    const e = new TerminalEngine()
    e.writeLine('ab\tX')
    expect(rowText(e, 0)).toBe('ab      X')
  })

  it('ESC[2J clears previously committed rows (watch -c pattern)', () => {
    const e = new TerminalEngine()
    e.writeLine('old content')
    e.writeLine('\x1b[2J\x1b[HEvery 0.5s: date')
    const rows = e.snapshot()
    expect(rows).toHaveLength(1)
    expect(rowText(e, 0)).toBe('Every 0.5s: date')
  })

  it('cursor-up rewrites an earlier row (progress-bar pattern)', () => {
    const e = new TerminalEngine()
    e.writeLine('step 1/3')
    e.writeLine('\x1b[1A\x1b[Kstep 2/3')
    const rows = e.snapshot()
    expect(rowText(e, 0)).toBe('step 2/3')
    // the earlier cursor row never received content — not rendered
    expect(rows).toHaveLength(1)
  })

  it('SGR colors map to design tokens and reset returns to default', () => {
    const e = new TerminalEngine()
    e.writeLine('\x1b[31mred\x1b[0m plain')
    const spans = e.snapshot()[0].spans
    expect(spans).toHaveLength(2)
    expect(spans[0].text).toBe('red')
    expect(spans[0].className).toContain('text-term-close')
    expect(spans[1].text).toBe(' plain')
    expect(spans[1].className).toBe('')
  })

  it('supports the common SGR palette via term tokens only', () => {
    const e = new TerminalEngine()
    e.writeLine(
      '\x1b[32mgreen\x1b[33myellow\x1b[34mblue\x1b[36mcyan\x1b[35mmagenta\x1b[1;4mbold-under',
    )
    const spans = e.snapshot()[0].spans
    const cls = (i: number) => spans[i].className
    expect(cls(0)).toContain('text-term-zoom')
    expect(cls(1)).toContain('text-term-minimize')
    expect(cls(2)).toContain('text-term-blue')
    expect(cls(3)).toContain('text-term-cyan')
    expect(cls(4)).toContain('text-term-purple')
    expect(cls(5)).toContain('font-bold')
    expect(cls(5)).toContain('underline')
  })

  it('consumes 256-color and truecolor params without corrupting parsing', () => {
    const e = new TerminalEngine()
    e.writeLine('\x1b[38;5;196mred256\x1b[38;2;255;0;0mrgb\x1b[0m tail')
    const rows = e.snapshot()
    // color specs are consumed (no token-safe mapping) — text must be pristine
    expect(rows[0].spans.map((s) => s.text).join('')).toBe('red256rgb tail')
    expect(rows[0].spans.every((s) => s.className === '')).toBe(true)
  })

  it('unknown CSI sequences and stray control chars are consumed, not rendered', () => {
    const e = new TerminalEngine()
    e.writeLine('a\x1b[99Zb\x07c')
    expect(rowText(e, 0)).toBe('abc')
  })

  it('OSC title sequences are skipped entirely', () => {
    const e = new TerminalEngine()
    e.writeLine('\x1b]0;my title\x07visible')
    expect(rowText(e, 0)).toBe('visible')
  })

  it('escape sequences split across writeLine calls still parse', () => {
    // A sequence can arrive fragmented across two on_write chunks server-side;
    // the server reassembles lines, but a CSI spanning two *line events* cannot
    // happen (no LF inside CSI). Fragment handling within one line is the
    // server's job — here we only pin that a dangling ESC at line end does not
    // leak into the rendered text.
    const e = new TerminalEngine()
    e.writeLine('x\x1b')
    expect(rowText(e, 0)).toBe('x')
  })

  it('snapshot respects maxLines (scrollback cap)', () => {
    const e = new TerminalEngine({ maxLines: 3 })
    for (let i = 0; i < 10; i++) e.writeLine(`line ${i}`)
    const rows = e.snapshot()
    expect(rows).toHaveLength(3)
    expect(rowText(e, 2)).toBe('line 9')
  })

  it('reset clears committed rows and cursor state', () => {
    const e = new TerminalEngine()
    e.writeLine('junk')
    e.reset()
    expect(e.snapshot()).toEqual([])
    e.writeLine('fresh')
    expect(rowText(e, 0)).toBe('fresh')
  })

  it('spans compact consecutive cells with equal style', () => {
    const e = new TerminalEngine()
    e.writeLine('\x1b[32mgreen ok\x1b[0m tail')
    const rows = e.snapshot()
    expect(rows[0].spans).toHaveLength(2)
    expect(rows[0].spans[0].text).toBe('green ok')
    expect(rows[0].spans[1].text).toBe(' tail')
  })

  it('long line without LF is one row (visual wrap stays a CSS concern)', () => {
    const e = new TerminalEngine()
    e.writeLine('x'.repeat(500))
    const rows = e.snapshot()
    expect(rows).toHaveLength(1)
    expect(rows[0].spans[0].text).toHaveLength(500)
  })
})
