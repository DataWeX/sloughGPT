/**
 * Tests for the SGR renderer used by the browser shell.
 *
 * The escape samples below mirror live `/shell/exec` responses (`help`,
 * `logs`) so the parser is pinned to what the backend actually sends.
 */
import { describe, it, expect } from 'vitest'
import { parseAnsi, stripAnsi } from './ansi'

const ESC = '\u001b'

describe('parseAnsi', () => {
  it('returns a single unstyled segment for plain text', () => {
    expect(parseAnsi('hello')).toEqual([{ text: 'hello', className: '' }])
  })

  it('returns nothing for an empty string', () => {
    expect(parseAnsi('')).toEqual([])
  })

  it('never leaks escape characters into segment text', () => {
    const segments = parseAnsi(`${ESC}[36mcyan${ESC}[0m`)
    for (const seg of segments) {
      expect(seg.text).not.toContain(ESC)
      expect(seg.text).not.toContain('[')
    }
  })

  it('renders the cyan help header with the primary token', () => {
    const segments = parseAnsi(`${ESC}[36mConsole Logs${ESC}[0m`)
    expect(segments).toEqual([{ text: 'Console Logs', className: 'text-primary' }])
  })

  it('renders the yellow WRN badge with the warning token', () => {
    const segments = parseAnsi(`${ESC}[33mWRN${ESC}[0m`)
    expect(segments).toEqual([{ text: 'WRN', className: 'text-warning' }])
  })

  it('renders bold and dim independently', () => {
    expect(parseAnsi(`${ESC}[1mTitle${ESC}[0m`)).toEqual([
      { text: 'Title', className: 'font-bold' },
    ])
    expect(parseAnsi(`${ESC}[2m693 buffered${ESC}[0m`)).toEqual([
      { text: '693 buffered', className: 'opacity-60' },
    ])
  })

  it('combines style attributes on one segment', () => {
    const segments = parseAnsi(`${ESC}[1;31mfail${ESC}[0m`)
    expect(segments).toEqual([{ text: 'fail', className: 'text-destructive font-bold' }])
  })

  it('keeps colour across a reset boundary', () => {
    const segments = parseAnsi(`a${ESC}[32m b${ESC}[0m c`)
    expect(segments).toEqual([
      { text: 'a', className: '' },
      { text: ' b', className: 'text-success' },
      { text: ' c', className: '' },
    ])
  })

  it('merges adjacent segments that share a class', () => {
    const segments = parseAnsi(`${ESC}[36mone${ESC}[0m${ESC}[36mtwo${ESC}[0m`)
    expect(segments).toEqual([{ text: 'onetwo', className: 'text-primary' }])
  })

  it('handles a real multi-line logs sample', () => {
    const raw =
      `  ${ESC}[1mConsole Logs${ESC}[0m ${ESC}[2m(693 buffered)${ESC}[0m\n` +
      `  ${ESC}[2m----${ESC}[0m ${ESC}[33mWRN${ESC}[0m ok`
    const segments = parseAnsi(raw)
    // NOTE: keyed scans, not a Map -- several segments legitimately share a
    // class, and a Map keyed by className would keep only the last of them.
    const textsWithClass = (cls: string) =>
      segments
        .filter((s) => s.className === cls)
        .map((s) => s.text)
        .join('')
    expect(textsWithClass('font-bold')).toContain('Console Logs')
    expect(textsWithClass('opacity-60')).toContain('(693 buffered)')
    expect(textsWithClass('opacity-60')).toContain('----')
    expect(textsWithClass('text-warning')).toBe('WRN')
    expect(segments.map((s) => s.text).join('')).toContain('ok')
  })

  it('applies the 256-colour cube only when it maps to a base colour', () => {
    expect(parseAnsi(`${ESC}[38;5;1mred${ESC}[0m`)).toEqual([
      { text: 'red', className: 'text-destructive' },
    ])
    // 214 has no token mapping - falls back rather than guessing a hex value
    expect(parseAnsi(`${ESC}[38;5;214morange${ESC}[0m`)).toEqual([
      { text: 'orange', className: '' },
    ])
  })

  it('ignores truecolor sequences without breaking the text', () => {
    const segments = parseAnsi(`${ESC}[38;2;255;0;0mred${ESC}[0m`)
    expect(segments).toEqual([{ text: 'red', className: '' }])
  })

  it('drops cursor-motion and screen-control sequences', () => {
    // Never sent by /shell/exec, but must not render as garbage if they appear.
    const raw = `line${ESC}[1A${ESC}[2K${ESC}[2J${ESC}[H${ESC}[?25l`
    expect(parseAnsi(raw)).toEqual([{ text: 'line', className: '' }])
  })

  it('drops OSC sequences', () => {
    const raw = `before${ESC}]0;window title\u0007after`
    expect(parseAnsi(raw)).toEqual([{ text: 'beforeafter', className: '' }])
  })

  it('drops a truncated escape instead of leaking it', () => {
    expect(parseAnsi(`text${ESC}[36`)).toEqual([{ text: 'text', className: '' }])
    expect(parseAnsi(`text${ESC}`)).toEqual([{ text: 'text', className: '' }])
  })

  it('resets every attribute on 0', () => {
    const segments = parseAnsi(`${ESC}[1;4;31ma${ESC}[0mb`)
    expect(segments).toEqual([
      { text: 'a', className: 'text-destructive font-bold underline' },
      { text: 'b', className: '' },
    ])
  })

  it('clears bold and dim on 22 without clearing colour', () => {
    const segments = parseAnsi(`${ESC}[1;36ma${ESC}[22mb`)
    expect(segments).toEqual([
      { text: 'a', className: 'text-primary font-bold' },
      { text: 'b', className: 'text-primary' },
    ])
  })

  it('treats an empty SGR parameter as reset', () => {
    expect(parseAnsi(`${ESC}[36ma${ESC}[mb`)).toEqual([
      { text: 'a', className: 'text-primary' },
      { text: 'b', className: '' },
    ])
  })

  it('swaps colour roles on inverse video', () => {
    // ESC[7;37;41m = inverse + fg white(7) + bg red(1); inverse swaps them so
    // the visible text is red-on-white rather than white-on-red.
    const segments = parseAnsi(`${ESC}[7;37;41mhi${ESC}[0m`)
    expect(segments).toEqual([{ text: 'hi', className: 'text-destructive bg-background' }])
  })

  it('honours default-colour resets 39 and 49', () => {
    const segments = parseAnsi(`${ESC}[36ma${ESC}[39mb`)
    expect(segments).toEqual([
      { text: 'a', className: 'text-primary' },
      { text: 'b', className: '' },
    ])
  })
})

describe('stripAnsi', () => {
  it('returns plain text unchanged', () => {
    expect(stripAnsi('hello world')).toBe('hello world')
  })

  it('removes colour codes', () => {
    expect(stripAnsi(`${ESC}[36mhi${ESC}[0m`)).toBe('hi')
  })

  it('removes control sequences and OSC', () => {
    expect(stripAnsi(`a${ESC}[2J${ESC}[Hb${ESC}]0;t\u0007c`)).toBe('abc')
  })

  it('preserves newlines and spacing', () => {
    expect(stripAnsi(`${ESC}[2mline1${ESC}[0m\n  line2`)).toBe('line1\n  line2')
  })

  it('round-trips with parseAnsi', () => {
    const raw = `${ESC}[1mhead${ESC}[0m\n${ESC}[33mWRN${ESC}[0m body`
    expect(stripAnsi(raw)).toBe(
      parseAnsi(raw)
        .map((s) => s.text)
        .join(''),
    )
  })
})
