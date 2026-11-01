import { describe, expect, it, beforeEach } from 'vitest'
import { syncHtmlTheme } from './sync-html-theme'

describe('syncHtmlTheme', () => {
  beforeEach(() => {
    document.documentElement.className = ''
  })

  it('adds mode class', () => {
    syncHtmlTheme('dark', 'blue')
    expect(document.documentElement.classList.contains('dark')).toBe(true)
  })

  it('removes opposite mode class', () => {
    document.documentElement.classList.add('light')
    syncHtmlTheme('dark', 'blue')
    expect(document.documentElement.classList.contains('light')).toBe(false)
    expect(document.documentElement.classList.contains('dark')).toBe(true)
  })

  it('adds theme class', () => {
    syncHtmlTheme('light', 'purple')
    expect(document.documentElement.classList.contains('theme-purple')).toBe(true)
  })

  it('removes old theme classes', () => {
    document.documentElement.classList.add('theme-blue')
    syncHtmlTheme('dark', 'green')
    expect(document.documentElement.classList.contains('theme-blue')).toBe(false)
    expect(document.documentElement.classList.contains('theme-green')).toBe(true)
  })

  it('preserves non-theme non-mode classes', () => {
    document.documentElement.classList.add('next-font')
    syncHtmlTheme('dark', 'red')
    expect(document.documentElement.classList.contains('next-font')).toBe(true)
  })

  it('noops when document is undefined', () => {
    const orig = globalThis.document
    ;(globalThis as any).document = undefined
    expect(() => syncHtmlTheme('dark', 'blue')).not.toThrow()
    globalThis.document = orig
  })
})

describe('syncHtmlTheme — palette', () => {
  beforeEach(() => {
    document.documentElement.className = ''
  })

  it('adds palette class when palette is neural-precision', () => {
    syncHtmlTheme('dark', 'purple', 'neural-precision')
    expect(document.documentElement.classList.contains('palette-neural-precision')).toBe(true)
  })

  it('does not add palette class when palette is noir-violet', () => {
    syncHtmlTheme('dark', 'purple', 'noir-violet')
    expect(document.documentElement.classList.contains('palette-noir-violet')).toBe(false)
  })

  it('does not add palette class when palette is undefined', () => {
    syncHtmlTheme('dark', 'purple')
    expect(document.documentElement.className).not.toContain('palette-')
  })

  it('removes old palette classes when switching', () => {
    document.documentElement.classList.add('palette-neural-precision')
    syncHtmlTheme('dark', 'purple', 'noir-violet')
    expect(document.documentElement.classList.contains('palette-neural-precision')).toBe(false)
  })

  it('removes all palette classes even without palette param', () => {
    document.documentElement.classList.add('palette-neural-precision')
    syncHtmlTheme('dark', 'purple')
    expect(document.documentElement.classList.contains('palette-neural-precision')).toBe(false)
  })
})
