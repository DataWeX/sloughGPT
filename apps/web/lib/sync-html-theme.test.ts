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
