import { describe, expect, it, beforeEach } from 'vitest'
import { syncHtmlTheme, resetThemeSyncGuard } from './sync-html-theme'

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

describe('syncHtmlTheme — high contrast', () => {
  beforeEach(() => {
    document.documentElement.className = ''
  })

  it('applies dark and high-contrast together (high contrast layers on dark)', () => {
    syncHtmlTheme('high-contrast', 'blue')
    const root = document.documentElement
    expect(root.classList.contains('dark')).toBe(true)
    expect(root.classList.contains('high-contrast')).toBe(true)
    expect(root.classList.contains('light')).toBe(false)
  })

  it('drops both classes when leaving high contrast for light', () => {
    syncHtmlTheme('high-contrast', 'blue')
    syncHtmlTheme('light', 'blue')
    const root = document.documentElement
    expect(root.classList.contains('dark')).toBe(false)
    expect(root.classList.contains('high-contrast')).toBe(false)
    expect(root.classList.contains('light')).toBe(true)
  })

  it('strips high-contrast when switching to plain dark', () => {
    syncHtmlTheme('high-contrast', 'blue')
    syncHtmlTheme('dark', 'blue')
    const root = document.documentElement
    expect(root.classList.contains('high-contrast')).toBe(false)
    expect(root.classList.contains('dark')).toBe(true)
  })
})

describe('syncHtmlTheme — swap transition', () => {
  const armed = () => document.documentElement.classList.contains('theme-transitioning')

  beforeEach(() => {
    document.documentElement.className = ''
    resetThemeSyncGuard()
  })

  it('never arms the transition on the first sync (first paint)', () => {
    syncHtmlTheme('dark', 'purple')
    expect(armed()).toBe(false)
  })

  it('arms the transition for a user-initiated change', () => {
    syncHtmlTheme('dark', 'purple') // mount
    syncHtmlTheme('light', 'purple') // change
    expect(armed()).toBe(true)
  })

  it('disarms once the swap window has elapsed', async () => {
    syncHtmlTheme('dark', 'purple')
    syncHtmlTheme('light', 'purple')
    expect(armed()).toBe(true)
    await new Promise((r) => setTimeout(r, 260))
    expect(armed()).toBe(false)
  })

  it('honours an explicit animate=false even after the first sync', () => {
    syncHtmlTheme('dark', 'purple')
    syncHtmlTheme('light', 'purple', undefined, false)
    expect(armed()).toBe(false)
  })

  it('resetThemeSyncGuard returns the module to a cold-document state', () => {
    syncHtmlTheme('dark', 'purple')
    resetThemeSyncGuard()
    syncHtmlTheme('light', 'purple')
    expect(armed()).toBe(false)
  })

  it('keeps the mode/theme/palette swap itself correct while armed', () => {
    syncHtmlTheme('dark', 'purple')
    syncHtmlTheme('high-contrast', 'blue', 'neural-precision')
    const root = document.documentElement
    expect(armed()).toBe(true)
    expect(root.classList.contains('high-contrast')).toBe(true)
    expect(root.classList.contains('theme-blue')).toBe(true)
    expect(root.classList.contains('palette-neural-precision')).toBe(true)
  })
})
