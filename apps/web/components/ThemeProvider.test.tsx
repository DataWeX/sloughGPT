import { render, screen, waitFor } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'

import { ThemeProvider, useTheme } from './ThemeProvider'

function TestChild() {
  const { theme, mode, palette, setTheme, setMode, setPalette } = useTheme()
  return (
    <div>
      <span data-testid="theme">{theme}</span>
      <span data-testid="mode">{mode}</span>
      <span data-testid="palette">{palette}</span>
      <button data-testid="set-theme" onClick={() => setTheme('blue')}>set theme</button>
      <button data-testid="set-mode" onClick={() => setMode('light')}>set mode</button>
      <button data-testid="set-palette" onClick={() => setPalette('neural-precision')}>set palette</button>
    </div>
  )
}

describe('ThemeProvider', () => {
  beforeEach(() => {
    localStorage.clear()
  })

  it('renders children', () => {
    render(<ThemeProvider><div>hello</div></ThemeProvider>)
    expect(screen.getByText('hello')).toBeInTheDocument()
  })

  it('uses default theme and mode when no localStorage', () => {
    const { container } = render(<ThemeProvider><TestChild /></ThemeProvider>)
    const themes = container.querySelectorAll('[data-testid="theme"]')
    const lastTheme = themes[themes.length - 1]
    const modes = container.querySelectorAll('[data-testid="mode"]')
    const lastMode = modes[modes.length - 1]
    expect(lastTheme.textContent).toBe('purple')
    expect(lastMode.textContent).toBe('dark')
  })

  it('reads saved theme and mode from localStorage', () => {
    localStorage.setItem('man_theme', 'green')
    localStorage.setItem('man_mode', 'light')
    const { container } = render(<ThemeProvider><TestChild /></ThemeProvider>)
    const themes = container.querySelectorAll('[data-testid="theme"]')
    const lastTheme = themes[themes.length - 1]
    const modes = container.querySelectorAll('[data-testid="mode"]')
    const lastMode = modes[modes.length - 1]
    expect(lastTheme.textContent).toBe('green')
    expect(lastMode.textContent).toBe('light')
  })

  it('falls back to defaults for invalid localStorage values', () => {
    localStorage.setItem('man_theme', 'invalid')
    localStorage.setItem('man_mode', 'invalid')
    const { container } = render(<ThemeProvider><TestChild /></ThemeProvider>)
    const themes = container.querySelectorAll('[data-testid="theme"]')
    const lastTheme = themes[themes.length - 1]
    const modes = container.querySelectorAll('[data-testid="mode"]')
    const lastMode = modes[modes.length - 1]
    expect(lastTheme.textContent).toBe('purple')
    expect(lastMode.textContent).toBe('dark')
  })

  it('throws when useTheme is used outside provider', () => {
    const error = vi.spyOn(console, 'error').mockImplementation(() => {})
    expect(() => render(<TestChild />)).toThrow('useTheme must be used within a ThemeProvider')
    error.mockRestore()
  })

  it('exposes setTheme and setMode', () => {
    const { container } = render(<ThemeProvider><TestChild /></ThemeProvider>)
    const btns = container.querySelectorAll('[data-testid="set-theme"]')
    expect(btns.length).toBeGreaterThanOrEqual(1)
  })
})

describe('ThemeProvider — palette', () => {
  beforeEach(() => {
    localStorage.clear()
  })

  it('defaults palette to noir-violet when no localStorage', () => {
    const { container } = render(<ThemeProvider><TestChild /></ThemeProvider>)
    const palettes = container.querySelectorAll('[data-testid="palette"]')
    const last = palettes[palettes.length - 1]
    expect(last.textContent).toBe('noir-violet')
  })

  it('reads saved palette from localStorage', () => {
    localStorage.setItem('man_palette', 'neural-precision')
    const { container } = render(<ThemeProvider><TestChild /></ThemeProvider>)
    const palettes = container.querySelectorAll('[data-testid="palette"]')
    const last = palettes[palettes.length - 1]
    expect(last.textContent).toBe('neural-precision')
  })

  it('falls back to noir-violet for invalid localStorage value', () => {
    localStorage.setItem('man_palette', 'solarized')
    const { container } = render(<ThemeProvider><TestChild /></ThemeProvider>)
    const palettes = container.querySelectorAll('[data-testid="palette"]')
    const last = palettes[palettes.length - 1]
    expect(last.textContent).toBe('noir-violet')
  })

  it('setPalette updates context value', async () => {
    const { container } = render(<ThemeProvider><TestChild /></ThemeProvider>)
    const btn = container.querySelector('[data-testid="set-palette"]') as HTMLButtonElement
    btn.click()
    await waitFor(() => {
      const palettes = container.querySelectorAll('[data-testid="palette"]')
      const last = palettes[palettes.length - 1]
      expect(last.textContent).toBe('neural-precision')
    })
  })

  it('persists palette to localStorage', async () => {
    const { container } = render(<ThemeProvider><TestChild /></ThemeProvider>)
    const btn = container.querySelector('[data-testid="set-palette"]') as HTMLButtonElement
    btn.click()
    await waitFor(() => {
      expect(localStorage.getItem('man_palette')).toBe('neural-precision')
    })
  })
})
