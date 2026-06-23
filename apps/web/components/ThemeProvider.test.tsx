// @vitest-environment jsdom
import { render, screen, waitFor } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'

import { ThemeProvider, useTheme } from './ThemeProvider'

function TestChild() {
  const { theme, mode, setTheme, setMode } = useTheme()
  return (
    <div>
      <span data-testid="theme">{theme}</span>
      <span data-testid="mode">{mode}</span>
      <button data-testid="set-theme" onClick={() => setTheme('blue')}>set theme</button>
      <button data-testid="set-mode" onClick={() => setMode('light')}>set mode</button>
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
