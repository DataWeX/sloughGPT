import { render, screen, waitFor, act } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'

vi.mock('@/lib/db', () => {
  const store = new Map<string, unknown>()
  return {
    chatDB: {
      getKV: vi.fn((key: string) => Promise.resolve(store.get(key) as string | undefined)),
      setKV: vi.fn((key: string, value: unknown) => { store.set(key, value); return Promise.resolve() }),
    },
  }
})

import { ThemeProvider, useTheme } from './ThemeProvider'
import { chatDB } from '@/lib/db'

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

  it('reads saved theme and mode from chatDB', async () => {
    const store = new Map<string, unknown>([
      ['man_theme', 'green'],
      ['man_mode', 'light'],
    ])
    vi.mocked(chatDB.getKV).mockImplementation((key: string) => Promise.resolve(store.get(key) as string | undefined))
    const { container } = render(<ThemeProvider><TestChild /></ThemeProvider>)
    await waitFor(() => {
      const themes = container.querySelectorAll('[data-testid="theme"]')
      const lastTheme = themes[themes.length - 1]
      const modes = container.querySelectorAll('[data-testid="mode"]')
      const lastMode = modes[modes.length - 1]
      expect(lastTheme.textContent).toBe('green')
      expect(lastMode.textContent).toBe('light')
    })
  })

  it('falls back to defaults for invalid chatDB values', async () => {
    vi.mocked(chatDB.getKV).mockImplementation((key: string) =>
      Promise.resolve(key === 'man_theme' ? 'invalid' : key === 'man_mode' ? 'invalid' : undefined)
    )
    const { container } = render(<ThemeProvider><TestChild /></ThemeProvider>)
    await waitFor(() => {
      const themes = container.querySelectorAll('[data-testid="theme"]')
      const lastTheme = themes[themes.length - 1]
      const modes = container.querySelectorAll('[data-testid="mode"]')
      const lastMode = modes[modes.length - 1]
      expect(lastTheme.textContent).toBe('purple')
      expect(lastMode.textContent).toBe('dark')
    })
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

  it('reads saved palette from chatDB', async () => {
    vi.mocked(chatDB.getKV).mockImplementation((key: string) =>
      Promise.resolve(key === 'man_palette' ? 'neural-precision' : undefined)
    )
    const { container } = render(<ThemeProvider><TestChild /></ThemeProvider>)
    await waitFor(() => {
      const palettes = container.querySelectorAll('[data-testid="palette"]')
      const last = palettes[palettes.length - 1]
      expect(last.textContent).toBe('neural-precision')
    })
  })

  it('falls back to noir-violet for invalid chatDB value', async () => {
    vi.mocked(chatDB.getKV).mockImplementation((key: string) =>
      Promise.resolve(key === 'man_palette' ? 'solarized' : undefined)
    )
    const { container } = render(<ThemeProvider><TestChild /></ThemeProvider>)
    await waitFor(() => {
      const palettes = container.querySelectorAll('[data-testid="palette"]')
      const last = palettes[palettes.length - 1]
      expect(last.textContent).toBe('noir-violet')
    })
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

  it('persists palette to chatDB', async () => {
    vi.mocked(chatDB.setKV).mockResolvedValue(undefined)
    const { container } = render(<ThemeProvider><TestChild /></ThemeProvider>)
    await act(async () => { await new Promise(r => setTimeout(r, 0)) })
    const btn = container.querySelector('[data-testid="set-palette"]') as HTMLButtonElement
    btn.click()
    await waitFor(() => {
      expect(chatDB.setKV).toHaveBeenCalledWith('man_palette', 'neural-precision')
    })
  })
})
