import { render, screen, fireEvent } from '@testing-library/react'
import { renderToString } from 'react-dom/server'
import { describe, it, expect, beforeEach } from 'vitest'

import { ThemeProvider } from './ThemeProvider'
import { ThemeSwitcher } from './ThemeSwitcher'
import { MODE_STORAGE_KEY, THEME_STORAGE_KEY, PALETTE_STORAGE_KEY } from '@/lib/theme-storage'

function renderSwitcher() {
  return render(
    <ThemeProvider>
      <ThemeSwitcher />
    </ThemeProvider>,
  )
}

describe('ThemeSwitcher', () => {
  beforeEach(() => {
    localStorage.clear()
  })

  it('renders mode-neutral markup on the server (prevents hydration mismatch)', () => {
    const html = renderToString(
      <ThemeProvider>
        <ThemeSwitcher />
      </ThemeProvider>,
    )
    // Every checked state is gated on `mounted`: the server never claims a
    // selection, because stored mode/colour/palette only exist client-side.
    expect(html).not.toContain('aria-checked="true"')
    // Labels are mode-stable — both segments exist server-side.
    expect(html).toContain('Switch to light mode')
    expect(html).toContain('Switch to dark mode')
    // Structure present: one group, three radiogroups.
    expect(html).toContain('role="radiogroup"')
    expect(html).toContain('aria-label="Theme settings"')
  })

  it('reflects stored selections only after client mount', () => {
    localStorage.setItem(MODE_STORAGE_KEY, 'light')
    localStorage.setItem(THEME_STORAGE_KEY, 'blue')
    localStorage.setItem(PALETTE_STORAGE_KEY, 'neural-precision')
    renderSwitcher()
    expect(screen.getByRole('radio', { name: 'Switch to light mode' })).toHaveAttribute(
      'aria-checked',
      'true',
    )
    expect(screen.getByRole('radio', { name: 'Periwinkle' })).toHaveAttribute(
      'aria-checked',
      'true',
    )
    expect(screen.getByRole('radio', { name: 'Neural Precision' })).toHaveAttribute(
      'aria-checked',
      'true',
    )
    const caption = screen.getByTestId('theme-caption').textContent ?? ''
    expect(caption).toContain('Periwinkle')
    expect(caption).toContain('Neural Precision')
  })

  it('mode segment switches mode and persists it', () => {
    renderSwitcher()
    expect(screen.getByRole('radio', { name: 'Switch to dark mode' })).toHaveAttribute(
      'aria-checked',
      'true',
    )
    fireEvent.click(screen.getByRole('radio', { name: 'Switch to light mode' }))
    expect(screen.getByRole('radio', { name: 'Switch to light mode' })).toHaveAttribute(
      'aria-checked',
      'true',
    )
    expect(localStorage.getItem(MODE_STORAGE_KEY)).toBe('light')
  })

  it('arrow keys rove focus and selection follows focus (colour radiogroup)', () => {
    renderSwitcher()
    const blue = screen.getByRole('radio', { name: 'Periwinkle' })
    fireEvent.click(blue)
    expect(blue).toHaveAttribute('aria-checked', 'true')

    blue.focus()
    fireEvent.keyDown(blue, { key: 'ArrowRight' })
    const lilac = screen.getByRole('radio', { name: 'Lilac' })
    expect(lilac).toHaveAttribute('aria-checked', 'true')
    expect(lilac).toHaveFocus()
    expect(screen.getByTestId('theme-caption').textContent).toContain('Lilac')

    fireEvent.keyDown(lilac, { key: 'ArrowRight' })
    expect(screen.getByRole('radio', { name: 'Rose' })).toHaveAttribute('aria-checked', 'true')
  })

  it('renders 11 radios across 3 radiogroups inside the settings group', () => {
    renderSwitcher()
    expect(screen.getByRole('group', { name: 'Theme settings' })).toBeTruthy()
    expect(screen.getAllByRole('radiogroup').length).toBe(3)
    // 2 mode + 7 colours + 2 palettes
    expect(screen.getAllByRole('radio').length).toBe(11)
    // default palette checked after mount
    expect(screen.getByRole('radio', { name: 'Noir Violet' })).toHaveAttribute(
      'aria-checked',
      'true',
    )
  })

  it('palette chip selection persists the stored palette', () => {
    renderSwitcher()
    fireEvent.click(screen.getByRole('radio', { name: 'Neural Precision' }))
    expect(screen.getByRole('radio', { name: 'Neural Precision' })).toHaveAttribute(
      'aria-checked',
      'true',
    )
    expect(localStorage.getItem(PALETTE_STORAGE_KEY)).toBe('neural-precision')
  })

  it('collapsed variant: compact mode toggle only, no radiogroups', () => {
    render(
      <ThemeProvider>
        <ThemeSwitcher collapsed />
      </ThemeProvider>,
    )
    expect(screen.queryAllByRole('radiogroup').length).toBe(0)
    // default mode is dark → the toggle's target is light
    const toggle = screen.getByRole('button', { name: 'Switch to light mode' })
    fireEvent.click(toggle)
    expect(localStorage.getItem(MODE_STORAGE_KEY)).toBe('light')
    expect(screen.getByRole('button', { name: 'Switch to dark mode' })).toBeTruthy()
  })

  it('collapsed variant: mode-stable markup on the server', () => {
    const html = renderToString(
      <ThemeProvider>
        <ThemeSwitcher collapsed />
      </ThemeProvider>,
    )
    expect(html).toContain('Switch to light mode')
    expect(html).not.toContain('Switch to dark mode')
  })
})
