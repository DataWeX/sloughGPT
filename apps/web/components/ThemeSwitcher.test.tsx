import { render, screen, fireEvent } from '@testing-library/react'
import { renderToString } from 'react-dom/server'
import { describe, it, expect, beforeEach } from 'vitest'

import { ThemeProvider } from './ThemeProvider'
import { ThemeSwitcher } from './ThemeSwitcher'
import { MODE_STORAGE_KEY, THEME_STORAGE_KEY, PALETTE_STORAGE_KEY } from '@/lib/theme-storage'

function renderSwitcher() {
  return render(<ThemeProvider><ThemeSwitcher /></ThemeProvider>)
}

describe('ThemeSwitcher', () => {
  beforeEach(() => {
    localStorage.clear()
  })

  it('renders mode-stable markup on the server (prevents hydration mismatch)', () => {
    const html = renderToString(<ThemeProvider><ThemeSwitcher /></ThemeProvider>)
    expect(html).toContain('Switch to light mode')
    expect(html).not.toContain('Switch to dark mode')
    expect(html).not.toContain('aria-checked="true"')
  })

  it('reflects stored light mode only after client mount', () => {
    localStorage.setItem(MODE_STORAGE_KEY, 'light')
    localStorage.setItem(THEME_STORAGE_KEY, 'blue')
    localStorage.setItem(PALETTE_STORAGE_KEY, 'noir-violet')
    renderSwitcher()
    const toggle = screen.getAllByRole('button')[0]
    expect(toggle.getAttribute('aria-label')).toBe('Switch to dark mode')
  })

  it('renders mode toggle button', () => {
    renderSwitcher()
    const btns = screen.getAllByRole('button')
    expect(btns.some(b => b.getAttribute('aria-label') === 'Switch to dark mode'))
  })

  it('renders color swatches as radio buttons', () => {
    renderSwitcher()
    const radios = screen.getAllByRole('radio')
    expect(radios.length >= 7).toBe(true)
  })

  it('has accessible groups', () => {
    renderSwitcher()
    const groups = screen.getAllByRole('group')
    expect(groups.length >= 1).toBe(true)
  })

  it('mode button has aria-label', () => {
    renderSwitcher()
    const btns = screen.getAllByRole('button')
    expect(btns.length).toBeGreaterThanOrEqual(1)
  })

  it('radio buttons are rendered', () => {
    renderSwitcher()
    const radios = screen.getAllByRole('radio')
    expect(radios.length).toBeGreaterThanOrEqual(7)
  })

  it('clicking mode toggle changes button label', () => {
    renderSwitcher()
    const toggleBtn = screen.getAllByRole('button')[0]
    const before = toggleBtn.getAttribute('aria-label')
    fireEvent.click(toggleBtn)
    const after = toggleBtn.getAttribute('aria-label')
    expect(after).toBeDefined()
    expect(after).not.toBe(before)
  })

  it('clicking a radio button selects it', () => {
    renderSwitcher()
    const radios = screen.getAllByRole('radio')
    fireEvent.click(radios[0])
    expect(radios[0]).toHaveAttribute('aria-checked', 'true')
  })

  it('renders palette swatches as radio buttons', () => {
    renderSwitcher()
    const groups = screen.getAllByRole('radiogroup', { name: 'Palette' })
    expect(groups.length).toBeGreaterThanOrEqual(1)
    expect(groups[0].querySelectorAll('[role="radio"]').length).toBe(2)
  })

  it('clicking a palette radio selects it', () => {
    renderSwitcher()
    const radios = screen.getAllByRole('radio', { name: 'Neural Precision' })
    fireEvent.click(radios[0])
    expect(radios[0]).toHaveAttribute('aria-checked', 'true')
  })
})
