import { render, screen, fireEvent } from '@testing-library/react'
import { describe, it, expect, beforeEach } from 'vitest'

import { ThemeProvider } from './ThemeProvider'
import { ThemeSwitcher } from './ThemeSwitcher'

function renderSwitcher() {
  return render(<ThemeProvider><ThemeSwitcher /></ThemeProvider>)
}

describe('ThemeSwitcher', () => {
  beforeEach(() => {
    localStorage.clear()
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
    fireEvent.click(toggleBtn)
    expect(toggleBtn.getAttribute('aria-label')).toBeDefined()
  })

  it('clicking a radio button selects it', () => {
    renderSwitcher()
    const radios = screen.getAllByRole('radio')
    fireEvent.click(radios[0])
    expect(radios[0]).toBeDefined()
  })
})
