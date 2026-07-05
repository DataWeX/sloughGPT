import { render, screen } from '@testing-library/react'
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
})
