import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { V86TerminalPanel } from './V86TerminalPanel'

vi.mock('@/hooks/useV86', () => ({
  useV86: () => ({
    isBooted: false,
    error: null,
    init: vi.fn(),
    reset: vi.fn(),
  }),
}))

afterEach(() => cleanup())

describe('V86TerminalPanel', () => {
  it('shows Booting when not booted', () => {
    render(<V86TerminalPanel />)
    expect(screen.getAllByText('Booting...').length).toBeGreaterThanOrEqual(1)
  })
  it('renders screen container', () => {
    const { container } = render(<V86TerminalPanel />)
    expect(container.querySelector('[data-testid="v86-screen"]')).not.toBeNull()
  })
  it('applies className', () => {
    const { container } = render(<V86TerminalPanel className="h-96" />)
    const el = container.querySelector('[class*="h-96"]')
    expect(el).not.toBeNull()
  })
})
