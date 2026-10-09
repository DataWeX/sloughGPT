import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'

vi.mock('@/components/shell/TerminalPanel', () => ({
  TerminalPanel: (props: Record<string, unknown>) => <div data-testid="shell-panel">TerminalPanel</div>,
}))

import ShellPage from './page'

describe('ShellPage', () => {
  afterEach(() => cleanup())

  it('renders page title', () => {
    render(<ShellPage />)
    expect(screen.getByText('Shell')).toBeTruthy()
  })

  it('renders TerminalPanel component', () => {
    render(<ShellPage />)
    expect(screen.getByTestId('shell-panel')).toBeTruthy()
  })
})
