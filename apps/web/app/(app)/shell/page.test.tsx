import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'

vi.mock('@/components/shell/ShellPanel', () => ({
  ShellPanel: (props: Record<string, unknown>) => <div data-testid="shell-panel">ShellPanel</div>,
}))

import ShellPage from './page'

describe('ShellPage', () => {
  afterEach(() => cleanup())

  it('renders page title', () => {
    render(<ShellPage />)
    expect(screen.getByText('Shell')).toBeTruthy()
  })

  it('renders ShellPanel component', () => {
    render(<ShellPage />)
    expect(screen.getByTestId('shell-panel')).toBeTruthy()
  })
})
