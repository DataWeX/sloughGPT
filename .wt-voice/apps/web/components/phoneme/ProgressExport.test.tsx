import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import ProgressExport from './ProgressExport'

vi.mock('@/lib/phoneme-store', () => ({
  usePhonemeStore: (selector: (state: unknown) => unknown) => selector({
    history: [],
  }),
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (selector: (state: unknown) => unknown) => selector({
    addToast: vi.fn(),
  }),
}))

describe('ProgressExport', () => {
  it('renders without errors', () => {
    render(<ProgressExport />)
    expect(screen.getByText('Export / Import')).toBeDefined()
  })

  it('shows export button', () => {
    render(<ProgressExport />)
    expect(screen.getAllByText(/Export/).length).toBeGreaterThan(0)
  })

  it('shows import button', () => {
    render(<ProgressExport />)
    expect(screen.getAllByText(/Import/).length).toBeGreaterThan(0)
  })
})
