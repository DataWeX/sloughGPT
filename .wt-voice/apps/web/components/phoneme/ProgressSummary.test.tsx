import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import ProgressSummary from './ProgressSummary'

// Mock the store
vi.mock('@/lib/phoneme-store', () => ({
  usePhonemeStore: (selector: (state: unknown) => unknown) => selector({
    history: [],
    pronunciationBestStreak: 0,
  }),
}))

describe('ProgressSummary', () => {
  it('renders without errors', () => {
    render(<ProgressSummary />)
    expect(screen.getByText('Progress Summary')).toBeDefined()
  })

  it('shows empty state when no history', () => {
    render(<ProgressSummary />)
    expect(screen.getAllByText(/No practice data yet/).length).toBeGreaterThan(0)
  })
})
