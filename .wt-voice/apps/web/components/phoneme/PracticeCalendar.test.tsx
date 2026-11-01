import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import PracticeCalendar from './PracticeCalendar'

vi.mock('@/lib/phoneme-store', () => ({
  usePhonemeStore: (selector: (state: unknown) => unknown) => selector({
    history: [],
  }),
}))

describe('PracticeCalendar', () => {
  it('renders without errors', () => {
    render(<PracticeCalendar />)
    expect(screen.getByText('Practice Calendar')).toBeDefined()
  })

  it('shows empty state when no history', () => {
    render(<PracticeCalendar />)
    expect(screen.getAllByText(/Start practicing to see/).length).toBeGreaterThan(0)
  })
})
