import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import DailyGoal from './DailyGoal'

vi.mock('@/lib/phoneme-store', () => ({
  usePhonemeStore: (selector: (state: unknown) => unknown) => selector({
    history: [],
  }),
}))

describe('DailyGoal', () => {
  it('renders without errors', () => {
    render(<DailyGoal />)
    expect(screen.getByText('Daily Goal')).toBeDefined()
  })

  it('shows progress counter', () => {
    render(<DailyGoal />)
    expect(screen.getAllByText(/0\/10/).length).toBeGreaterThan(0)
  })
})
