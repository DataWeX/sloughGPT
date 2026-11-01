import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import Achievements from './Achievements'

vi.mock('@/lib/phoneme-store', () => ({
  usePhonemeStore: (selector: (state: unknown) => unknown) => selector({
    history: [],
    quiz: { total: 0 },
    flashcardSRS: {},
    pronunciationBestStreak: 0,
    challenge: { score: 0 },
  }),
}))

describe('Achievements', () => {
  it('renders without errors', () => {
    render(<Achievements />)
    expect(screen.getByText('Achievements')).toBeDefined()
  })

  it('shows progress counter', () => {
    render(<Achievements />)
    expect(screen.getAllByText(/0\/12/).length).toBeGreaterThan(0)
  })
})
