import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import WordDifficultyRanker from './WordDifficultyRanker'

vi.mock('@/lib/phoneme-store', () => ({
  usePhonemeStore: (selector: (state: unknown) => unknown) => selector({
    history: [],
  }),
}))

describe('WordDifficultyRanker', () => {
  it('renders without errors', () => {
    render(<WordDifficultyRanker />)
    expect(screen.getByText('Word Difficulty')).toBeDefined()
  })

  it('shows empty state when no history', () => {
    render(<WordDifficultyRanker />)
    expect(screen.getAllByText(/Practice more words/).length).toBeGreaterThan(0)
  })
})
