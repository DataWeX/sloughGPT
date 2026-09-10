import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import PronunciationPatterns from './PronunciationPatterns'

// Mock the store
vi.mock('@/lib/phoneme-store', () => ({
  usePhonemeStore: (selector: (state: unknown) => unknown) => selector({
    history: [],
  }),
}))

describe('PronunciationPatterns', () => {
  it('renders without errors', () => {
    render(<PronunciationPatterns />)
    expect(screen.getByText('Pronunciation Patterns')).toBeDefined()
  })

  it('shows empty state when no history', () => {
    render(<PronunciationPatterns />)
    expect(screen.getAllByText(/Practice more to see/).length).toBeGreaterThan(0)
  })
})
