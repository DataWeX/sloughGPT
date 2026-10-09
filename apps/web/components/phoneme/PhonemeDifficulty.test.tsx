import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import PhonemeDifficulty from './PhonemeDifficulty'

// Mock the store
vi.mock('@/lib/phoneme-store', () => ({
  usePhonemeStore: (selector: (state: unknown) => unknown) => selector({
    history: [],
  }),
}))

describe('PhonemeDifficulty', () => {
  it('renders without errors', () => {
    render(<PhonemeDifficulty />)
    expect(screen.getByText('Phoneme Difficulty')).toBeDefined()
  })

  it('shows empty state when no history', () => {
    render(<PhonemeDifficulty />)
    expect(screen.getAllByText(/Practice more to see/).length).toBeGreaterThan(0)
  })
})
