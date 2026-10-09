import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import PronunciationJournal from './PronunciationJournal'

// Mock the store
vi.mock('@/lib/phoneme-store', () => ({
  usePhonemeStore: (selector: (state: unknown) => unknown) => selector({
    history: [],
  }),
}))

describe('PronunciationJournal', () => {
  it('renders without errors', () => {
    render(<PronunciationJournal />)
    expect(screen.getByText('Pronunciation Journal')).toBeDefined()
  })

  it('shows empty state when no history', () => {
    render(<PronunciationJournal />)
    expect(screen.getAllByText(/No practice data yet/).length).toBeGreaterThan(0)
  })
})
