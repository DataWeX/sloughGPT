import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import WordFamilies from './WordFamilies'

vi.mock('@/lib/phoneme-store', () => ({
  usePhonemeStore: (selector: (state: unknown) => unknown) => selector({
    history: [],
  }),
}))

vi.mock('@/lib/phoneme-controller', () => ({
  toIPA: vi.fn((phonemes: string[]) => phonemes.map(p => `[${p}]`)),
}))

describe('WordFamilies', () => {
  it('renders without errors', () => {
    render(<WordFamilies />)
    expect(screen.getByText('Word Families')).toBeDefined()
  })

  it('shows empty state when no history', () => {
    render(<WordFamilies />)
    expect(screen.getAllByText(/Practice more words/).length).toBeGreaterThan(0)
  })
})
