import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import RhymeFinder from './RhymeFinder'

vi.mock('@/lib/phoneme-store', () => ({
  usePhonemeStore: (selector: (state: unknown) => unknown) => selector({
    addToHistory: vi.fn(),
  }),
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (selector: (state: unknown) => unknown) => selector({
    addToast: vi.fn(),
  }),
}))

vi.mock('@/lib/phoneme-controller', () => ({
  phonemeController: { encode: vi.fn() },
  PHONEME_LANGUAGES: [
    { value: 'en', label: 'English' },
  ],
  toIPA: vi.fn((phonemes: string[]) => phonemes),
}))

describe('RhymeFinder', () => {
  it('renders without errors', () => {
    render(<RhymeFinder />)
    expect(screen.getByText('Rhyme Finder')).toBeDefined()
  })

  it('shows input field', () => {
    render(<RhymeFinder />)
    expect(screen.getAllByPlaceholderText('Enter a word to find rhymes...').length).toBeGreaterThan(0)
  })
})
