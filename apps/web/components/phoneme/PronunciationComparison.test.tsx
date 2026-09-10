import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import PronunciationComparison from './PronunciationComparison'

vi.mock('@/lib/phoneme-store', () => ({
  usePhonemeStore: (selector: (state: unknown) => unknown) => selector({
    history: [],
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
  toIPA: vi.fn((phonemes: string[]) => phonemes.map(p => `[${p}]`)),
}))

describe('PronunciationComparison', () => {
  it('renders without errors', () => {
    render(<PronunciationComparison />)
    expect(screen.getByText('Pronunciation Comparison')).toBeDefined()
  })

  it('shows input fields', () => {
    render(<PronunciationComparison />)
    expect(screen.getAllByPlaceholderText('First word...').length).toBeGreaterThan(0)
    expect(screen.getAllByPlaceholderText('Second word...').length).toBeGreaterThan(0)
  })
})
