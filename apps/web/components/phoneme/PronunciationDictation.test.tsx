import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import PronunciationDictation from './PronunciationDictation'

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
}))

describe('PronunciationDictation', () => {
  it('renders without errors', () => {
    render(<PronunciationDictation />)
    expect(screen.getByText('Pronunciation Dictation')).toBeDefined()
  })

  it('shows start button', () => {
    render(<PronunciationDictation />)
    expect(screen.getAllByText(/Start Dictation/).length).toBeGreaterThan(0)
  })
})
