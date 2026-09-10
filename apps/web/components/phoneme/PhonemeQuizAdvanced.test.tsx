import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import PhonemeQuizAdvanced from './PhonemeQuizAdvanced'

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
  toIPA: vi.fn((phonemes: string[]) => phonemes.map(p => `[${p}]`)),
}))

describe('PhonemeQuizAdvanced', () => {
  it('renders without errors', () => {
    render(<PhonemeQuizAdvanced />)
    expect(screen.getByText('Advanced Quiz')).toBeDefined()
  })

  it('shows start button', () => {
    render(<PhonemeQuizAdvanced />)
    expect(screen.getAllByText(/Start Quiz/).length).toBeGreaterThan(0)
  })
})
