import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import PhonemePatternQuiz from './PhonemePatternQuiz'

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

describe('PhonemePatternQuiz', () => {
  it('renders without errors', () => {
    render(<PhonemePatternQuiz />)
    expect(screen.getByText('Pattern Recognition Quiz')).toBeDefined()
  })

  it('shows start button', () => {
    render(<PhonemePatternQuiz />)
    expect(screen.getAllByText(/Start Quiz/).length).toBeGreaterThan(0)
  })
})
