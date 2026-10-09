import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import MultipleChoiceQuiz from './MultipleChoiceQuiz'

// Mock the store
vi.mock('@/lib/phoneme-store', () => ({
  usePhonemeStore: (selector: (state: unknown) => unknown) => selector({
    quizScore: 0,
    quizTotal: 0,
    quizStreak: 0,
    quizBestStreak: 0,
    incrementQuizScore: vi.fn(),
  }),
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (selector: (state: unknown) => unknown) => selector({
    addToast: vi.fn(),
  }),
}))

vi.mock('@/lib/phoneme-controller', () => ({
  phonemeController: {
    encode: vi.fn(),
  },
  PHONEME_LANGUAGES: [
    { value: 'en', label: 'English' },
    { value: 'de', label: 'German' },
    { value: 'fr', label: 'French' },
    { value: 'es', label: 'Spanish' },
    { value: 'it', label: 'Italian' },
    { value: 'pt', label: 'Portuguese' },
  ],
}))

describe('MultipleChoiceQuiz', () => {
  it('renders without errors', () => {
    render(<MultipleChoiceQuiz />)
    expect(screen.getByText('Multiple Choice Quiz')).toBeDefined()
  })

  it('shows score', () => {
    render(<MultipleChoiceQuiz />)
    expect(screen.getAllByText('0/0').length).toBeGreaterThan(0)
  })
})
