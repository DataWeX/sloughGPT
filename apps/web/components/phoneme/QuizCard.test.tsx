import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, act } from '@testing-library/react'
import QuizCard from './QuizCard'

vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
  usePathname: () => '/phoneme',
}))

vi.mock('@/lib/phoneme-controller', () => ({
  phonemeController: {
    encode: vi.fn().mockResolvedValue({
      word: 'hello', language: 'en', phonemes: ['HH', 'EH', 'L', 'OW'], ids: [1, 2, 3, 4], decoded: 'hello',
    }),
    score: vi.fn(),
  },
  PHONEME_LANGUAGES: [
    { value: 'en', label: 'English' },
    { value: 'de', label: 'German' },
    { value: 'fr', label: 'French' },
    { value: 'es', label: 'Spanish' },
    { value: 'it', label: 'Italian' },
    { value: 'pt', label: 'Portuguese' },
  ],
  toIPA: vi.fn((phonemes: string[]) => phonemes),
}))

vi.mock('@/lib/phoneme-store', () => ({
  usePhonemeStore: vi.fn((selector: any) => selector({
    quizScore: 0, quizTotal: 0, quizStreak: 0, quizBestStreak: 0,
    incrementQuizScore: vi.fn(), resetQuiz: vi.fn(),
  })),
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: vi.fn((selector: any) => selector({ addToast: vi.fn() })),
}))

describe('QuizCard', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('renders quiz sections', () => {
    render(<QuizCard />)
    expect(screen.getAllByText('Quiz').length).toBeGreaterThan(0)
    expect(screen.getAllByText('Word of the Day').length).toBeGreaterThan(0)
  })

  it('shows idle state before starting', () => {
    render(<QuizCard />)
    expect(screen.getAllByText('Click Start Quiz to begin').length).toBeGreaterThan(0)
    expect(screen.queryByText('Hint — Phonemes:')).not.toBeInTheDocument()
  })

  it('shows game UI after starting', async () => {
    render(<QuizCard />)
    const startBtn = screen.getAllByRole('button').find(b => b.textContent?.includes('Start Quiz'))!

    await act(async () => {
      fireEvent.click(startBtn)
      await new Promise(r => setTimeout(r, 100))
    })

    expect(screen.getByText('Check')).toBeInTheDocument()
    expect(screen.getAllByText('Reveal').length).toBeGreaterThan(0)
    expect(screen.getByText('Hint — Phonemes:')).toBeInTheDocument()
  })

  it('shows new word button after starting', async () => {
    render(<QuizCard />)
    const startBtn = screen.getAllByRole('button').find(b => b.textContent?.includes('Start Quiz'))!

    await act(async () => {
      fireEvent.click(startBtn)
      await new Promise(r => setTimeout(r, 100))
    })

    expect(screen.getAllByText('New Word').length).toBeGreaterThan(0)
  })
})
