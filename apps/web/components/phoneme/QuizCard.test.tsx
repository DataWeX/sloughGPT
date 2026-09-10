import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'

vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
  usePathname: () => '/phoneme',
}))

vi.mock('@/lib/phoneme-controller', () => ({
  phonemeController: {
    encode: vi.fn().mockResolvedValue({
      word: 'hello',
      language: 'en',
      phonemes: ['HH', 'EH', 'L', 'OW'],
      ids: [1, 2, 3, 4],
      decoded: 'hello',
    }),
    score: vi.fn().mockResolvedValue({
      target: 'hello',
      spoken: 'helo',
      language: 'en',
      score: 0.75,
      precision: 0.8,
      recall: 0.7,
      target_phonemes: ['HH', 'EH', 'L', 'OW'],
      spoken_phonemes: ['HH', 'EH', 'L', 'OW'],
    }),
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

vi.mock('@/lib/phoneme-store', () => {
  const store = {
    quizScore: 0,
    quizTotal: 0,
    quizStreak: 0,
    quizBestStreak: 0,
    incrementQuizScore: vi.fn(),
    resetQuiz: vi.fn(),
  }
  return {
    usePhonemeStore: vi.fn((selector: any) => selector(store)),
  }
})

vi.mock('@/lib/toast-store', () => ({
  useToastStore: vi.fn((selector: any) => selector({ addToast: vi.fn() })),
}))

describe('QuizCard', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders quiz title', async () => {
    const { default: QuizCard } = await import('./QuizCard')
    render(<QuizCard />)
    expect(screen.getAllByText('Quiz').length).toBeGreaterThan(0)
  })

  it('renders word of the day title', async () => {
    const { default: QuizCard } = await import('./QuizCard')
    render(<QuizCard />)
    expect(screen.getAllByText('Word of the Day').length).toBeGreaterThan(0)
  })

  it('renders language selector', async () => {
    const { default: QuizCard } = await import('./QuizCard')
    render(<QuizCard />)
    expect(screen.getAllByText('English').length).toBeGreaterThan(0)
  })

  it('renders difficulty selector', async () => {
    const { default: QuizCard } = await import('./QuizCard')
    render(<QuizCard />)
    expect(screen.getAllByText('Medium').length).toBeGreaterThan(0)
  })

  it('renders start quiz button', async () => {
    const { default: QuizCard } = await import('./QuizCard')
    render(<QuizCard />)
    expect(screen.getAllByText('Start Quiz').length).toBeGreaterThan(0)
  })

  it('shows initial state message', async () => {
    const { default: QuizCard } = await import('./QuizCard')
    render(<QuizCard />)
    expect(screen.getAllByText('Click Start Quiz to begin').length).toBeGreaterThan(0)
  })

  it('renders check and reveal buttons after starting', async () => {
    const { default: QuizCard } = await import('./QuizCard')
    render(<QuizCard />)
    fireEvent.click(screen.getAllByText('Start Quiz')[0])
    await waitFor(() => {
      expect(screen.getByText('Check')).toBeInTheDocument()
      expect(screen.getByText('Reveal')).toBeInTheDocument()
    })
  })

  it('renders hint phonemes after starting', async () => {
    const { default: QuizCard } = await import('./QuizCard')
    render(<QuizCard />)
    fireEvent.click(screen.getAllByText('Start Quiz')[0])
    await waitFor(() => {
      expect(screen.getByText('Hint — Phonemes:')).toBeInTheDocument()
    })
  })

  it('renders new word button', async () => {
    const { default: QuizCard } = await import('./QuizCard')
    render(<QuizCard />)
    fireEvent.click(screen.getAllByText('Start Quiz')[0])
    await waitFor(() => {
      expect(screen.getByText('New Word')).toBeInTheDocument()
    })
  })

  it('shows empty state when no quiz started', async () => {
    const { default: QuizCard } = await import('./QuizCard')
    render(<QuizCard />)
    expect(screen.queryByText('Check')).not.toBeInTheDocument()
    expect(screen.queryByText('Reveal')).not.toBeInTheDocument()
  })
})
