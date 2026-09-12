import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import React from 'react'

vi.mock('next/navigation', () => ({
  usePathname: () => '/phoneme',
  useRouter: () => ({ push: vi.fn() }),
}))

vi.mock('@/hooks/useLocale', () => ({
  useLocale: () => ({ t: (k: string) => k }),
  LOCALES: [],
}))

vi.mock('@/lib/phoneme-store', () => ({
  usePhonemeStore: () => ({ history: [], achievements: [], dailyGoal: 0 }),
}))

vi.mock('@/hooks/usePhonemeShortcuts', () => ({
  usePhonemeShortcuts: vi.fn(),
}))

vi.mock('@/components/phoneme/EncodingCard', () => ({
  default: () => <div data-testid="encoding-card">EncodingCard</div>,
}))

vi.mock('@/components/phoneme/ScoringCard', () => ({
  default: () => <div data-testid="scoring-card">ScoringCard</div>,
}))

vi.mock('@/components/phoneme/ComparisonCard', () => ({
  default: () => <div data-testid="comparison-card">ComparisonCard</div>,
}))

vi.mock('@/components/phoneme/PracticeCard', () => ({
  default: () => <div data-testid="practice-card">PracticeCard</div>,
}))

vi.mock('@/components/phoneme/BatchCard', () => ({
  default: () => <div data-testid="batch-card">BatchCard</div>,
}))

vi.mock('@/components/phoneme/DetectLanguageCard', () => ({
  default: () => <div data-testid="detect-language-card">DetectLanguageCard</div>,
}))

vi.mock('@/components/phoneme/FlashcardCard', () => ({
  default: () => <div data-testid="flashcard-card">FlashcardCard</div>,
}))

vi.mock('@/components/phoneme/ChallengeCard', () => ({
  default: () => <div data-testid="challenge-card">ChallengeCard</div>,
}))

vi.mock('@/components/phoneme/WordOfDayCard', () => ({
  default: () => <div data-testid="word-of-day-card">WordOfDayCard</div>,
}))

vi.mock('@/components/phoneme/QuizCard', () => ({
  default: () => <div data-testid="quiz-card">QuizCard</div>,
}))

vi.mock('@/components/phoneme/MultipleChoiceQuiz', () => ({
  default: () => <div data-testid="multiple-choice-quiz">MultipleChoiceQuiz</div>,
}))

vi.mock('@/components/phoneme/SynthesisCard', () => ({
  default: () => <div data-testid="synthesis-card">SynthesisCard</div>,
}))

vi.mock('@/components/phoneme/HistoryCard', () => ({
  default: () => <div data-testid="history-card">HistoryCard</div>,
}))

vi.mock('@/components/phoneme/ProgressSummary', () => ({
  default: () => <div data-testid="progress-summary">ProgressSummary</div>,
}))

vi.mock('@/components/phoneme/PhonemeDifficulty', () => ({
  default: () => <div data-testid="phoneme-difficulty">PhonemeDifficulty</div>,
}))

vi.mock('@/components/phoneme/PronunciationPatterns', () => ({
  default: () => <div data-testid="pronunciation-patterns">PronunciationPatterns</div>,
}))

vi.mock('@/components/phoneme/PronunciationJournal', () => ({
  default: () => <div data-testid="pronunciation-journal">PronunciationJournal</div>,
}))

vi.mock('@/components/phoneme/MinimalPairsCard', () => ({
  default: () => <div data-testid="minimal-pairs-card">MinimalPairsCard</div>,
}))

vi.mock('@/components/phoneme/RhymeFinder', () => ({
  default: () => <div data-testid="rhyme-finder">RhymeFinder</div>,
}))

vi.mock('@/components/phoneme/PhonemeFrequency', () => ({
  default: () => <div data-testid="phoneme-frequency">PhonemeFrequency</div>,
}))

vi.mock('@/components/phoneme/PhonemeClusters', () => ({
  default: () => <div data-testid="phoneme-clusters">PhonemeClusters</div>,
}))

vi.mock('@/components/phoneme/PronunciationCoach', () => ({
  default: () => <div data-testid="pronunciation-coach">PronunciationCoach</div>,
}))

vi.mock('@/components/phoneme/CustomWordList', () => ({
  default: () => <div data-testid="custom-word-list">CustomWordList</div>,
}))

vi.mock('@/components/phoneme/WordFamilies', () => ({
  default: () => <div data-testid="word-families">WordFamilies</div>,
}))

vi.mock('@/components/phoneme/ProgressExport', () => ({
  default: () => <div data-testid="progress-export">ProgressExport</div>,
}))

vi.mock('@/components/phoneme/Achievements', () => ({
  default: () => <div data-testid="achievements">Achievements</div>,
}))

vi.mock('@/components/phoneme/PronunciationTips', () => ({
  default: () => <div data-testid="pronunciation-tips">PronunciationTips</div>,
}))

vi.mock('@/components/phoneme/DailyGoal', () => ({
  default: () => <div data-testid="daily-goal">DailyGoal</div>,
}))

vi.mock('@/components/phoneme/ProficiencyTracker', () => ({
  default: () => <div data-testid="proficiency-tracker">ProficiencyTracker</div>,
}))

vi.mock('@/components/phoneme/PronunciationDictation', () => ({
  default: () => <div data-testid="pronunciation-dictation">PronunciationDictation</div>,
}))

vi.mock('@/components/phoneme/PhonemePatternQuiz', () => ({
  default: () => <div data-testid="phoneme-pattern-quiz">PhonemePatternQuiz</div>,
}))

vi.mock('@/components/phoneme/WordDifficultyRanker', () => ({
  default: () => <div data-testid="word-difficulty-ranker">WordDifficultyRanker</div>,
}))

vi.mock('@/components/phoneme/PhonemeTypingMode', () => ({
  default: () => <div data-testid="phoneme-typing-mode">PhonemeTypingMode</div>,
}))

vi.mock('@/components/phoneme/PronunciationSpeed', () => ({
  default: () => <div data-testid="pronunciation-speed">PronunciationSpeed</div>,
}))

vi.mock('@/components/phoneme/PronunciationComparison', () => ({
  default: () => <div data-testid="pronunciation-comparison">PronunciationComparison</div>,
}))

vi.mock('@/components/phoneme/PronunciationHeatmap', () => ({
  default: () => <div data-testid="pronunciation-heatmap">PronunciationHeatmap</div>,
}))

vi.mock('@/components/phoneme/ConfusionMatrix', () => ({
  default: () => <div data-testid="confusion-matrix">ConfusionMatrix</div>,
}))

vi.mock('@/components/phoneme/PracticeCalendar', () => ({
  default: () => <div data-testid="practice-calendar">PracticeCalendar</div>,
}))

vi.mock('@/components/phoneme/PronunciationGame', () => ({
  default: () => <div data-testid="pronunciation-game">PronunciationGame</div>,
}))

vi.mock('@/components/phoneme/PhonemeNetwork', () => ({
  default: () => <div data-testid="phoneme-network">PhonemeNetwork</div>,
}))

vi.mock('@/components/phoneme/PronunciationRecording', () => ({
  default: () => <div data-testid="pronunciation-recording">PronunciationRecording</div>,
}))

vi.mock('@/components/phoneme/PhonemeQuizAdvanced', () => ({
  default: () => <div data-testid="phoneme-quiz-advanced">PhonemeQuizAdvanced</div>,
}))

vi.mock('@/components/phoneme/PhonemeReference', () => ({
  default: () => <div data-testid="phoneme-reference">PhonemeReference</div>,
}))

vi.mock('@/components/phoneme/TabErrorBoundary', () => ({
  default: ({ children }: any) => <div>{children}</div>,
}))

import PhonemePage from './page'

describe('PhonemePage', () => {
  it('renders the page with tabs', () => {
    render(<PhonemePage />)
    expect(screen.getByRole('heading', { name: /Phoneme/i })).toBeDefined()
  })

  it('shows the encoding tab by default', () => {
    render(<PhonemePage />)
    expect(screen.getByTestId('encoding-card')).toBeDefined()
  })
})
