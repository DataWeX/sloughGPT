import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import ChallengeCard from './ChallengeCard'

// Mock the store
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
  phonemeController: {
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
}))

describe('ChallengeCard', () => {
  it('renders without errors', () => {
    render(<ChallengeCard />)
    expect(screen.getByText('Daily Challenge')).toBeDefined()
  })

  it('shows challenge date', () => {
    render(<ChallengeCard />)
    const today = new Date().toISOString().split('T')[0]
    expect(screen.getAllByText(today).length).toBeGreaterThan(0)
  })
})
